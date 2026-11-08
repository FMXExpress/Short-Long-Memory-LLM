unit Mem.Guard;

(* Licensing guard against stored-value confabulation.

   Measured failure (bench/README.md): asked about something never said, the
   model answers with a value the user gave for a DIFFERENT subject: "What's
   my horse's name?" -> "Biscuit" (the dog), "What's my aunt called?" -> "Ana"
   (the sister). Fine-tuning does not fix this at small scale (SFT negatives,
   DPO: abstention does not generalize); a licensing check against the fact
   store can (cf. "Stemming Hallucination in Language Models Using a Licensing
   Oracle", arXiv 2511.06073).

   The check: if the question names a subject of the user ("my horse's name")
   and the draft answer contains stored values, every one of those values
   must belong to a fact about that subject. Otherwise the answer is not
   licensed. Questions without an explicit "my X" are left alone, as are
   answers that use no stored value: this guards against leaking the wrong
   memory, not against inventing new values. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, Mem.Store, Mem.Retrieval;

const
  ABSTAIN_REPLY = 'You haven''t told me that yet.';

{ the possessive subject of a question: "What's my horse's name?" -> "horse's name",
  "Where does my aunt live?" -> "aunt"; '' when the question has none }
function QuestionSubject(const AQuestion: string): string;

(* True when a fact is about someone the question does not mention: "my
   manager's employer" for "Who is my employer?". A possessive followed by
   "name" is the fact's topic ("my partner's name"), not an owner. *)
function ForeignOwner(const AQuestion, ASubject: string): Boolean;

(* True when two fact subjects of the form "<owner>'s <attribute>" name
   different slots: another owner ("Ana's pet" / "Mia's pet", "my grandma's
   name" / "my grandpa's name") or another attribute of the same owner
   ("Tomas's job" / "Tomas's city"). Subjects without that form are left to
   the caller. Measured (bench/README.md, integration set): the cross-encoder
   alone merged 14 of 112 such pairs, and a merge overwrites the older fact. *)
function DistinctSubjects(const A, B: string): Boolean;

{ True when the answer may stand. ABlockedBy names the first stored fact whose
  value the answer used for a different subject. }
function Licensed(ARet: TRetrieval; const AQuestion, AAnswer: string; const AFacts: TFactArray;
  out ABlockedBy: string; AMinScore: Single = 0.03): Boolean;

implementation

const
  { words that end the noun phrase after "my": verbs and fillers }
  StopAfter: array[0..33] of string = ('called', 'named', 'live', 'lives', 'work', 'works', 'do', 'does',
    'play', 'plays', 'born', 'own', 'go', 'goes', 'drive', 'like', 'likes', 'love', 'eat', 'prefer',
    'again', 'is', 'was', 'are', 'for', 'at', 'in', 'on', 'from', 'to', 'and', 'or', 'with', 'about');
  (* words that name the same thing in a question about the user: "my
     employer" asks about "where I work", "my boss" about "my manager", "my
     girl" about "my daughter". Measured (bench/README.md, memory loop 2): the
     guard withheld these right answers because no word matched and the
     cross-encoder scored the pairs under its threshold. None of the
     look-alikes the guard exists for (horse/dog, aunt/sister, stepmother/
     mother, parrot/cat) shares a group. *)
  Related: array[0..9] of string = (
    ' work employer company job salary workplace ',
    ' boss manager supervisor ',
    ' daughter girl ',
    ' son boy ',
    ' mother mom mum mama ',
    ' father dad papa ',
    ' partner wife husband spouse girlfriend boyfriend fiance fiancee ',
    ' kid kids child children ',
    ' home live house ',
    ' car vehicle ');
  { words that carry no subject on their own }
  Weak: array[0..6] of string = ('name', 'favorite', 'favourite', 'the', 'my', 'what', 'who');

function Words(const S: string): TStringList;
var
  I: Integer;
  T: string;
begin
  T := LowerCase(S);
  for I := 1 to Length(T) do
    if not (T[I] in ['a'..'z', '0'..'9', '''', #128..#255]) then
      T[I] := ' ';
  Result := TStringList.Create;
  Result.Delimiter := ' ';
  Result.StrictDelimiter := True;
  Result.DelimitedText := T;
  for I := Result.Count - 1 downto 0 do
    if Result[I] = '' then
      Result.Delete(I);
end;

function InList(const W: string; const AList: array of string): Boolean;
var
  S: string;
begin
  for S in AList do
    if W = S then
      Exit(True);
  Result := False;
end;

function QuestionSubject(const AQuestion: string): string;
var
  W: TStringList;
  I, J: Integer;
begin
  Result := '';
  W := Words(AQuestion);
  try
    for I := 0 to W.Count - 2 do
      if W[I] = 'my' then
      begin
        J := I + 1;
        while (J < W.Count) and (J <= I + 3) and not InList(W[J], StopAfter) do
        begin
          Result := Result + IfThen(Result = '', '', ' ') + W[J];
          Inc(J);
        end;
        Exit;
      end;
  finally
    W.Free;
  end;
end;

function Stem(const W: string): string;
begin
  Result := W;
  if (Length(Result) > 2) and (Copy(Result, Length(Result) - 1, 2) = '''s') then
    SetLength(Result, Length(Result) - 2);
end;

function SameGroup(const A, B: string): Boolean;
var
  G: string;
begin
  for G in Related do
    if (Pos(' ' + A + ' ', G) > 0) and (Pos(' ' + B + ' ', G) > 0) then
      Exit(True);
  Result := False;
end;

function ForeignOwner(const AQuestion, ASubject: string): Boolean;
var
  QW, FW: TStringList;
  I, J: Integer;
  Owner: string;
  Found: Boolean;
begin
  Result := False;
  FW := Words(ASubject);
  QW := Words(AQuestion);
  try
    for I := 0 to FW.Count - 2 do
    begin
      Owner := Stem(FW[I]);
      if (Owner = FW[I]) or InList(Owner, Weak) or (FW[I + 1] = 'name') then
        Continue;
      Found := False;
      for J := 0 to QW.Count - 1 do
        if (Stem(QW[J]) = Owner) or SameGroup(Stem(QW[J]), Owner) then
          Found := True;
      if not Found then
        Exit(True);
    end;
  finally
    FW.Free;
    QW.Free;
  end;
end;

{ "<owner>'s <attribute>": the owner's stem and the attribute's last word; False if not that form }
function OwnerSlot(const ASubject: string; out AOwner, AAttr: string): Boolean;
var
  W: TStringList;
  I: Integer;
begin
  Result := False;
  W := Words(ASubject);
  try
    for I := 0 to W.Count - 2 do
      if Stem(W[I]) <> W[I] then
      begin
        AOwner := Stem(W[I]);
        AAttr := W[W.Count - 1];
        Exit(True);
      end;
  finally
    W.Free;
  end;
end;

function DistinctSubjects(const A, B: string): Boolean;
var
  OA, AA, OB, AB: string;
begin
  if not OwnerSlot(A, OA, AA) or not OwnerSlot(B, OB, AB) then
    Exit(False);
  Result := ((OA <> OB) and not SameGroup(OA, OB)) or ((AA <> AB) and not SameGroup(AA, AB));
end;

{ is the stored fact about what the question asks? }
function SameThing(ARet: TRetrieval; const AQuestion, AQSubject: string; const AFact: TFactRec;
  AMinScore: Single): Boolean;
var
  QW, FW: TStringList;
  I, J: Integer;
  Owner: string;
  OwnerOk: Boolean;
begin
  { shared content word: "dog" in "my dog's name", "aunt" vs "aunt" }
  QW := Words(AQSubject);
  FW := Words(AFact.Subject);
  try
    (* an owner in the question must be the fact's too: "my partner's
       birthday" is not answered by "my birthday" (memory loop 5) *)
    Owner := '';
    for I := 0 to QW.Count - 2 do
      if (Stem(QW[I]) <> QW[I]) and not InList(Stem(QW[I]), Weak) then
      begin
        Owner := Stem(QW[I]);
        Break;
      end;
    OwnerOk := Owner = '';
    for J := 0 to FW.Count - 1 do
      if (Owner <> '') and ((Stem(FW[J]) = Owner) or SameGroup(Owner, Stem(FW[J]))) then
        OwnerOk := True;
    if OwnerOk then
      for I := 0 to QW.Count - 1 do
        if not InList(Stem(QW[I]), Weak) and (Length(Stem(QW[I])) >= 3) then
          for J := 0 to FW.Count - 1 do
            if (Stem(QW[I]) = Stem(FW[J])) or SameGroup(Stem(QW[I]), Stem(FW[J])) then
              Exit(True);
  finally
    QW.Free;
    FW.Free;
  end;
  (* A synonym ("my boy" for "my son's name", "my boss" for "my manager") needs
     the cross-encoder to read question and fact together. Measured on
     bench/facts.json: bge-reranker-base scores real paraphrases up to 0.98
     (median 0.14) and look-alikes ("horse's name" vs the dog fact) at most
     0.03; ms-marco cannot separate the two groups at any threshold. *)
  Result := ARet.HasReranker and
    (ARet.RerankScores(AQuestion, [AFact.Subject + ': ' + AFact.Text])[0] >= AMinScore);
end;

function HasValue(const AAnswer, AValue: string): Boolean;
var
  A, V: TStringList;
  I, J: Integer;
  Hit: Boolean;
begin
  A := Words(AAnswer);
  V := Words(AValue);
  try
    Result := False;
    if V.Count = 0 then
      Exit;
    for I := 0 to A.Count - V.Count do
    begin
      Hit := True;
      for J := 0 to V.Count - 1 do
        if Stem(A[I + J]) <> Stem(V[J]) then
        begin
          Hit := False;
          Break;
        end;
      if Hit then
        Exit(True);
    end;
  finally
    A.Free;
    V.Free;
  end;
end;

(* Binding check, the source-monitoring half of licensing: a value stored
   about a named person ("Sofia's city is Bologna") may answer a question
   about "my sister" only if the sister is Sofia. Measured (bench/README.md,
   integration set, small profile, retrieval only): 15 of 24 two-step
   questions ("Which city does my sister live in?") were answered with
   another person's value ("Hana lives in Bologna"), and the check below
   passed them, because the sister's own name in the answer is licensed.

   The people a value may belong to: the question's person ("my sister") when
   that person has a stored name, else the user's people the answer names
   ("Your cousin Lucia works as a nurse" for "Which of my relatives is a
   nurse?", when the nurse is Mia), plus anyone the question names itself.
   The answer is unbound if it uses a value held only by other named people. *)
function Unbound(const AQuestion, AQSubject, AAnswer: string; const AFacts: TFactArray;
  out ABlockedBy: string): Boolean;
var
  QW: TStringList;
  Rel, Name, Owner, Attr, OtherBy: string;
  I, J: Integer;
  Bound, Other: Boolean;
begin
  Result := False;
  ABlockedBy := '';
  QW := Words(AQSubject);
  try
    if QW.Count = 0 then
      Exit;
    Rel := Stem(QW[0]);
  finally
    QW.Free;
  end;
  { Name: ' name ' for the question's person, else ' a b ' for the user's people named in the answer }
  Name := '';
  for I := 0 to High(AFacts) do
    if OwnerSlot(AFacts[I].Subject, Owner, Attr) and (Attr = 'name') and
      SameText(Copy(Trim(AFacts[I].Subject), 1, 3), 'my ') and ((Owner = Rel) or SameGroup(Owner, Rel)) then
      Name := ' ' + LowerCase(Trim(AFacts[I].Value)) + ' ';
  if Name = '' then
    for I := 0 to High(AFacts) do
      if OwnerSlot(AFacts[I].Subject, Owner, Attr) and (Attr = 'name') and
        SameText(Copy(Trim(AFacts[I].Subject), 1, 3), 'my ') and HasValue(AAnswer, AFacts[I].Value) then
        Name := IfThen(Name = '', ' ', Name) + LowerCase(Trim(AFacts[I].Value)) + ' ';
  if Name = '' then
    Exit;
  QW := Words(AQuestion);
  try
    for I := 0 to QW.Count - 1 do
      QW[I] := Stem(QW[I]);
    for I := 0 to High(AFacts) do
    begin
      { values stored about named people (not "my ..."), used in the answer }
      if SameText(Copy(Trim(AFacts[I].Subject), 1, 3), 'my ') or not OwnerSlot(AFacts[I].Subject, Owner, Attr) or
        not HasValue(AAnswer, AFacts[I].Value) then
        Continue;
      Bound := False;
      Other := False;
      OtherBy := '';
      for J := 0 to High(AFacts) do
        if SameText(Trim(AFacts[J].Value), Trim(AFacts[I].Value)) and
          not SameText(Copy(Trim(AFacts[J].Subject), 1, 3), 'my ') and OwnerSlot(AFacts[J].Subject, Owner, Attr) then
        begin
          if (Pos(' ' + Owner + ' ', Name) > 0) or (QW.IndexOf(Owner) >= 0) then
            Bound := True
          else if OtherBy = '' then
            OtherBy := AFacts[J].Text;
          Other := True;
        end;
      if Other and not Bound then
      begin
        ABlockedBy := OtherBy;
        Exit(True);
      end;
    end;
  finally
    QW.Free;
  end;
end;

(* A derived fact ("Your grandpa Zoe's city is Oslo.", subject "my grandpa's city")
   holds the whole link the guard's rules otherwise piece together, so it licenses an
   answer that matches it: the question names the value and the answer the person or
   relation ("Which of my relatives lives in Oslo?" "Zoe lives in Oslo."), or the
   question names the relation and the answer the value ("Which city does my grandpa
   live in?" "Oslo."). Measured (bench/README.md, derived facts): with derived facts
   in memory, all 11 answers the rule guard withheld were right, and 10 of them were
   "which relative" answers a derived fact backed. *)
function DerivedLicenses(const AQuestion, AAnswer: string; const AFact: TFactRec): Boolean;
var
  LSubj, LRel, LText, LName: string;
  P: Integer;
begin
  Result := False;
  if AFact.Kind <> 'derived' then
    Exit;
  LSubj := AFact.Subject;   { my <rel>'s <attr> }
  P := Pos('''s ', LSubj);
  if (P < 5) or not SameText(Copy(LSubj, 1, 3), 'my ') then
    Exit;
  LRel := Copy(LSubj, 4, P - 4);
  LText := AFact.Text;      { Your <rel> <Name>'s <attr> ... }
  LName := Copy(LText, Length('Your ' + LRel + ' ') + 1, MaxInt);
  P := Pos('''s ', LName);
  if P < 2 then
    Exit;
  LName := Copy(LName, 1, P - 1);
  if HasValue(AQuestion, AFact.Value) and (HasValue(AAnswer, LName) or HasValue(AAnswer, LRel)) then
    Exit(True);
  if HasValue(AQuestion, LRel) and HasValue(AAnswer, AFact.Value) then
    Exit(True);
end;

function Licensed(ARet: TRetrieval; const AQuestion, AAnswer: string; const AFacts: TFactArray;
  out ABlockedBy: string; AMinScore: Single = 0.03): Boolean;
var
  QS: string;
  I: Integer;
  AnyValue, AnyLicensed: Boolean;
begin
  ABlockedBy := '';
  Result := True;
  QS := QuestionSubject(AQuestion);
  if QS = '' then
    Exit;
  for I := 0 to High(AFacts) do
    if DerivedLicenses(AQuestion, AAnswer, AFacts[I]) then
      Exit(True);
  if Unbound(AQuestion, QS, AAnswer, AFacts, ABlockedBy) then
    Exit(False);
  AnyValue := False;
  AnyLicensed := False;
  for I := 0 to High(AFacts) do
    if HasValue(AAnswer, AFacts[I].Value) then
    begin
      AnyValue := True;
      if SameThing(ARet, AQuestion, QS, AFacts[I], AMinScore) then
        AnyLicensed := True
      else if ABlockedBy = '' then
        ABlockedBy := AFacts[I].Text;
    end;
  Result := not AnyValue or AnyLicensed;
end;

end.
