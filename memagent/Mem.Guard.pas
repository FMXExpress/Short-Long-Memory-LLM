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

{ is the stored fact about what the question asks? }
function SameThing(ARet: TRetrieval; const AQuestion, AQSubject: string; const AFact: TFactRec;
  AMinScore: Single): Boolean;
var
  QW, FW: TStringList;
  I, J: Integer;
begin
  { shared content word: "dog" in "my dog's name", "aunt" vs "aunt" }
  QW := Words(AQSubject);
  FW := Words(AFact.Subject);
  try
    for I := 0 to QW.Count - 1 do
      if not InList(Stem(QW[I]), Weak) and (Length(Stem(QW[I])) >= 3) then
        for J := 0 to FW.Count - 1 do
          if Stem(QW[I]) = Stem(FW[J]) then
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
