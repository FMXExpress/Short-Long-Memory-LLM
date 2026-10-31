unit Mem.Priors;

(* Priors: the user's standing instructions about how to answer ("from now on
   keep answers short", "never use emojis", "I prefer code examples"), kept
   apart from facts about the user. Facts are retrieved when relevant; the
   most confident priors are in every system prompt.

   A prior's confidence is the share of evidence for it,
     confidence = alpha / (alpha + beta),
   where alpha and beta start as pseudo-counts and each user message adds one
   to either side at most once (TMemStore.AddPriorEvidence). The starting
   counts are its inertia: a prior the user set with /prior (3:1) takes more
   contradiction to fall than one picked up from a remark (1:1 + 1). This
   replaces the "LLM proposes a new confidence, apply delta * volatility"
   scheme of observer-style prior trackers, where the same evidence is read
   again on every turn and the value drifts to whatever the observer asks.

   This unit is the rules side: it finds instructions in a message and
   decides whether a new one confirms or contradicts a stored one. A
   standing instruction needs a marker (from now on, always, never, don't +
   an answering verb, I prefer + an answer word); a one-off request ("use
   Python for this") is not one. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, RegExpr;

type
  TPriorCandidate = record
    Belief: string;     { as it goes into the prompt: "Never use emojis." }
    Key: string;        { its content words, sorted, without negators }
    Polarity: Integer;  { +1 do, -1 don't }
  end;
  TPriorCandidates = array of TPriorCandidate;

  TPriorMatch = (pmNone, pmSame, pmOpposite);

{ the standing instructions in AText, one per sentence at most }
function DetectPriors(const AText: string): TPriorCandidates;
{ a prior from /prior <text>: the text as given }
function PriorFromText(const AText: string): TPriorCandidate;
{ content-word key and polarity of a belief }
function PriorKey(const ABelief: string; out APolarity: Integer): string;
{ the same instruction (pmSame), its negation (pmOpposite), or unrelated }
function MatchPrior(const AKeyA: string; APolA: Integer; const AKeyB: string; APolB: Integer): TPriorMatch;

implementation

uses
  Mem.Extract;

const
  NEGATORS: array[0..7] of string = ('never', 'dont', 'not', 'no', 'stop', 'avoid', 'without', 'nt');
  FILLER: array[0..26] of string = ('please', 'always', 'from', 'now', 'on', 'going', 'forward', 'in', 'the',
    'future', 'you', 'your', 'a', 'an', 'to', 'do', 'should', 'be', 'i', 'prefer', 'it', 'when', 'user', 'can',
    'feel', 'free', 'fine');
  { what an instruction about answering talks about }
  ANSWER_WORDS: array[0..27] of string = ('answer', 'answers', 'reply', 'replies', 'response', 'responses',
    'explanation', 'explanations', 'code', 'comments', 'comment', 'example', 'examples', 'summary',
    'summaries', 'bullet', 'bullets', 'tone', 'format', 'units', 'language', 'emoji', 'emojis', 'markdown',
    'sentence', 'sentences', 'words', 'jargon');
  { verbs a "don't ..." instruction to the assistant starts with }
  ANSWER_VERBS: array[0..17] of string = ('use', 'say', 'call', 'mention', 'apologize', 'apologise',
    'repeat', 'explain', 'include', 'add', 'write', 'answer', 'reply', 'respond', 'give', 'start', 'end',
    'ask');

function InList(const S: string; const AList: array of string): Boolean;
var
  W: string;
begin
  for W in AList do
    if S = W then
      Exit(True);
  Result := False;
end;

{ lowercase words, apostrophes dropped (don't -> dont) }
function Words(const S: string): TStringList;
var
  I: Integer;
  W: string;
  C: Char;
begin
  Result := TStringList.Create;
  W := '';
  for I := 1 to Length(S) + 1 do
  begin
    if I <= Length(S) then
      C := LowerCase(S[I])
    else
      C := ' ';
    if C in ['a'..'z', '0'..'9', #128..#255] then
      W := W + C
    else if C = '''' then
      Continue
    else
    begin
      if W <> '' then
        Result.Add(W);
      W := '';
    end;
  end;
end;

{ using -> us, uses -> us, replies -> reply: crude, but the same on both
  sides of a comparison, which is all a word-set match needs }
function Stem(const W: string): string;
begin
  Result := W;
  if (Length(Result) > 4) and EndsStr('ing', Result) then
    SetLength(Result, Length(Result) - 3)
  else if (Length(Result) > 4) and EndsStr('ies', Result) then
    Result := Copy(Result, 1, Length(Result) - 3) + 'y'
  else if (Length(Result) > 3) and EndsStr('es', Result) and not EndsStr('ses', Result) then
    SetLength(Result, Length(Result) - 2)
  else if (Length(Result) > 3) and EndsStr('s', Result) and not EndsStr('ss', Result) then
    SetLength(Result, Length(Result) - 1);
  if (Length(Result) > 2) and EndsStr('e', Result) then
    SetLength(Result, Length(Result) - 1);
end;

function StemIn(const W: string; const AList: array of string): Boolean;
var
  X: string;
begin
  for X in AList do
    if Stem(LowerCase(W)) = Stem(X) then
      Exit(True);
  Result := False;
end;

function HasAny(const S: string; const AList: array of string): Boolean;
var
  L: TStringList;
  I: Integer;
begin
  L := Words(S);
  try
    for I := 0 to L.Count - 1 do
      if InList(L[I], AList) then
        Exit(True);
    Result := False;
  finally
    L.Free;
  end;
end;

function FirstWord(const S: string): string;
var
  L: TStringList;
begin
  L := Words(S);
  try
    if L.Count > 0 then
      Result := L[0]
    else
      Result := '';
  finally
    L.Free;
  end;
end;

function PriorKey(const ABelief: string; out APolarity: Integer): string;
var
  L, LKey: TStringList;
  I: Integer;
  W: string;
begin
  APolarity := 1;
  L := Words(ABelief);
  LKey := TStringList.Create;
  try
    LKey.Sorted := True;
    LKey.Duplicates := dupIgnore;
    for I := 0 to L.Count - 1 do
    begin
      W := L[I];
      if InList(W, NEGATORS) or (W = 'dont') or (W = 'doesnt') then
      begin
        APolarity := -APolarity;
        Continue;
      end;
      if InList(W, FILLER) then
        Continue;
      W := Stem(W);
      if InList(W, FILLER) then
        Continue;
      LKey.Add(W);
    end;
    Result := '';
    for I := 0 to LKey.Count - 1 do
      Result := Result + IfThen(Result = '', '', ' ') + LKey[I];
  finally
    LKey.Free;
    L.Free;
  end;
end;

function MatchPrior(const AKeyA: string; APolA: Integer; const AKeyB: string; APolB: Integer): TPriorMatch;
var
  A, B: TStringList;
  I, LBoth: Integer;
begin
  Result := pmNone;
  A := TStringList.Create;
  B := TStringList.Create;
  try
    A.Delimiter := ' ';
    A.DelimitedText := AKeyA;
    B.Delimiter := ' ';
    B.DelimitedText := AKeyB;
    if (A.Count = 0) or (B.Count = 0) then
      Exit;
    LBoth := 0;
    for I := 0 to A.Count - 1 do
      if B.IndexOf(A[I]) >= 0 then
        Inc(LBoth);
    { Jaccard similarity of the content words }
    if LBoth / (A.Count + B.Count - LBoth) < 0.6 then
      Exit;
    if APolA = APolB then
      Result := pmSame
    else
      Result := pmOpposite;
  finally
    A.Free;
    B.Free;
  end;
end;

function Capital(const S: string): string;
begin
  Result := Trim(S);
  if Result <> '' then
    Result[1] := UpCase(Result[1]);
  if (Result <> '') and not (Result[Length(Result)] in ['.', '!']) then
    Result := Result + '.';
end;

function Candidate(const ABelief: string): TPriorCandidate;
begin
  Result.Belief := Capital(ABelief);
  Result.Key := PriorKey(Result.Belief, Result.Polarity);
end;

function PriorFromText(const AText: string): TPriorCandidate;
begin
  Result := Candidate(AText);
end;

function DetectPriors(const AText: string): TPriorCandidates;
var
  RFromNow, RAlwaysNever, RDont, RStop, RPrefer, RPlease, RAllow: TRegExpr;
  LPiece, S, LRest: string;
  LFound: Boolean;
  C: TPriorCandidate;
begin
  Result := nil;
  RPlease := TRegExpr.Create('^(?:please|pls|and|also|ok|okay|so|actually)[,]?\s+');
  { the reverse of a "don't": it contradicts one, or stands as a prior itself }
  RAllow := TRegExpr.Create('^(?:you can|you may|it''s fine to|it is fine to|feel free to|go ahead and)\s+(.+)$');
  RFromNow := TRegExpr.Create('^(?:from now on|going forward|in (?:the )?future|for the rest of (?:this|our) ' +
    '(?:chat|conversation)|next time)[,:]?\s+(.+)$');
  RAlwaysNever := TRegExpr.Create('^(always|never)\s+(.+)$');
  RDont := TRegExpr.Create('^(?:don''t|dont|do not)\s+(.+)$');
  RStop := TRegExpr.Create('^stop\s+(\w+ing)\b(.*)$');
  RPrefer := TRegExpr.Create('\bi(?: would|''d)? (?:prefer|like|want)\s+(?:it when you\s+|you to\s+|that you\s+)?(.+)$');
  try
    RPlease.ModifierI := True;
    RFromNow.ModifierI := True;
    RAlwaysNever.ModifierI := True;
    RDont.ModifierI := True;
    RStop.ModifierI := True;
    RPrefer.ModifierI := True;
    RAllow.ModifierI := True;
    for LPiece in Sentences(AText) do
    begin
      S := Trim(LPiece);
      if (S = '') or (S[Length(S)] = '?') then
        Continue;
      while (S <> '') and (S[Length(S)] in ['.', '!', ' ']) do
        SetLength(S, Length(S) - 1);
      while RPlease.Exec(S) do
        S := Copy(S, RPlease.MatchLen[0] + 1, MaxInt);
      LFound := False;
      if RFromNow.Exec(S) then
      begin
        LRest := RFromNow.Match[1];
        while RPlease.Exec(LRest) do
          LRest := Copy(LRest, RPlease.MatchLen[0] + 1, MaxInt);
        C := Candidate(LRest);
        LFound := C.Key <> '';
      end
      else if RAlwaysNever.Exec(S) and not SameText(Trim(RAlwaysNever.Match[2]), 'mind') then
      begin
        C := Candidate(S);
        LFound := C.Key <> '';
      end
      else if RDont.Exec(S) then
      begin
        LRest := RDont.Match[1];
        if StemIn(FirstWord(LRest), ANSWER_VERBS) or HasAny(LRest, ANSWER_WORDS) then
        begin
          C := Candidate('Don''t ' + LRest);
          LFound := C.Key <> '';
        end;
      end
      else if RStop.Exec(S) then
      begin
        C := Candidate('Avoid ' + RStop.Match[1] + RStop.Match[2]);
        LFound := (C.Key <> '') and (HasAny(S, ANSWER_WORDS) or StemIn(RStop.Match[1], ANSWER_VERBS));
      end
      else if RAllow.Exec(S) then
      begin
        LRest := RAllow.Match[1];
        if StemIn(FirstWord(LRest), ANSWER_VERBS) or HasAny(LRest, ANSWER_WORDS) then
        begin
          C := Candidate(LRest);
          LFound := C.Key <> '';
        end;
      end
      else if RPrefer.Exec(S) and HasAny(RPrefer.Match[1], ANSWER_WORDS) then
      begin
        LRest := RPrefer.Match[1];
        if Pos('you', LowerCase(Copy(S, RPrefer.MatchPos[0], RPrefer.MatchPos[1] - RPrefer.MatchPos[0]))) > 0 then
          C := Candidate(LRest)
        else
          C := Candidate('The user prefers ' + LRest);
        LFound := C.Key <> '';
      end;
      if LFound then
      begin
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)] := C;
      end;
    end;
  finally
    RPlease.Free;
    RFromNow.Free;
    RAlwaysNever.Free;
    RDont.Free;
    RStop.Free;
    RPrefer.Free;
    RAllow.Free;
  end;
end;

end.
