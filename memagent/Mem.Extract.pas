unit Mem.Extract;

(* Fact extraction in Pascal: a port of trainer/memlib.py (extract_facts_rules,
   has_statement, the LLM prompt, parse_json_objects, grounded) for engines
   that run the model in-process. Same output as the Python side, checked by
   memagent/tests/extract_test.lpr against cases dumped from memlib.

   Facts are JSON objects {text, subject, value, questions, confidence, kind},
   the shape TMemStore.AddFact takes. Text is UTF-8; non-ASCII bytes count as
   word characters, as Python's \w counts non-ASCII letters. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, RegExpr, fpjson, jsonparser;

const
  LLM_EXTRACT_PROMPT =
    'Extract durable facts about the user or their world from the message below.'#10 +
    'Only facts worth remembering for weeks (names, relations, places, jobs, dates, preferences).'#10 +
    'Ignore questions and requests; they are not facts.'#10 +
    'For each fact output one compact JSON object on its own line, and nothing else:'#10 +
    '{"text": "<the fact as a sentence addressed to the user>", "subject": "<what it is about, from the ' +
    'user''s side, e.g. my sister''s city>", "value": "<the key detail, copied exactly from the message>"}'#10 +
    'If there are no durable facts, output nothing.'#10#10 +
    'Message:'#10;

{ caller frees the results }
function ExtractFactsRules(const AText: string): TJSONArray;
function FactsFromLlmOutput(const AOut, ASource: string): TJSONArray;
function HasStatement(const AText: string): Boolean;
function Grounded(const AValue, ASource: string): Boolean;
function ParseJsonObjects(const AOut: string): TJSONArray;
function SubjectQuestions(const ASubject: string): TJSONArray;
function Sentences(const AText: string): TStringArray;

implementation

const
  WS = [' ', #9, #10, #11, #12, #13];
  StopValues: array[0..7] of string = ('it', 'this', 'that', 'there', 'here', 'a', 'an', 'the');
  Abbrevs: array[0..11] of string = ('dr', 'mr', 'mrs', 'ms', 'prof', 'st', 'jr', 'sr', 'vs', 'etc', 'e.g', 'i.e');
  LeadWords: array[0..7] of string = ('a', 'an', 'the', 'on', 'at', 'in', 'called', 'named');

function IsWordChar(C: Char): Boolean; inline;
begin
  Result := C in ['a'..'z', 'A'..'Z', '0'..'9', '_', #128..#255];
end;

function Min(A, B: Double): Double; inline;
begin
  if A < B then
    Result := A
  else
    Result := B;
end;

function InList(const S: string; const AList: array of string): Boolean;
var
  T: string;
begin
  for T in AList do
    if S = T then
      Exit(True);
  Result := False;
end;

function Rx(const AExpr: string; ACaseless: Boolean): TRegExpr;
begin
  Result := TRegExpr.Create;
  Result.WordChars := RegExprWordChars;
  { UTF-8 continuation and lead bytes: letters such as "é" stay inside a word }
  Result.WordChars := Result.WordChars + #128#129#130#131#132#133#134#135#136#137#138#139#140#141#142#143 +
    #144#145#146#147#148#149#150#151#152#153#154#155#156#157#158#159#160#161#162#163#164#165#166#167 +
    #168#169#170#171#172#173#174#175#176#177#178#179#180#181#182#183#184#185#186#187#188#189#190#191 +
    #192#193#194#195#196#197#198#199#200#201#202#203#204#205#206#207#208#209#210#211#212#213#214#215 +
    #216#217#218#219#220#221#222#223#224#225#226#227#228#229#230#231#232#233#234#235#236#237#238#239 +
    #240#241#242#243#244#245#246#247#248#249#250#251#252#253#254#255;
  Result.ModifierI := ACaseless;
  Result.ModifierS := False;   { "." stops at a newline, as in Python }
  Result.ModifierM := False;
  Result.Expression := AExpr;
end;

{ Python's str.strip(chars) }
function StripChars(const S: string; const AChars: TSysCharSet): string;
var
  A, B: Integer;
begin
  A := 1;
  B := Length(S);
  while (A <= B) and (S[A] in AChars) do
    Inc(A);
  while (B >= A) and (S[B] in AChars) do
    Dec(B);
  Result := Copy(S, A, B - A + 1);
end;

{ the first N code points of a UTF-8 string (Python's s[:n]) }
function Utf8Head(const S: string; N: Integer): string;
var
  I, LCount: Integer;
begin
  LCount := 0;
  I := 1;
  while I <= Length(S) do
  begin
    if (Byte(S[I]) and $C0) <> $80 then
    begin
      if LCount = N then
        Exit(Copy(S, 1, I - 1));
      Inc(LCount);
    end;
    Inc(I);
  end;
  Result := S;
end;

function Sentences(const AText: string): TStringArray;
var
  I, J, LStart: Integer;
  LTok: string;

  function Protected(AAt: Integer): Boolean;
  var
    K: Integer;
  begin
    { "Dr." "e.g.": the word before the dot is an abbreviation }
    K := AAt - 1;
    while (K >= 1) and (IsWordChar(AText[K]) or (AText[K] = '.')) do
      Dec(K);
    LTok := LowerCase(Copy(AText, K + 1, AAt - K - 1));
    while (LTok <> '') and (LTok[1] = '.') do
      Delete(LTok, 1, 1);
    Result := InList(LTok, Abbrevs);
  end;

  procedure Emit(AEnd: Integer);
  begin
    SetLength(Result, Length(Result) + 1);
    Result[High(Result)] := Copy(AText, LStart, AEnd - LStart);
  end;

begin
  (* re.split(r"(?<=[.!?\n])\s+") over the abbreviation-protected text: a
     whitespace run splits when the character before it ends a sentence *)
  Result := nil;
  LStart := 1;
  I := 2;
  while I <= Length(AText) do
  begin
    if (AText[I] in WS) and (AText[I - 1] in ['.', '!', '?', #10]) and
       not ((AText[I - 1] = '.') and Protected(I - 1)) then
    begin
      J := I;
      while (J <= Length(AText)) and (AText[J] in WS) do
        Inc(J);
      Emit(I);
      LStart := J;
      I := J + 1;
      Continue;
    end;
    Inc(I);
  end;
  Emit(Length(AText) + 1);
end;

function SubjectQuestions(const ASubject: string): TJSONArray;
var
  S: string;
begin
  S := ASubject;
  Result := TJSONArray.Create(['What is ' + S + '?', 'What''s ' + S + '?', 'Do you remember ' + S + '?',
    'Tell me ' + S + '.', 'Remind me, what is ' + S + '?', 'Can you recall ' + S + '?',
    'I forgot ' + S + '. What is it?', 'Do you know what ' + S + ' is?']);
end;

function CleanPhrase(const V: string): string;
var
  R: TRegExpr;
begin
  Result := StripChars(StripChars(V, WS), [' ', '.', '!', ',', ';', ':', '"', '''']);
  R := Rx('\s+(?:and|but|so|because)\s+(?:i|my|you|we)\b', True);
  try
    if R.Exec(Result) then
      Result := Copy(Result, 1, R.MatchPos[0] - 1);
  finally
    R.Free;
  end;
  Result := Utf8Head(StripChars(Result, [' ', '.', '!', ',', ';', ':', '"', '''']), 80);
end;

function CleanValue(const V: string): string;
var
  W: string;
  I, J: Integer;
  Changed: Boolean;
begin
  Result := CleanPhrase(V);
  { drop leading function words (case-sensitive, as in memlib), but never the last word }
  repeat
    Changed := False;
    for W in LeadWords do
      if (Length(Result) > Length(W)) and (Copy(Result, 1, Length(W)) = W) and (Result[Length(W) + 1] in WS) then
      begin
        I := Length(W) + 1;
        J := I;
        while (J <= Length(Result)) and (Result[J] in WS) do
          Inc(J);
        if J <= Length(Result) then
        begin
          Delete(Result, 1, J - 1);
          Changed := True;
          Break;
        end;
      end;
  until not Changed;
end;

function MakeFact(const AText, ASubject, AValue: string; AQuestions: TJSONArray; AConfidence: Double;
  const AKind: string = 'user'): TJSONObject;
begin
  Result := TJSONObject.Create(['text', AText, 'subject', ASubject, 'value', AValue]);
  Result.Add('questions', AQuestions);
  Result.Add('confidence', AConfidence);
  Result.Add('kind', AKind);
end;

function Q(const AItems: array of string): TJSONArray;
var
  S: string;
begin
  Result := TJSONArray.Create;
  for S in AItems do
    Result.Add(S);
end;

function ExtractFactsRules(const AText: string): TJSONArray;
var
  RRemember, RName, RMy, RLive, RWork, RThe: TRegExpr;
  LPiece, LSent, LWhat, LVerb, LPhrase, V: string;
  LBoost: Double;
  LQs: TJSONArray;
begin
  Result := TJSONArray.Create;
  RRemember := Rx('^(?:please\s+)?remember(?:\s+that)?[,:]?\s+(.*)$', True);
  RName := Rx('\bmy name is ([A-Z][\w''-]*(?:\s+[A-Z][\w''-]*)?)', False);
  RMy := Rx('\bmy ([a-z][\w'' -]{1,40}?) (is|are|was) (.+)$', True);
  RLive := Rx('\bi live in (.+)$', True);
  RWork := Rx('\bi work (?:at|for) (.+)$', True);
  RThe := Rx('^(?:the|our) ([a-z][\w'' -]{1,40}?) is (.+)$', True);
  try
    for LPiece in Sentences(AText) do
    begin
      LSent := Trim(LPiece);
      if (LSent = '') or (LSent[Length(LSent)] = '?') then
        Continue;
      LBoost := 0.0;
      if RRemember.Exec(LSent) then
      begin
        LSent := RRemember.Match[1];
        LBoost := 0.1;
      end;

      if RName.Exec(LSent) then
      begin
        V := CleanValue(RName.Match[1]);
        LQs := SubjectQuestions('my name');
        LQs.Add('Who am I?');
        Result.Add(MakeFact('Your name is ' + V + '.', 'my name', V, LQs, 0.9));
        Continue;
      end;
      if RMy.Exec(LSent) then
      begin
        LWhat := Trim(RMy.Match[1]);
        LVerb := LowerCase(RMy.Match[2]);
        LPhrase := CleanPhrase(RMy.Match[3]);
        V := CleanValue(RMy.Match[3]);
        if (V <> '') and not InList(LowerCase(V), StopValues) then
          Result.Add(MakeFact('Your ' + LWhat + ' ' + LVerb + ' ' + LPhrase + '.', 'my ' + LWhat, V,
            SubjectQuestions('my ' + LWhat), Min(1.0, 0.8 + LBoost)));
        Continue;
      end;
      if RLive.Exec(LSent) then
      begin
        V := CleanValue(RLive.Match[1]);
        Result.Add(MakeFact('You live in ' + V + '.', 'where I live', V,
          Q(['Where do I live?', 'Where is my home?', 'Do you remember where I live?',
             'Which city do I live in?', 'Remind me where I live.', 'Do you know where I live?']),
          Min(1.0, 0.8 + LBoost)));
        Continue;
      end;
      if RWork.Exec(LSent) then
      begin
        V := CleanValue(RWork.Match[1]);
        Result.Add(MakeFact('You work at ' + V + '.', 'where I work', V,
          Q(['Where do I work?', 'Who is my employer?', 'Do you remember where I work?',
             'Which company do I work for?', 'Remind me where I work.', 'Do you know where I work?']),
          Min(1.0, 0.8 + LBoost)));
        Continue;
      end;
      if RThe.Exec(LSent) then
      begin
        LWhat := Trim(RThe.Match[1]);
        V := CleanValue(RThe.Match[2]);
        if (V <> '') and not InList(LowerCase(V), StopValues) then
          Result.Add(MakeFact('The ' + LWhat + ' is ' + V + '.', 'the ' + LWhat, V,
            SubjectQuestions('the ' + LWhat), Min(1.0, 0.6 + LBoost), 'world'));
      end;
    end;
  finally
    RRemember.Free;
    RName.Free;
    RMy.Free;
    RLive.Free;
    RWork.Free;
    RThe.Free;
  end;
end;

function CountWords(const S: string): Integer;
var
  I: Integer;
  LIn: Boolean;
begin
  Result := 0;
  LIn := False;
  for I := 1 to Length(S) do
    if IsWordChar(S[I]) then
    begin
      if not LIn then
        Inc(Result);
      LIn := True;
    end
    else
      LIn := False;
end;

function HasStatement(const AText: string): Boolean;
var
  RGreet: TRegExpr;
  LPiece, LSent: string;
  N: Integer;
begin
  Result := False;
  RGreet := Rx('^(hi|hello|hey|thanks|thank you|ok|okay|good (morning|evening|night)|bye)\b', True);
  try
    for LPiece in Sentences(AText) do
    begin
      LSent := Trim(LPiece);
      N := CountWords(LSent);
      if (LSent <> '') and (LSent[Length(LSent)] <> '?') and (N >= 3) and
         not (RGreet.Exec(LSent) and (N < 6)) then
        Exit(True);
    end;
  finally
    RGreet.Free;
  end;
end;

{ [a-z0-9]+ runs of the lowercased text }
function AlnumWords(const S: string): TStringList;
var
  T, W: string;
  C: Char;
begin
  Result := TStringList.Create;
  T := LowerCase(S) + ' ';
  W := '';
  for C in T do
    if C in ['a'..'z', '0'..'9'] then
      W := W + C
    else if W <> '' then
    begin
      Result.Add(W);
      W := '';
    end;
end;

function Grounded(const AValue, ASource: string): Boolean;
var
  VW, SW: TStringList;
  I: Integer;
begin
  { every word of the value must occur in what the user wrote: the guard
    against the model inventing details }
  VW := AlnumWords(AValue);
  SW := AlnumWords(ASource);
  try
    SW.Sorted := True;
    Result := VW.Count > 0;
    for I := 0 to VW.Count - 1 do
      if SW.IndexOf(VW[I]) < 0 then
        Exit(False);
  finally
    VW.Free;
    SW.Free;
  end;
end;

(* the end of the balanced {...} starting at AFrom, string-aware; 0 if unbalanced *)
function ObjectEnd(const S: string; AFrom: Integer): Integer;
var
  I, LDepth: Integer;
  LInStr: Boolean;
begin
  LDepth := 0;
  LInStr := False;
  I := AFrom;
  while I <= Length(S) do
  begin
    if LInStr then
    begin
      if S[I] = '\' then
        Inc(I)
      else if S[I] = '"' then
        LInStr := False;
    end
    else
      case S[I] of
        '"': LInStr := True;
        '{': Inc(LDepth);
        '}':
          begin
            Dec(LDepth);
            if LDepth = 0 then
              Exit(I);
          end;
      end;
    Inc(I);
  end;
  Result := 0;
end;

function ParseJsonObjects(const AOut: string): TJSONArray;
var
  I, E: Integer;
  D: TJSONData;
begin
  { every JSON object in the model output, one per line or pretty-printed }
  Result := TJSONArray.Create;
  I := 1;
  while I <= Length(AOut) do
  begin
    I := PosEx('{', AOut, I);
    if I = 0 then
      Break;
    E := ObjectEnd(AOut, I);
    D := nil;
    if E > 0 then
      try
        D := GetJSON(Copy(AOut, I, E - I + 1));
      except
        on Exception do
          D := nil;
      end;
    if D is TJSONObject then
    begin
      Result.Add(D);
      I := E + 1;
    end
    else
    begin
      D.Free;
      Inc(I);
    end;
  end;
end;

function Field(AObj: TJSONObject; const AName: string): string;
var
  D: TJSONData;
begin
  D := AObj.Find(AName);
  if (D = nil) or (D.JSONType in [jtNull, jtArray, jtObject]) then
    Result := ''
  else
    Result := Trim(D.AsString);
end;

function FactsFromLlmOutput(const AOut, ASource: string): TJSONArray;
var
  LObjs: TJSONArray;
  I: Integer;
  T, S, V: string;
begin
  Result := TJSONArray.Create;
  LObjs := ParseJsonObjects(AOut);
  try
    for I := 0 to LObjs.Count - 1 do
    begin
      T := Field(LObjs.Objects[I], 'text');
      S := Field(LObjs.Objects[I], 'subject');
      V := Field(LObjs.Objects[I], 'value');
      if (T <> '') and (S <> '') and (V <> '') and Grounded(V, ASource) then
        Result.Add(MakeFact(T, S, V, SubjectQuestions(S), 0.7));
    end;
  finally
    LObjs.Free;
  end;
end;

end.
