unit Mem.Recall;

(* The RAG retrieval pipeline, shared by the agent and the benchmarks:

     stage 1  FTS5 + vector search, fused by reciprocal rank (Mem.Store)
     stage 2  the cross-encoder re-scores each (query, candidate) pair and
              drops the ones below rerank_min (Mem.Retrieval)

   Facts about someone the question does not name ("my manager's employer"
   for "Who is my employer?") rank after all others, in both stages. Measured
   (bench/README.md, memory loop 6): with 800 such look-alikes stored, the
   right fact otherwise left the candidate pool for 7 of 60 questions, and a
   bigger pool did not bring it back.

   Facts are matched as "subject: text" (FactDoc). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, StrUtils, Math, DateUtils, Mem.Config, Mem.Store, Mem.Retrieval, Mem.Guard;

type
  TRecallTiming = record
    EmbedMs, SearchMs, RerankMs: Double;
  end;

function FactDoc(const ASubject, AText: string): string;
procedure SortFacts(var A: TFactArray);
procedure SortMsgs(var A: TMsgArray);
function MinCosFor(ACfg: TMemConfig; ARet: TRetrieval): Single;
procedure RecallMemory(AStore: TMemStore; ARet: TRetrieval; ACfg: TMemConfig; const AQuery: string;
  ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray; out ATiming: TRecallTiming);
procedure RecallHops(AStore: TMemStore; ARet: TRetrieval; ACfg: TMemConfig; const AQuery: string;
  ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray; out ATiming: TRecallTiming);
function HopSeed(const AFact: TFactRec): string;
function ContextFacts(ACfg: TMemConfig): Integer;

implementation

function FactDoc(const ASubject, AText: string): string;
begin
  (* What retrieval sees for a fact. The subject carries the relation from the
     user's side ("my sister"), which an extracted sentence like "Ana moved to
     Lisbon" can lose; without it "where does my sister live?" misses. *)
  Result := ASubject + ': ' + AText;
end;

procedure SortFacts(var A: TFactArray);
var
  I, J: Integer;
  T: TFactRec;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J].Score < T.Score) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
end;

procedure SortMsgs(var A: TMsgArray);
var
  I, J: Integer;
  T: TMsgRec;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J].Score < T.Score) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
end;

function MinCosFor(ACfg: TMemConfig; ARet: TRetrieval): Single;
begin
  if ARet.Tier = 'onnx' then
    Result := ACfg.MinSimilarity
  else
    Result := ACfg.StaticMinSimilarity;
end;

{ stable: facts about someone the question does not name go last }
procedure OwnersLast(const AQuery: string; var A: TFactArray);
var
  LMine, LOther: TFactArray;
  I, N, M: Integer;
begin
  SetLength(LMine, Length(A));
  SetLength(LOther, Length(A));
  N := 0;
  M := 0;
  for I := 0 to High(A) do
    if ForeignOwner(AQuery, A[I].Subject) then
    begin
      LOther[M] := A[I];
      Inc(M);
    end
    else
    begin
      LMine[N] := A[I];
      Inc(N);
    end;
  for I := 0 to N - 1 do
    A[I] := LMine[I];
  for I := 0 to M - 1 do
    A[N + I] := LOther[I];
end;

function MsBetween(AFrom: TDateTime): Double;
begin
  Result := MilliSecondsBetween(Now, AFrom);
end;

procedure RecallMemory(AStore: TMemStore; ARet: TRetrieval; ACfg: TMemConfig; const AQuery: string;
  ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray; out ATiming: TRecallTiming);
var
  LVec: TVector;
  LDocs: array of string;
  LScores: TScoreArray;
  I, N: Integer;
  T: TDateTime;
begin
  ATiming := Default(TRecallTiming);
  T := Now;
  LVec := ARet.Embed(AQuery);
  ATiming.EmbedMs := MsBetween(T);
  T := Now;
  AFacts := AStore.HybridFacts(AQuery, LVec, ARet.EmbedderId, ACfg.RetrievalPool, MinCosFor(ACfg, ARet),
    ACfg.RagIncludeBaked, @ForeignOwner);
  AMsgs := AStore.HybridMessages(AQuery, LVec, ARet.EmbedderId, ABeforeId, ACfg.MsgPool,
    MinCosFor(ACfg, ARet));
  ATiming.SearchMs := MsBetween(T);
  if not ARet.HasReranker then
    Exit;
  T := Now;
  SetLength(LDocs, Length(AFacts));
  for I := 0 to High(AFacts) do
    LDocs[I] := FactDoc(AFacts[I].Subject, AFacts[I].Text);
  LScores := ARet.RerankScores(AQuery, LDocs);
  N := 0;
  for I := 0 to High(AFacts) do
    if LScores[I] >= ACfg.RerankMin then
    begin
      AFacts[N] := AFacts[I];
      AFacts[N].Score := LScores[I];
      Inc(N);
    end;
  SetLength(AFacts, N);
  SortFacts(AFacts);
  OwnersLast(AQuery, AFacts);
  SetLength(LDocs, Length(AMsgs));
  for I := 0 to High(AMsgs) do
    LDocs[I] := AMsgs[I].Content;
  LScores := ARet.RerankScores(AQuery, LDocs);
  N := 0;
  for I := 0 to High(AMsgs) do
    if LScores[I] >= ACfg.RerankMin then
    begin
      AMsgs[N] := AMsgs[I];
      AMsgs[N].Score := LScores[I];
      Inc(N);
    end;
  SetLength(AMsgs, N);
  SortMsgs(AMsgs);
  ATiming.RerankMs := MsBetween(T);
end;

(* The name a fact links to, which a second retrieval can follow:
     "my sister's name" = Ana   -> Ana   (the user's person: the value is the name)
     "Ana's city"       = Lisbon -> Ana  (a named person: the owner is the name)
   '' when there is none (a value that is not a name, a subject with no owner). *)
function HopSeed(const AFact: TFactRec): string;
var
  S, V: string;
  P: Integer;
begin
  Result := '';
  S := Trim(AFact.Subject);
  if SameText(Copy(S, 1, 3), 'my ') then
  begin
    V := Trim(AFact.Value);
    if (V <> '') and (V[1] in ['A'..'Z']) and (WordCount(V, [' ']) <= 3) then
      Result := V;
  end
  else
  begin
    P := Pos('''s ', S);
    if (P > 1) and (S[1] in ['A'..'Z']) then
      Result := Copy(S, 1, P - 1);
  end;
end;

{ AWord as a whole word in AText, case-sensitive: names are capitalised }
function MentionsWord(const AText, AWord: string): Boolean;
var
  P, E: Integer;
begin
  Result := False;
  P := PosEx(AWord, AText, 1);
  while P > 0 do
  begin
    E := P + Length(AWord);
    if ((P = 1) or not (AText[P - 1] in ['A'..'Z', 'a'..'z', '0'..'9'])) and
      ((E > Length(AText)) or not (AText[E] in ['A'..'Z', 'a'..'z', '0'..'9'])) then
      Exit(True);
    P := PosEx(AWord, AText, P + 1);
  end;
end;

function HasFact(const A: TFactArray; N: Integer; AId: Int64): Boolean;
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    if A[I].Id = AId then
      Exit(True);
  Result := False;
end;

function ContextFacts(ACfg: TMemConfig): Integer;
begin
  Result := ACfg.RagK;
  if ACfg.Hops >= 2 then
    Inc(Result, ACfg.HopK);
end;

procedure RecallHops(AStore: TMemStore; ARet: TRetrieval; ACfg: TMemConfig; const AQuery: string;
  ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray; out ATiming: TRecallTiming);
var
  LFirst, LSecond, LOut: TFactArray;
  LMsgs2: TMsgArray;
  LT2: TRecallTiming;
  LSeeds: string;
  LSeed: string;
  I, J, N, LAdded: Integer;
begin
  (* Retrieval finds facts that look like the question. A question that needs two
     linked facts ("Which city does my sister live in?" needs "Your sister's name
     is Ana" and "Ana's city is Lisbon") shares words with only one of them, so
     the other is missing from the prompt: one hop found both for 3 of 24 such
     questions in the integration benchmark (bench/README.md). The second hop
     follows each name in the first hits (HopSeed) with "<name> <question>", and
     puts the best new fact that mentions the name right after the fact that
     named it, at most HopK of them. *)
  RecallMemory(AStore, ARet, ACfg, AQuery, ABeforeId, LFirst, AMsgs, ATiming);
  AFacts := LFirst;
  if ACfg.Hops < 2 then
    Exit;
  SetLength(LOut, Length(LFirst) + ACfg.RagK);
  N := 0;
  LAdded := 0;
  LSeeds := '|';
  for I := 0 to Min(ACfg.RagK, Length(LFirst)) - 1 do
  begin
    if not HasFact(LOut, N, LFirst[I].Id) then
    begin
      LOut[N] := LFirst[I];
      Inc(N);
    end;
    LSeed := HopSeed(LFirst[I]);
    if (LSeed = '') or (LAdded >= ACfg.HopK) or (Pos('|' + LSeed + '|', LSeeds) > 0) then
      Continue;
    LSeeds := LSeeds + LSeed + '|';
    RecallMemory(AStore, ARet, ACfg, LSeed + ' ' + AQuery, ABeforeId, LSecond, LMsgs2, LT2);
    ATiming.EmbedMs := ATiming.EmbedMs + LT2.EmbedMs;
    ATiming.SearchMs := ATiming.SearchMs + LT2.SearchMs;
    ATiming.RerankMs := ATiming.RerankMs + LT2.RerankMs;
    for J := 0 to High(LSecond) do
      if not HasFact(LOut, N, LSecond[J].Id) and not HasFact(LFirst, Min(ACfg.RagK, Length(LFirst)), LSecond[J].Id) and
        MentionsWord(LSecond[J].Subject + ' ' + LSecond[J].Text, LSeed) then
      begin
        LOut[N] := LSecond[J];
        Inc(N);
        Inc(LAdded);
        Break;
      end;
  end;
  { the first hop's remaining facts keep their order after the linked ones }
  for I := Min(ACfg.RagK, Length(LFirst)) to High(LFirst) do
    if not HasFact(LOut, N, LFirst[I].Id) then
    begin
      LOut[N] := LFirst[I];
      Inc(N);
    end;
  SetLength(LOut, N);
  AFacts := LOut;
end;

end.
