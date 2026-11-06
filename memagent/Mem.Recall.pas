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
  SysUtils, DateUtils, Mem.Config, Mem.Store, Mem.Retrieval, Mem.Guard;

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
  AMsgs := AStore.HybridMessages(AQuery, LVec, ARet.EmbedderId, ABeforeId, ACfg.RetrievalPool,
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

end.
