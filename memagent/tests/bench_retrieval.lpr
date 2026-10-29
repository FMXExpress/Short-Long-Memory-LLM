program bench_retrieval;

(* Retrieval benchmark over bench/facts.json, through the production pipeline
   (Mem.Store + Mem.Retrieval + Mem.Recall), in several configurations.

     bench_retrieval <cache_dir> <facts.json> [filler_facts] [filler_messages]

   Per configuration: hit@1 / hit@3 (the right fact among the first 1 / rag_k=3
   injected), MRR, wrong@1 (another fact ranked first: interference), noise
   (unknown questions that still get a fact injected) and latency per query.
   filler_facts adds synthetic facts to measure how search time scales. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Classes, DateUtils, Math, fpjson, jsonparser,
  Mem.Config, Mem.Store, Mem.Retrieval, Mem.Recall;

type
  TMode = record
    Name, Embed, Rerank, RerankModel: string;
    RerankMin: Double;
  end;

const
  Modes: array[0..4] of TMode = (
    (Name: 'fts'; Embed: 'off'; Rerank: 'off'; RerankModel: ''; RerankMin: 0),
    (Name: 'static+fts'; Embed: 'static'; Rerank: 'off'; RerankModel: ''; RerankMin: 0),
    (Name: 'minilm+fts'; Embed: 'auto'; Rerank: 'off'; RerankModel: ''; RerankMin: 0),
    (Name: 'minilm+fts+msmarco'; Embed: 'auto'; Rerank: 'auto'; RerankModel: 'ms-marco-minilm'; RerankMin: 0.005),
    (Name: 'minilm+fts+bge'; Embed: 'auto'; Rerank: 'auto'; RerankModel: 'bge-reranker-base'; RerankMin: 0.0005));

var
  Data: TJSONObject;
  Facts, Unknown: TJSONArray;
  CacheDir: string;
  Filler, FillerMsgs: Integer;
  Sweep: Boolean;

type
  TCand = record
    Hit: Boolean;      { this candidate is the right fact }
    Cos, Score: Single;
  end;
  TCandList = array of TCand;

var
  QCands: array of TCandList;     { per test question, in pipeline order }
  UCands: array of TCandList;     { per unknown question }

{ the policy keeps a candidate when cos >= ACos and score >= AScore; reports
  hit@3 over the kept list, plus noise over the unknown questions }
procedure Policy(const AName: string; ACos, AScore: Single; ARagK: Integer);
var
  I, K, Kept, NQ, Hit3, Noise, Wrong1: Integer;
  First: Boolean;
begin
  NQ := Length(QCands); Hit3 := 0; Noise := 0; Wrong1 := 0;
  for I := 0 to High(QCands) do
  begin
    Kept := 0;
    First := True;
    for K := 0 to High(QCands[I]) do
      if (QCands[I][K].Cos >= ACos) and (QCands[I][K].Score >= AScore) then
      begin
        if First and not QCands[I][K].Hit then Inc(Wrong1);
        First := False;
        Inc(Kept);
        if QCands[I][K].Hit and (Kept <= ARagK) then Inc(Hit3);
      end;
  end;
  for I := 0 to High(UCands) do
    for K := 0 to High(UCands[I]) do
      if (UCands[I][K].Cos >= ACos) and (UCands[I][K].Score >= AScore) then
      begin
        Inc(Noise);
        Break;
      end;
  WriteLn(Format('    %-28s hit@3 %.3f  wrong@1 %.3f  noise %.3f',
    [AName, Hit3 / NQ, Wrong1 / NQ, Noise / Max(1, Length(UCands))]));
end;

function LoadJson(const APath: string): TJSONObject;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := GetJSON(L.Text) as TJSONObject;
  finally
    L.Free;
  end;
end;

function Percentile(var A: array of Double; P: Double): Double;
var
  I, J: Integer;
  T: Double;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J] > T) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
  Result := A[Min(High(A), Trunc(P * Length(A)))];
end;

procedure RunMode(const AMode: TMode);
var
  Cfg: TMemConfig;
  Ret: TRetrieval;
  Store: TMemStore;
  DbPath: string;
  I, J, K, NQ, Hit1, Hit3, Wrong1, Noise, Rank: Integer;
  Ids: array of Int64;
  F: TJSONObject;
  LId: Int64;
  LFacts: TFactArray;
  LMsgs: TMsgArray;
  LTiming: TRecallTiming;
  Mrr, TSum, ESum, SSum, RSum: Double;
  Lat: array of Double;
  T, TLoad: TDateTime;
  Q: string;
begin
  Cfg := TMemConfig.Create;
  Cfg.RetrievalPool := 8;
  Cfg.RagK := 3;
  Cfg.RagIncludeBaked := True;
  Cfg.MinSimilarity := 0.35;
  Cfg.StaticMinSimilarity := 0.3;
  Cfg.RerankMin := AMode.RerankMin;
  if Sweep then
  begin
    { keep everything the stages produce; the policies filter afterwards }
    Cfg.RerankMin := -1;
    Cfg.MinSimilarity := 0.2;
    Cfg.StaticMinSimilarity := 0.2;
  end;
  Ret := TRetrieval.Create(CacheDir, AMode.Embed, 'minilm', AMode.Rerank, AMode.RerankModel);
  if (AMode.Embed = 'auto') and (Ret.Tier <> 'onnx') or (AMode.Rerank = 'auto') and not Ret.HasReranker then
  begin
    WriteLn(Format('%-20s skipped (%s)', [AMode.Name, Ret.Describe]));
    Ret.Free;
    Cfg.Free;
    Exit;
  end;
  DbPath := GetTempFileName(GetTempDir, 'benchret');
  Store := TMemStore.Create(DbPath);
  try
    TLoad := Now;
    SetLength(Ids, Facts.Count);
    for I := 0 to Facts.Count - 1 do
    begin
      F := TJSONObject(Facts.Objects[I].Clone);
      try
        F.Add('kind', 'user');
        F.Add('confidence', 0.9);
        Store.AddFact(F, 0, 0, LId);
        Ids[I] := LId;
        if Ret.Tier <> 'off' then
          Store.SetFactVectors(LId, Ret.Embed(FactDoc(F.Get('subject', ''), F.Get('text', ''))),
            Ret.Embed(F.Get('subject', '')), Ret.EmbedderId);
      finally
        F.Free;
      end;
    end;
    for I := 1 to Filler do
    begin
      F := TJSONObject.Create(['subject', Format('filler topic %d', [I]), 'value', Format('item%d', [I]),
        'text', Format('Note %d: the shipment of crate %d goes to warehouse %d on day %d.', [I, I * 7, I mod 13, I mod 28]),
        'kind', 'world', 'confidence', 0.9]);
      try
        Store.AddFact(F, 0, 0, LId);
        if Ret.Tier <> 'off' then
          Store.SetFactVectors(LId, Ret.Embed(FactDoc(F.Get('subject', ''), F.Get('text', ''))),
            Ret.Embed(F.Get('subject', '')), Ret.EmbedderId);
      finally
        F.Free;
      end;
    end;
    for I := 1 to FillerMsgs do
    begin
      LId := Store.AddMessage('filler', 'user', Format('Can you check whether order %d shipped to %s on day %d? ' +
        'Also note task %d.', [I, 'warehouse ' + IntToStr(I mod 17), I mod 28, I * 3]));
      if Ret.Tier <> 'off' then
        Store.SetMessageVector(LId, Ret.Embed(Format('order %d warehouse %d day %d task %d',
          [I, I mod 17, I mod 28, I * 3])), Ret.EmbedderId);
    end;
    TLoad := Now - TLoad;

    NQ := 0; Hit1 := 0; Hit3 := 0; Wrong1 := 0; Mrr := 0;
    TSum := 0; ESum := 0; SSum := 0; RSum := 0;
    Lat := nil;
    QCands := nil;
    UCands := nil;
    for I := 0 to Facts.Count - 1 do
      for J := 0 to Facts.Objects[I].Arrays['tests'].Count - 1 do
      begin
        Q := Facts.Objects[I].Arrays['tests'].Strings[J];
        T := Now;
        RecallMemory(Store, Ret, Cfg, Q, High(Int64), LFacts, LMsgs, LTiming);
        SetLength(Lat, Length(Lat) + 1);
        Lat[High(Lat)] := MilliSecondsBetween(Now, T);
        TSum := TSum + Lat[High(Lat)];
        ESum := ESum + LTiming.EmbedMs;
        SSum := SSum + LTiming.SearchMs;
        RSum := RSum + LTiming.RerankMs;
        Inc(NQ);
        Rank := 0;
        for K := 0 to High(LFacts) do
          if LFacts[K].Id = Ids[I] then
          begin
            Rank := K + 1;
            Break;
          end;
        if Rank = 1 then Inc(Hit1);
        if (Rank >= 1) and (Rank <= Cfg.RagK) then Inc(Hit3);
        if Rank > 0 then Mrr := Mrr + 1 / Rank;
        if (Length(LFacts) > 0) and (LFacts[0].Id <> Ids[I]) then Inc(Wrong1);
        SetLength(QCands, Length(QCands) + 1);
        SetLength(QCands[High(QCands)], Length(LFacts));
        for K := 0 to High(LFacts) do
        begin
          QCands[High(QCands)][K].Hit := LFacts[K].Id = Ids[I];
          QCands[High(QCands)][K].Cos := LFacts[K].Cos;
          QCands[High(QCands)][K].Score := LFacts[K].Score;
        end;
      end;
    Noise := 0;
    for I := 0 to Unknown.Count - 1 do
    begin
      RecallMemory(Store, Ret, Cfg, Unknown.Objects[I].Get('q', ''), High(Int64), LFacts, LMsgs, LTiming);
      if Length(LFacts) > 0 then Inc(Noise);
      SetLength(UCands, Length(UCands) + 1);
      SetLength(UCands[High(UCands)], Length(LFacts));
      for K := 0 to High(LFacts) do
      begin
        UCands[High(UCands)][K].Cos := LFacts[K].Cos;
        UCands[High(UCands)][K].Score := LFacts[K].Score;
      end;
    end;
    WriteLn(Format('%-20s hit@1 %.3f  hit@3 %.3f  mrr %.3f  wrong@1 %.3f  noise %.3f  ' +
      'ms/query %.1f (p95 %.0f; embed %.1f search %.1f rerank %.1f)  load %.1fs',
      [AMode.Name, Hit1 / NQ, Hit3 / NQ, Mrr / NQ, Wrong1 / NQ, Noise / Unknown.Count,
       TSum / NQ, Percentile(Lat, 0.95), ESum / NQ, SSum / NQ, RSum / NQ, TLoad * 86400]));
    if Sweep and (Ret.Tier = 'onnx') then
    begin
      Policy('as configured above', -9, -9, Cfg.RagK);
      for K := 0 to 5 do
        Policy(Format('cos>=%.2f', [0.25 + K * 0.05]), 0.25 + K * 0.05, -9, Cfg.RagK);
      if Ret.HasReranker then
        for J := 0 to 6 do
          Policy(Format('rerank>=%g', [Power(10, -5 + J * 0.5)]), -9, Power(10, -5 + J * 0.5), Cfg.RagK);
      if Ret.HasReranker then
        for K := 0 to 3 do
          for J := 0 to 4 do
            Policy(Format('cos>=%.2f & rerank>=%g', [0.25 + K * 0.05, Power(10, -5 + J)]),
              0.25 + K * 0.05, Power(10, -5 + J), Cfg.RagK);
    end;
    Flush(Output);
  finally
    Store.Free;
    Ret.Free;
    Cfg.Free;
    DeleteFile(DbPath);
    DeleteFile(DbPath + '-wal');
    DeleteFile(DbPath + '-shm');
  end;
end;

var
  M: Integer;
begin
  CacheDir := ParamStr(1);
  Data := LoadJson(ParamStr(2));
  Filler := StrToIntDef(ParamStr(3), 0);
  FillerMsgs := StrToIntDef(ParamStr(4), 0);
  Sweep := GetEnvironmentVariable('BENCH_SWEEP') = '1';
  Facts := Data.Arrays['facts'];
  Unknown := Data.Arrays['unknown'];
  WriteLn(Format('%d facts (+%d filler facts, +%d filler messages), %d test questions, %d unknown',
    [Facts.Count, Filler, FillerMsgs, Facts.Count * 2, Unknown.Count]));
  for M := Low(Modes) to High(Modes) do
    if (GetEnvironmentVariable('BENCH_MODE') = '') or (GetEnvironmentVariable('BENCH_MODE') = Modes[M].Name) then
      RunMode(Modes[M]);
  Data.Free;
end.
