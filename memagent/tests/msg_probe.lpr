program msg_probe;

(* Which stored messages retrieval returns for each question, without the model.

   msg_probe <cache_dir> <memory.db> <before_id> <pool> < questions.txt > hits.jsonl

   One question per line in; one JSON line out per question, with every message the
   hybrid stage and the reranker keep (up to pool), best first. before_id limits the
   search to messages stored before it, as a turn would. Used to measure how much of
   a LoCoMo answer's evidence reaches the prompt (bench/locomo_recall.py). *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, fpjson, Mem.Config, Mem.Store, Mem.Retrieval, Mem.Recall;

var
  Cfg: TMemConfig;
  Ret: TRetrieval;
  Store: TMemStore;
  Line: string;
  Facts: TFactArray;
  Msgs: TMsgArray;
  Timing: TRecallTiming;
  Out_, Hit: TJSONObject;
  Hits: TJSONArray;
  I: Integer;
begin
  if ParamCount < 4 then
  begin
    WriteLn(StdErr, 'usage: msg_probe <cache_dir> <memory.db> <before_id> <pool> < questions > hits.jsonl');
    Halt(2);
  end;
  Cfg := TMemConfig.Create;
  Cfg.RetrievalPool := 8;
  Cfg.MsgPool := StrToInt(ParamStr(4));
  Cfg.RagK := 3;
  Cfg.RagIncludeBaked := True;
  Cfg.MinSimilarity := 0.25;
  Cfg.StaticMinSimilarity := 0.3;
  Cfg.RerankMin := StrToFloatDef(GetEnvironmentVariable('PROBE_RERANK_MIN'), 0.0001);
  Ret := TRetrieval.Create(ParamStr(1), 'auto', 'minilm', 'auto', 'ms-marco-minilm');
  Store := TMemStore.Create(ParamStr(2));
  try
    while not EOF(Input) do
    begin
      ReadLn(Line);
      if Trim(Line) = '' then
        Continue;
      RecallMemory(Store, Ret, Cfg, Line, StrToInt64(ParamStr(3)), Facts, Msgs, Timing);
      Out_ := TJSONObject.Create;
      Hits := TJSONArray.Create;
      for I := 0 to High(Msgs) do
      begin
        Hit := TJSONObject.Create;
        Hit.Add('id', Msgs[I].Id);
        Hit.Add('score', Msgs[I].Score);
        Hit.Add('cos', Msgs[I].Cos);
        Hit.Add('content', Msgs[I].Content);
        Hits.Add(Hit);
      end;
      Out_.Add('q', Line);
      Out_.Add('hits', Hits);
      WriteLn(Out_.AsJSON);
      Out_.Free;
    end;
  finally
    Store.Free;
    Ret.Free;
    Cfg.Free;
  end;
end.
