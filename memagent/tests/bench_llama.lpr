program bench_llama;

(* The in-process trainer and engine for bench/bench_memory.py --trainer llama.

     bench_llama --config memagent.ini --data DIR [--profile P] train [--set key=value ...]
       consolidates DIR/memory.db with Mem.Train (gate off, CPU) and prints
       the summary JSON line
     bench_llama --config memagent.ini --data DIR [--profile P] ask ADAPTER_DIR|- IN.json OUT.json
       IN.json: [{"system": .., "user": .., "lora": true|false}, ...]
       OUT.json: the answers, greedy, 40 tokens, as bench_memory.ask

   --set overrides the profile's training settings: examples, per_fact,
   epochs, lr, batch, rank, alpha, targets (llama.cpp names, comma-separated),
   anchors. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Classes, fpjson, jsonparser, Mem.Config, Mem.Consolidate, Mem.Train,
  Mem.Engine, Mem.Engine.Llama, Mem.LlamaChat;

var
  Cfg: TMemConfig;
  Cmd: string;
  Args: TStringList;

procedure ApplySet(const AKeyValue: string);
var
  K, V: string;
  LFmt: TFormatSettings;
begin
  LFmt := DefaultFormatSettings;
  LFmt.DecimalSeparator := '.';
  K := Copy(AKeyValue, 1, Pos('=', AKeyValue) - 1);
  V := Copy(AKeyValue, Pos('=', AKeyValue) + 1, MaxInt);
  if K = 'examples' then Cfg.TrainExamples := V
  else if K = 'per_fact' then Cfg.TrainParaphrases := StrToInt(V)
  else if K = 'epochs' then Cfg.TrainEpochs := StrToInt(V)
  else if K = 'lr' then Cfg.TrainLr := StrToFloat(V, LFmt)
  else if K = 'batch' then Cfg.TrainBatch := StrToInt(V)
  else if K = 'rank' then Cfg.LoraRank := StrToInt(V)
  else if K = 'alpha' then Cfg.LoraAlpha := StrToFloat(V, LFmt)
  else if K = 'targets' then Cfg.LoraTargets := V
  else if K = 'anchors' then Cfg.TrainAnchors := StrToInt(V)
  else if K = 'max_windows' then Cfg.TrainMaxWindows := StrToInt(V)
  else
    raise Exception.Create('unknown --set key: ' + K);
end;

procedure Train;
var
  LJob: TTrainJob;
begin
  Cfg.GateMinRecall := 0;
  Cfg.GateMaxPplRatio := 1e9;
  LJob := TLlamaTrainThread.Create(Cfg);
  try
    LJob.WaitFor;
    WriteLn(LJob.Summary);
    ExitCode := LJob.ExitCode;
  finally
    LJob.Free;
  end;
end;

procedure Ask(const AAdapter, AIn, AOut: string);
var
  LEngine: TLlamaEngine;
  LIn: TJSONArray;
  LOut: TJSONArray;
  LItem: TJSONObject;
  LMsgs: TJSONArray;
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  LEngine := TLlamaEngine.Create(Cfg);
  try
    if AAdapter <> '-' then
      LEngine.LoadAdapter(AAdapter, 'bench');
    L.LoadFromFile(AIn);
    LIn := GetJSON(L.Text) as TJSONArray;
    LOut := TJSONArray.Create;
    try
      for I := 0 to LIn.Count - 1 do
      begin
        LItem := LIn.Objects[I];
        LMsgs := ChatMessages(['system', LItem.Get('system', ''), 'user', LItem.Get('user', '')]);
        try
          LOut.Add(LEngine.Generate(LMsgs, LItem.Get('lora', False), nil, 40));
        finally
          LMsgs.Free;
        end;
      end;
      L.Text := LOut.AsJSON;
      L.SaveToFile(AOut);
    finally
      LIn.Free;
      LOut.Free;
    end;
  finally
    LEngine.Free;
    L.Free;
  end;
end;

var
  I: Integer;
begin
  Cfg := TMemConfig.Create;
  Args := TStringList.Create;
  try
    Cfg.Load;
    Cfg.InferGpu := -1;
    Cfg.TrainGpu := -1;
    Cmd := '';
    I := 1;
    while I <= ParamCount do
    begin
      if (ParamStr(I) = '--config') or (ParamStr(I) = '--data') or (ParamStr(I) = '--profile') then
        Inc(I)
      else if ParamStr(I) = '--set' then
      begin
        Inc(I);
        ApplySet(ParamStr(I));
      end
      else if Cmd = '' then
        Cmd := ParamStr(I)
      else
        Args.Add(ParamStr(I));
      Inc(I);
    end;
    if Cmd = 'train' then
      Train
    else if (Cmd = 'ask') and (Args.Count = 3) then
      Ask(Args[0], Args[1], Args[2])
    else
    begin
      WriteLn(StdErr, 'usage: bench_llama --config INI --data DIR [--profile P] train [--set k=v ...]');
      WriteLn(StdErr, '       bench_llama --config INI --data DIR [--profile P] ask ADAPTER_DIR|- IN.json OUT.json');
      ExitCode := 2;
    end;
  finally
    Args.Free;
    Cfg.Free;
  end;
end.
