program memagent;

(* memagent: a Free Pascal agent with two-tier memory.
   Short-term: the chat context plus FTS5 RAG over messages and facts.
   Long-term:  facts baked into a LoRA by a background trainer on a second
               GPU and hot-swapped into the running model. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cwstring,{$ENDIF}
  Mem.Utf8,
  SysUtils, Classes, StrUtils, {$IFDEF UNIX}termio,{$ENDIF}
  Mem.Config, Mem.Db, Mem.Store, Mem.Engine, Mem.Consolidate, Mem.Retrieval, Mem.Agent;

var
  Cfg: TMemConfig;
  Store: TMemStore;
  Engine: TEngine;
  Cons: TConsolidator;
  Ret, Guard: TRetrieval;
  Agent: TAgent;
  Line: string;
  Interactive: Boolean;

begin
  {$IFDEF UNIX}
  Interactive := IsATTY(0) = 1;
  {$ELSE}
  Interactive := True;
  {$ENDIF}
  Cfg := TMemConfig.Create;
  try
    try
      Cfg.Load;
    except
      on E: Exception do
      begin
        WriteLn(StdErr, 'memagent: configuration error: ', E.Message);
        ExitCode := 2;
        Exit;
      end;
    end;
    if Cfg.Provision then
    begin
      (* Local retrieval models: ONNX Runtime + embedder + reranker *)
      if ProvisionRetrieval(Cfg.CacheDir, Cfg.EmbedModel,
        IfThen(Cfg.Reranker = 'off', '', Cfg.RerankModel)) and
        ((Cfg.Guard = 'off') or ProvisionRetrieval(Cfg.CacheDir, Cfg.EmbedModel, Cfg.GuardModel)) then
        WriteLn('retrieval models ready in ', Cfg.CacheDir)
      else
        ExitCode := 1;
      Exit;
    end;
    WriteLn('memagent - profile ', Cfg.ProfileName, ' (', Cfg.BaseModel, ')');
    WriteLn('loading model on ', DeviceName(Cfg.InferGpu), ', training on ', DeviceName(Cfg.TrainGpu), '...');
    Ret := TRetrieval.Create(Cfg.CacheDir, Cfg.Embedder, Cfg.EmbedModel, Cfg.Reranker, Cfg.RerankModel);
    WriteLn('recall: ', Ret.Describe);
    Guard := nil;
    if Cfg.Guard <> 'off' then
    begin
      Guard := TRetrieval.Create(Cfg.CacheDir, 'off', '', 'auto', Cfg.GuardModel);
      if not Guard.HasReranker then
      begin
        if Cfg.Guard = 'on' then
          WriteLn(StdErr, 'guard: ', Cfg.GuardModel, ' is not provisioned (memagent --provision); guard off');
        FreeAndNil(Guard);
      end;
    end;
    WriteLn('guard: ', IfThen(Guard = nil, 'off', 'on (' + Cfg.GuardModel + ')'));
    Store := TMemStore.Create(Cfg.DbPath);
    Engine := TPyWorkerEngine.Create(Cfg);
    Cons := TConsolidator.Create(Cfg);
    Agent := TAgent.Create(Cfg, Store, Engine, Cons, Ret, Guard);
    try
      WriteLn('ready: ', Engine.Describe, '. /help for commands.');
      Agent.Startup;
      while True do
      begin
        if Interactive then
          Write('you> ');
        Flush(Output);
        if EOF(Input) then
          Break;
        ReadLn(Line);
        Line := Trim(Line);
        if Line = '' then
        begin
          Agent.PollConsolidation;
          Continue;
        end;
        if not Interactive then
          WriteLn('you> ', Line);
        try
          if Line[1] = '/' then
          begin
            if not Agent.Command(Line) then
              Break;
          end
          else
            Agent.Turn(Line);
        except
          on E: Exception do
            WriteLn('[error] ', E.ClassName, ': ', E.Message);
        end;
      end;
      if Cons.Running then
      begin
        WriteLn('[sleep] waiting for the running consolidation to finish...');
        Cons.WaitFor;
        Agent.PollConsolidation;
      end;
    finally
      Agent.Free;
      Cons.Free;
      Engine.Free;
      Store.Free;
      Ret.Free;
      Guard.Free;
    end;
  finally
    Cfg.Free;
  end;
end.
