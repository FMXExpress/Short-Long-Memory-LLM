unit Mem.Engine;

(* The model, as seen by the agent. TEngine is the seam: today the only
   implementation drives the Python worker (trainer/worker.py) over JSON
   lines, and an in-process llama.cpp engine can replace it later without
   the agent noticing. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Process, fpjson, jsonparser, Mem.Config;

type
  EEngine = class(Exception);
  TTokenEvent = procedure(const AText: string) of object;

  TEngine = class
  public
    // AMessages is [{"role": .., "content": ..}, ...]; ownership stays with the caller
    function Generate(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
      AMaxNewTokens: Integer = 0): string; virtual; abstract;
    { caller frees the result }
    { AMode: 'rules' (instant), 'llm' (model pass only), 'auto' (rules, else model) }
    function ExtractFacts(const AText, AMode: string): TJSONArray; virtual; abstract;
    procedure LoadAdapter(const APath, AName: string); virtual; abstract;
    procedure ClearAdapter; virtual; abstract;
    function ActiveAdapter: string; virtual; abstract;
    function Describe: string; virtual; abstract;
  end;

  TPyWorkerEngine = class(TEngine)
  private
    FCfg: TMemConfig;
    FProc: TProcess;
    FBuf: string;
    FNextId: Integer;
    FActive, FActivePath: string;
    FDevice: string;
    FRestarts: Integer;
    function ReadLine: string;
    procedure SendLine(const S: string);
    procedure Start;
    procedure Stop;
    procedure EnsureRunning;
    function Call(AReq: TJSONObject; AOnToken: TTokenEvent): TJSONObject;
  public
    constructor Create(ACfg: TMemConfig);
    destructor Destroy; override;
    function Generate(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
      AMaxNewTokens: Integer = 0): string; override;
    function ExtractFacts(const AText, AMode: string): TJSONArray; override;
    procedure LoadAdapter(const APath, AName: string); override;
    procedure ClearAdapter; override;
    function ActiveAdapter: string; override;
    function Describe: string; override;
  end;

procedure InheritEnvironment(AProc: TProcess);

implementation

procedure InheritEnvironment(AProc: TProcess);
var
  I: Integer;
begin
  { Setting any variable on TProcess replaces the whole environment, so copy it first }
  for I := 1 to GetEnvironmentVariableCount do
    AProc.Environment.Add(GetEnvironmentString(I));
end;

constructor TPyWorkerEngine.Create(ACfg: TMemConfig);
begin
  inherited Create;
  FCfg := ACfg;
  Start;
end;

destructor TPyWorkerEngine.Destroy;
begin
  Stop;
  inherited Destroy;
end;

procedure TPyWorkerEngine.Start;
var
  LReady: TJSONData;
  LLine: string;
begin
  FBuf := '';
  FProc := TProcess.Create(nil);
  FProc.Executable := FCfg.PythonExe;
  FProc.Parameters.Add('-u');
  FProc.Parameters.Add(FCfg.TrainerScript('worker.py'));
  FProc.Parameters.Add('--profile');
  FProc.Parameters.Add(FCfg.ProfileName);
  FProc.Parameters.Add('--models');
  FProc.Parameters.Add(FCfg.ModelsFile);
  FProc.Parameters.Add('--log');
  FProc.Parameters.Add(IncludeTrailingPathDelimiter(FCfg.LogDir) + 'worker.log');
  InheritEnvironment(FProc);
  FProc.Environment.Add(GpuEnv(FCfg.InferGpu));
  FProc.Options := [poUsePipes];
  FProc.Execute;

  (* The worker prints {"id":0,"type":"ready",...} once the model is loaded.
     Loading can take minutes for a big model, so there is no timeout here:
     a dead process ends the wait through EOF instead. *)
  LLine := ReadLine;
  LReady := GetJSON(LLine);
  try
    if TJSONObject(LReady).Get('type', '') <> 'ready' then
      raise EEngine.Create('worker failed to start: ' + TJSONObject(LReady).Get('message', LLine));
    FDevice := TJSONObject(LReady).Get('device', '?');
  finally
    LReady.Free;
  end;
end;

procedure TPyWorkerEngine.Stop;
begin
  if FProc = nil then
    Exit;
  if FProc.Running then
  begin
    try
      SendLine('{"id":-1,"op":"shutdown"}');
      FProc.CloseInput;
    except
      on Exception do ;
    end;
    FProc.WaitOnExit(10000);
    if FProc.Running then
      FProc.Terminate(1);
  end;
  FreeAndNil(FProc);
end;

procedure TPyWorkerEngine.EnsureRunning;
begin
  if (FProc <> nil) and FProc.Running then
    Exit;
  Inc(FRestarts);
  WriteLn(StdErr, '[engine] worker is not running, restarting (', FRestarts, ')');
  Flush(StdErr);   { FPC buffers StdErr; without this the notice shows up at exit }
  FreeAndNil(FProc);
  Start;
  (* A fresh worker comes up on the bare base model; put the memory back. *)
  if FActivePath <> '' then
  begin
    FActive := '';
    LoadAdapter(FActivePath, ExtractFileName(ExcludeTrailingPathDelimiter(FActivePath)));
  end;
end;

procedure TPyWorkerEngine.SendLine(const S: string);
var
  L: string;
begin
  L := S + #10;
  FProc.Input.WriteBuffer(L[1], Length(L));
end;

function TPyWorkerEngine.ReadLine: string;
var
  P, N: Integer;
  LChunk: array[0..4095] of Char;
begin
  repeat
    P := Pos(#10, FBuf);
    if P > 0 then
    begin
      Result := Copy(FBuf, 1, P - 1);
      Delete(FBuf, 1, P);
      Exit;
    end;
    N := FProc.Output.Read(LChunk, SizeOf(LChunk));
    if N <= 0 then
      raise EEngine.CreateFmt('worker exited (see %s)',
        [IncludeTrailingPathDelimiter(FCfg.LogDir) + 'worker.log']);
    SetLength(Result, N);
    Move(LChunk[0], Result[1], N);
    FBuf := FBuf + Result;
  until False;
end;

function TPyWorkerEngine.Call(AReq: TJSONObject; AOnToken: TTokenEvent): TJSONObject;
var
  LId: Integer;
  LMsg: TJSONData;
  LObj: TJSONObject;
  LType: string;
begin
  try
    EnsureRunning;
    Inc(FNextId);
    LId := FNextId;
    AReq.Add('id', LId);
    SendLine(AReq.AsJSON);
  finally
    AReq.Free;
  end;
  repeat
    LMsg := GetJSON(ReadLine);
    LObj := LMsg as TJSONObject;
    if LObj.Get('id', -1) <> LId then
    begin
      LMsg.Free;
      Continue;
    end;
    LType := LObj.Get('type', '');
    if LType = 'token' then
    begin
      if Assigned(AOnToken) then
        AOnToken(LObj.Get('text', ''));
      LMsg.Free;
    end
    else if LType = 'error' then
    begin
      try
        raise EEngine.Create('worker: ' + LObj.Get('message', '?'));
      finally
        LMsg.Free;
      end;
    end
    else
      Exit(LObj);
  until False;
end;

function TPyWorkerEngine.Generate(AMessages: TJSONArray; AUseAdapter: Boolean;
  AOnToken: TTokenEvent; AMaxNewTokens: Integer): string;
var
  LReq, LRes: TJSONObject;
begin
  LReq := TJSONObject.Create(['op', 'generate', 'adapter', AUseAdapter,
    'stream', Assigned(AOnToken)]);
  LReq.Add('messages', AMessages.Clone);
  if AMaxNewTokens > 0 then
    LReq.Add('max_new_tokens', AMaxNewTokens);
  LRes := Call(LReq, AOnToken);
  try
    Result := LRes.Get('text', '');
  finally
    LRes.Free;
  end;
end;

function TPyWorkerEngine.ExtractFacts(const AText, AMode: string): TJSONArray;
var
  LRes: TJSONObject;
begin
  LRes := Call(TJSONObject.Create(['op', 'extract_facts', 'text', AText, 'mode', AMode]), nil);
  try
    Result := LRes.Arrays['facts'].Clone as TJSONArray;
  finally
    LRes.Free;
  end;
end;

procedure TPyWorkerEngine.LoadAdapter(const APath, AName: string);
begin
  Call(TJSONObject.Create(['op', 'load_adapter', 'path', APath, 'name', AName]), nil).Free;
  FActive := AName;
  FActivePath := APath;
end;

procedure TPyWorkerEngine.ClearAdapter;
begin
  Call(TJSONObject.Create(['op', 'clear_adapter']), nil).Free;
  FActive := '';
  FActivePath := '';
end;

function TPyWorkerEngine.ActiveAdapter: string;
begin
  Result := FActive;
end;

function TPyWorkerEngine.Describe: string;
begin
  Result := Format('%s (%s) on %s via python worker', [FCfg.ProfileName, FCfg.BaseModel, FDevice]);
end;

end.
