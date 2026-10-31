unit Mem.Consolidate;

(* Consolidation on a background thread, so the chat keeps going while the
   LoRA trains. Two trainers, chosen by [consolidate] trainer:

   - llama (default): in-process on llama.cpp (Mem.Train), no Python;
   - python: trainer/consolidate.py as a one-shot process on [gpu] train.

   Either ends with an exit code and a JSON summary: 0 promoted, 2 rejected
   by the eval gate, 3 nothing to train, 4 cancelled, anything else is an
   error.

   With memory tiers (llama trainer) a run is a deep sleep (every fact, a
   fresh adapter on the base model) or a fast run (the facts the deep
   adapter does not hold, a fresh adapter trained on top of the frozen deep
   one). A fast run yields to the chat: it waits between training windows
   while a turn is being answered (HoldTraining). *)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Process, Mem.Config;

type
  { a consolidation run: ExitCode and Summary (JSON) are set when it finishes }
  TTrainJob = class(TThread)
  protected
    FCfg: TMemConfig;
    FExitCode: Integer;
    FSummary: string;
    FTier: string;
    FDeepId, FDeepPath: string;
  public
    { ATier: deep, or fast on top of the deep adapter ADeepId at ADeepPath }
    constructor Create(ACfg: TMemConfig; const ATier: string = 'deep'; const ADeepId: string = '';
      const ADeepPath: string = '');
    property ExitCode: Integer read FExitCode;
    property Summary: string read FSummary;
    property Tier: string read FTier;
  end;

  { trainer = python: trainer/consolidate.py }
  TConsolidateThread = class(TTrainJob)
  protected
    procedure Execute; override;
  end;

  TConsolidator = class
  private
    FCfg: TMemConfig;
    FThread: TTrainJob;
    FStartedAt: TDateTime;
    function LockFile: string;
  public
    constructor Create(ACfg: TMemConfig);
    destructor Destroy; override;
    function Start(const ATier: string = 'deep'; const ADeepId: string = '';
      const ADeepPath: string = ''): Boolean;
    function Running: Boolean;
    { the tier of the running (or finished, not yet taken) run; '' when none }
    function Tier: string;
    { asks the running run to stop at its next window; it ends with exit 4 }
    procedure Cancel;
    { True once a run has finished; hands back its result and resets }
    function TakeResult(out AExitCode: Integer; out ASummary: string): Boolean;
    procedure WaitFor;
    property StartedAt: TDateTime read FStartedAt;
  end;

{ True while a chat turn is being answered: a training run waits between windows }
procedure HoldTraining(AHold: Boolean);
function TrainingHeld: Boolean;

implementation

uses
  Mem.Engine, Mem.Train;

var
  GHold: LongInt = 0;

procedure HoldTraining(AHold: Boolean);
begin
  if AHold then
    InterLockedExchange(GHold, 1)
  else
    InterLockedExchange(GHold, 0);
end;

function TrainingHeld: Boolean;
begin
  Result := InterLockedExchangeAdd(GHold, 0) <> 0;
end;

constructor TTrainJob.Create(ACfg: TMemConfig; const ATier, ADeepId, ADeepPath: string);
begin
  FCfg := ACfg;
  FExitCode := -1;
  FTier := ATier;
  FDeepId := ADeepId;
  FDeepPath := ADeepPath;
  inherited Create(False);
end;

procedure TConsolidateThread.Execute;
var
  LProc: TProcess;
  LLog: TFileStream;
  LBuf: array[0..4095] of Byte;
  LTail, LLine: string;
  N, P: Integer;
  LLogName: string;
begin
  LLogName := IncludeTrailingPathDelimiter(FCfg.LogDir) + 'consolidate.log';
  if FileExists(LLogName) then
    LLog := TFileStream.Create(LLogName, fmOpenWrite or fmShareDenyNone)
  else
    LLog := TFileStream.Create(LLogName, fmCreate);
  LProc := TProcess.Create(nil);
  try
    LLog.Seek(0, soEnd);
    LProc.Executable := FCfg.PythonExe;
    LProc.Parameters.Add('-u');
    LProc.Parameters.Add(FCfg.TrainerScript('consolidate.py'));
    LProc.Parameters.Add('--db');
    LProc.Parameters.Add(FCfg.DbPath);
    LProc.Parameters.Add('--out');
    LProc.Parameters.Add(FCfg.AdapterDir);
    LProc.Parameters.Add('--profile');
    LProc.Parameters.Add(FCfg.ProfileName);
    LProc.Parameters.Add('--models');
    LProc.Parameters.Add(FCfg.ModelsFile);
    LProc.Parameters.Add('--priors');
    LProc.Parameters.Add(FCfg.PriorsFile);
    if FCfg.Engine = 'llama' then
    begin
      LProc.Parameters.Add('--gguf');
      LProc.Parameters.Add(FCfg.LlamaSrcDir);
    end;
    InheritEnvironment(LProc);
    LProc.Environment.Add(GpuEnv(FCfg.TrainGpu));
    { one pipe for both streams: the whole run goes to the log, and only
      the last JSON line matters to us }
    LProc.Options := [poUsePipes, poStderrToOutPut];
    try
      LProc.Execute;
    except
      on E: Exception do
      begin
        FSummary := '{"exit":1,"reason":"cannot start ' + StringReplace(FCfg.PythonExe, '\', '/', [rfReplaceAll]) +
          ' (install Python and trainer/requirements.txt)"}';
        FExitCode := 1;
        Exit;
      end;
    end;
    LTail := '';
    repeat
      N := LProc.Output.Read(LBuf, SizeOf(LBuf));
      if N > 0 then
      begin
        LLog.WriteBuffer(LBuf, N);
        SetString(LLine, PChar(@LBuf[0]), N);
        LTail := LTail + LLine;
        if Length(LTail) > 65536 then
          Delete(LTail, 1, Length(LTail) - 65536);
      end;
    until N <= 0;
    LProc.WaitOnExit;
    FExitCode := LProc.ExitStatus;
    { last line that looks like JSON }
    LTail := TrimRight(LTail);
    P := Length(LTail);
    while (P > 0) and (LTail[P] <> #10) do
      Dec(P);
    FSummary := Copy(LTail, P + 1, MaxInt);
    if (FSummary = '') or (FSummary[1] <> '{') then
      FSummary := '{"exit":' + IntToStr(FExitCode) + ',"reason":"no summary line"}';
  finally
    LProc.Free;
    LLog.Free;
  end;
end;

constructor TConsolidator.Create(ACfg: TMemConfig);
begin
  inherited Create;
  FCfg := ACfg;
  { a lock left behind by a crashed run would block consolidation forever }
  if FileExists(LockFile) then
    DeleteFile(LockFile);
end;

destructor TConsolidator.Destroy;
begin
  if FThread <> nil then
  begin
    FThread.WaitFor;
    FThread.Free;
  end;
  if FileExists(LockFile) then
    DeleteFile(LockFile);
  inherited Destroy;
end;

function TConsolidator.LockFile: string;
begin
  Result := IncludeTrailingPathDelimiter(FCfg.DataDir) + 'consolidate.lock';
end;

function TConsolidator.Start(const ATier, ADeepId, ADeepPath: string): Boolean;
var
  F: TextFile;
begin
  Result := False;
  if (FThread <> nil) or FileExists(LockFile) then
    Exit;
  AssignFile(F, LockFile);
  Rewrite(F);
  WriteLn(F, GetProcessID);
  CloseFile(F);
  FStartedAt := Now;
  if FCfg.Trainer = 'llama' then
    FThread := TLlamaTrainThread.Create(FCfg, ATier, ADeepId, ADeepPath)
  else
    FThread := TConsolidateThread.Create(FCfg);
  Result := True;
end;

function TConsolidator.Tier: string;
begin
  if FThread = nil then
    Result := ''
  else
    Result := FThread.Tier;
end;

procedure TConsolidator.Cancel;
begin
  if FThread <> nil then
    FThread.Terminate;
end;

function TConsolidator.Running: Boolean;
begin
  Result := (FThread <> nil) and not FThread.Finished;
end;

function TConsolidator.TakeResult(out AExitCode: Integer; out ASummary: string): Boolean;
begin
  Result := (FThread <> nil) and FThread.Finished;
  if not Result then
    Exit;
  FThread.WaitFor;
  AExitCode := FThread.ExitCode;
  ASummary := FThread.Summary;
  FreeAndNil(FThread);
  DeleteFile(LockFile);
end;

procedure TConsolidator.WaitFor;
begin
  if FThread <> nil then
    FThread.WaitFor;
end;

end.
