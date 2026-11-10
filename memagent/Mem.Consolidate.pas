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
   while a turn is being answered (HoldTraining).

   With [consolidate] train_when = idle (layout one-gpu-idle) a run first waits
   until nothing has happened for idle_min minutes (and the clock is inside
   night, when set), then unloads the chat model (the release callback) so it
   has the device to itself. The next message stops the run (exit 4, redone at
   the next idle time) and the engine loads its model back. The handover is one
   lock: the trainer releases the chat model only while no line is being
   handled, and a line takes the device back before it touches the engine. *)

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
  protected
    { train_when = idle: waits for idle time, then releases the chat model; False when cancelled }
    function AwaitIdle: Boolean;
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

type
  TReleaseProc = procedure of object;

{ idle training: the callback that unloads the chat model (the engine's Release) }
procedure SetReleaseChat(AProc: TReleaseProc);
(* a line (message or command) starts: marks activity and keeps an idle run from
   taking the device; True when the chat model had been released for training,
   which the caller then stops (unless the line asked for it) *)
function BeginActivity: Boolean;
procedure EndActivity;
{ the chat model is loaded again (or the run that released it is over) }
procedure ChatBack;
function ChatReleased: Boolean;
{ /sleep, /wait, /nap, quitting: an idle run starts without waiting for idle time }
procedure TrainNow(AOn: Boolean);
{ an idle run is waiting for idle time }
function WaitingForIdle: Boolean;

implementation

uses
  DateUtils, Mem.Engine, Mem.Train, Mem.Layout;

var
  GHold: LongInt = 0;
  GDevice: TRTLCriticalSection;
  GRelease: TReleaseProc = nil;
  GReleased: Boolean = False;
  GBusy: Integer = 0;
  GLastActivity: TDateTime = 0;
  GTrainNow: Boolean = False;
  GWaiting: LongInt = 0;

procedure SetReleaseChat(AProc: TReleaseProc);
begin
  EnterCriticalSection(GDevice);
  GRelease := AProc;
  LeaveCriticalSection(GDevice);
end;

function BeginActivity: Boolean;
begin
  EnterCriticalSection(GDevice);
  try
    Inc(GBusy);
    GLastActivity := Now;
    Result := GReleased;
  finally
    LeaveCriticalSection(GDevice);
  end;
end;

procedure EndActivity;
begin
  EnterCriticalSection(GDevice);
  try
    if GBusy > 0 then
      Dec(GBusy);
    GLastActivity := Now;
  finally
    LeaveCriticalSection(GDevice);
  end;
end;

procedure ChatBack;
begin
  EnterCriticalSection(GDevice);
  GReleased := False;
  LeaveCriticalSection(GDevice);
end;

function ChatReleased: Boolean;
begin
  EnterCriticalSection(GDevice);
  Result := GReleased;
  LeaveCriticalSection(GDevice);
end;

procedure TrainNow(AOn: Boolean);
begin
  EnterCriticalSection(GDevice);
  GTrainNow := AOn;
  LeaveCriticalSection(GDevice);
end;

function WaitingForIdle: Boolean;
begin
  Result := InterLockedExchangeAdd(GWaiting, 0) <> 0;
end;

{ inside GDevice }
function IdleNow(ACfg: TMemConfig): Boolean;
begin
  if GTrainNow then
    Exit(True);
  Result := (GBusy = 0) and (InterLockedExchangeAdd(GHold, 0) = 0) and
    (MilliSecondsBetween(Now, GLastActivity) >= Round(ACfg.IdleMin * 60000)) and InNightWindow(ACfg.NightWindow, Now);
end;

function TTrainJob.AwaitIdle: Boolean;
var
  LGot: Boolean;
begin
  Result := True;
  if FCfg.TrainWhen <> 'idle' then
    Exit;
  try
    repeat
      if Terminated then
        Exit(False);
      EnterCriticalSection(GDevice);
      try
        LGot := IdleNow(FCfg);
        if LGot then
        begin
          { /wait or quitting holds GBusy while it waits for this run: the device is ours anyway }
          if Assigned(GRelease) and not GReleased then
            GRelease();
          GReleased := True;
        end;
      finally
        LeaveCriticalSection(GDevice);
      end;
      if LGot then
        Exit(True);
      Sleep(500);
    until False;
  finally
    InterLockedExchange(GWaiting, 0);
  end;
end;

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
  if not AwaitIdle then
  begin
    FSummary := '{"exit":4,"reason":"cancelled while waiting for idle time"}';
    FExitCode := 4;
    Exit;
  end;
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
  if (FThread <> nil) or FileExists(LockFile) or (FCfg.Trainer = 'none') then
    Exit;
  AssignFile(F, LockFile);
  Rewrite(F);
  WriteLn(F, GetProcessID);
  CloseFile(F);
  FStartedAt := Now;
  { waiting from the start, so /status says so before the thread gets going }
  if FCfg.TrainWhen = 'idle' then
    InterLockedExchange(GWaiting, 1);
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

initialization
  InitCriticalSection(GDevice);

finalization
  DoneCriticalSection(GDevice);

end.
