unit Mem.Consolidate;

(* Runs trainer/consolidate.py as a one-shot process on the training GPU,
   on a background thread so the chat keeps going while the LoRA trains.
   The trainer's last stdout line is a JSON summary; its exit code says
   what happened: 0 promoted, 2 rejected by the eval gate, 3 nothing to
   train, anything else is an error. *)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Process, Mem.Config;

type
  TConsolidateThread = class(TThread)
  private
    FCfg: TMemConfig;
    FExitCode: Integer;
    FSummary: string;
  protected
    procedure Execute; override;
  public
    constructor Create(ACfg: TMemConfig);
    property ExitCode: Integer read FExitCode;
    property Summary: string read FSummary;
  end;

  TConsolidator = class
  private
    FCfg: TMemConfig;
    FThread: TConsolidateThread;
    FStartedAt: TDateTime;
    function LockFile: string;
  public
    constructor Create(ACfg: TMemConfig);
    destructor Destroy; override;
    function Start: Boolean;
    function Running: Boolean;
    { True once a run has finished; hands back its result and resets }
    function TakeResult(out AExitCode: Integer; out ASummary: string): Boolean;
    procedure WaitFor;
    property StartedAt: TDateTime read FStartedAt;
  end;

implementation

uses
  Mem.Engine;

constructor TConsolidateThread.Create(ACfg: TMemConfig);
begin
  FCfg := ACfg;
  FExitCode := -1;
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

function TConsolidator.Start: Boolean;
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
  FThread := TConsolidateThread.Create(FCfg);
  Result := True;
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
