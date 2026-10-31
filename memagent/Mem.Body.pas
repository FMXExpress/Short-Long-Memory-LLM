unit Mem.Body;

(* A body: something memagent can ask to act in the world, next to the
   engine it talks with. Optional ([body] driver=none by default) and
   generic: the agent hands a body a task in language and gets back a stream
   of events (a subtask started, an observation, done, failed), and can stop
   it after any event.

   Drivers:
     mock     in-process, for tests and demos: a task "A, then B" runs as
              subtasks A and B; [body] args can give it plans,
              "goal => step; step | goal => ...", to stand in for a policy
              that decomposes a goal into subtasks of its own
     lerobot  body/lerobot_body.py: a LeRobot policy on a robot or in a
              simulator through LeRobot's RolloutController; with a policy
              that plans in language (FineART-VLA) it reports each subtask
     process  any program speaking the same protocol, [body] command=

   Protocol (one JSON object per line, stdin -> program, stdout -> memagent;
   the program's logs go elsewhere):
     -> {"op": "hello"}
     <- {"event": "ready", "name": "..", "plans": true}
     -> {"op": "act", "task": "..", "context": [".."], "rules": [".."]}
     <- {"event": "subtask" | "observation" | "info", "text": ".."}  (any number)
     <- {"event": "done" | "failed", "text": ".."}                    (ends the act)
     -> {"op": "stop"}       (during an act: stop moving; the act then ends)
     -> {"op": "quit"}

   What the agent does with the events (memory, the rule check that stops a
   subtask a prior forbids) is in Mem.Agent. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, Process, fpjson, jsonparser, Mem.Config;

type
  EBody = class(Exception);

  TBodyEventKind = (bkSubtask, bkObservation, bkInfo, bkDone, bkFailed);

  TBodyEvent = record
    Kind: TBodyEventKind;
    Text: string;
  end;

  { return False to stop the body }
  TBodyEventFunc = function(const AEvent: TBodyEvent): Boolean of object;

  TBody = class
  public
    function Describe: string; virtual; abstract;
    { True when it reports its own subtasks in language }
    function Plans: Boolean; virtual; abstract;
    (* runs ATask until done, failed, stopped (AOnEvent returned False) or
       AMaxSeconds; the last event (done or failed) is returned too *)
    function Act(const ATask: string; AContext, ARules: TStrings; AOnEvent: TBodyEventFunc;
      AMaxSeconds: Integer): TBodyEvent; virtual; abstract;
  end;

  TMockBody = class(TBody)
  private
    FPlans: TStringList;   { goal=steps }
  public
    constructor Create(const APlans: string);
    destructor Destroy; override;
    function Describe: string; override;
    function Plans: Boolean; override;
    function Act(const ATask: string; AContext, ARules: TStrings; AOnEvent: TBodyEventFunc;
      AMaxSeconds: Integer): TBodyEvent; override;
  end;

  TProcessBody = class(TBody)
  private
    FProc: TProcess;
    FBuf, FName, FLog: string;
    FPlans: Boolean;
    procedure Send(AObj: TJSONObject);
    { the next line, or '' after ATimeoutMs (-1: wait for ever) }
    function ReadLine(ATimeoutMs: Integer): string;
    function ReadEvent(ATimeoutMs: Integer; out AEvent: TBodyEvent): Boolean;
  public
    { AArgs[0] is the program; the rest its arguments }
    constructor Create(AArgs: TStrings; const ALogFile: string);
    destructor Destroy; override;
    function Describe: string; override;
    function Plans: Boolean; override;
    function Act(const ATask: string; AContext, ARules: TStrings; AOnEvent: TBodyEventFunc;
      AMaxSeconds: Integer): TBodyEvent; override;
  end;

{ [body] driver: nil for none }
function CreateBody(ACfg: TMemConfig): TBody;
{ a task split into its steps: "A, then B" / "A and then B" / "A; B" }
function TaskSteps(const ATask: string): TStringList;
(* what a finished step says about the world: "put the mug in the left
   cabinet" -> a fact "The mug is in the left cabinet." about "where the mug
   is"; nil if it says nothing *)
function FactFromStep(const AStep: string): TJSONObject;

implementation

uses
  RegExpr, DateUtils, Mem.Engine;

function Event(AKind: TBodyEventKind; const AText: string): TBodyEvent;
begin
  Result.Kind := AKind;
  Result.Text := AText;
end;

function TaskSteps(const ATask: string): TStringList;
var
  S: string;
  R: TRegExpr;
  P: Integer;
begin
  Result := TStringList.Create;
  R := TRegExpr.Create('\s*(?:,\s*and then|,\s*then|\band then|\bthen,?|;)\s+');
  R.ModifierI := True;
  try
    S := Trim(ATask);
    while (S <> '') and (S[Length(S)] in ['.', '!']) do
      SetLength(S, Length(S) - 1);
    while R.Exec(S) do
    begin
      P := R.MatchPos[0];
      if Trim(Copy(S, 1, P - 1)) <> '' then
        Result.Add(Trim(Copy(S, 1, P - 1)));
      S := Copy(S, P + R.MatchLen[0], MaxInt);
    end;
    if Trim(S) <> '' then
      Result.Add(Trim(S));
  finally
    R.Free;
  end;
end;

function FactFromStep(const AStep: string): TJSONObject;
var
  R: TRegExpr;
  LThing, LPrep, LPlace, LSubject, LVerb: string;
begin
  Result := nil;
  R := TRegExpr.Create('^(?:put|place|move|store|leave|hang|set)\s+(?:the\s+|my\s+|a\s+)?(.+?)\s+' +
    '(in|into|inside|on|onto|under|next to|behind)\s+(.+)$');
  R.ModifierI := True;
  try
    if not R.Exec(Trim(AStep)) then
      Exit;
    LThing := Trim(R.Match[1]);
    LPrep := LowerCase(R.Match[2]);
    if LPrep = 'into' then
      LPrep := 'in';
    if LPrep = 'onto' then
      LPrep := 'on';
    LPlace := Trim(R.Match[3]);
    while (LPlace <> '') and (LPlace[Length(LPlace)] in ['.', '!']) do
      SetLength(LPlace, Length(LPlace) - 1);
    if (LThing = '') or (LPlace = '') then
      Exit;
    LSubject := 'where the ' + LThing + ' is';
    { "the keys are", "the mug is" }
    if (Length(LThing) > 2) and (LThing[Length(LThing)] = 's') and (LThing[Length(LThing) - 1] <> 's') then
      LVerb := ' are '
    else
      LVerb := ' is ';
    Result := TJSONObject.Create(['text', 'The ' + LThing + LVerb + LPrep + ' ' + LPlace + '.',
      'subject', LSubject, 'value', LPlace,
      'questions', TJSONArray.Create(['Where is the ' + LThing + '?', 'Where did you put the ' + LThing + '?',
        'Where can I find the ' + LThing + '?', 'Do you know where the ' + LThing + ' is?',
        'Where did the ' + LThing + ' go?', 'Remind me where the ' + LThing + ' is.']),
      'confidence', 0.8, 'kind', 'world']);
  finally
    R.Free;
  end;
end;

{ TMockBody }

function TMockBody.Describe: string;
begin
  Result := 'mock (runs each step of a task as a subtask, for tests)';
end;

function TMockBody.Plans: Boolean;
begin
  Result := True;
end;

constructor TMockBody.Create(const APlans: string);
var
  LPlan: string;
  P: Integer;
begin
  inherited Create;
  FPlans := TStringList.Create;
  for LPlan in SplitString(APlans, '|') do
  begin
    P := Pos('=>', LPlan);
    if P > 0 then
      FPlans.Values[LowerCase(Trim(Copy(LPlan, 1, P - 1)))] := Trim(Copy(LPlan, P + 2, MaxInt));
  end;
end;

destructor TMockBody.Destroy;
begin
  FPlans.Free;
  inherited Destroy;
end;

function TMockBody.Act(const ATask: string; AContext, ARules: TStrings; AOnEvent: TBodyEventFunc;
  AMaxSeconds: Integer): TBodyEvent;
var
  LSteps: TStringList;
  I: Integer;
begin
  if FPlans.Values[LowerCase(Trim(ATask))] <> '' then
    LSteps := TaskSteps(FPlans.Values[LowerCase(Trim(ATask))])
  else
    LSteps := TaskSteps(ATask);
  try
    if Assigned(AOnEvent) then
      AOnEvent(Event(bkInfo, Format('mock: %d step(s), %d rule(s), %d fact(s) of context',
        [LSteps.Count, ARules.Count, AContext.Count])));
    for I := 0 to LSteps.Count - 1 do
      if Assigned(AOnEvent) and not AOnEvent(Event(bkSubtask, LSteps[I])) then
        Exit(Event(bkFailed, 'stopped during: ' + LSteps[I]));
    Result := Event(bkDone, ATask);
  finally
    LSteps.Free;
  end;
end;

{ TProcessBody }

constructor TProcessBody.Create(AArgs: TStrings; const ALogFile: string);
var
  I: Integer;
  E: TBodyEvent;
begin
  inherited Create;
  if AArgs.Count = 0 then
    raise EBody.Create('no body program ([body] command)');
  FLog := ALogFile;
  FProc := TProcess.Create(nil);
  FProc.Executable := AArgs[0];
  for I := 1 to AArgs.Count - 1 do
    FProc.Parameters.Add(AArgs[I]);
  FProc.Parameters.Add('--log');
  FProc.Parameters.Add(ALogFile);
  InheritEnvironment(FProc);
  (* stderr is not read: the program sends its logs to --log, so a full
     stderr pipe cannot block it *)
  FProc.Options := [poUsePipes];
  try
    FProc.Execute;
  except
    on E2: Exception do
      raise EBody.CreateFmt('cannot start %s: %s', [AArgs[0], E2.Message]);
  end;
  Send(TJSONObject.Create(['op', 'hello']));
  { loading a policy can take minutes; a dead program ends the wait through EOF }
  if not ReadEvent(-1, E) or (E.Kind <> bkInfo) or (FName = '') then
    raise EBody.CreateFmt('body program did not start: %s (see %s)', [E.Text, FLog]);
end;

destructor TProcessBody.Destroy;
begin
  if (FProc <> nil) and FProc.Running then
  begin
    try
      Send(TJSONObject.Create(['op', 'quit']));
    except
      on Exception do ;
    end;
    if not FProc.WaitOnExit(10000) then
      FProc.Terminate(1);
  end;
  FProc.Free;
  inherited Destroy;
end;

procedure TProcessBody.Send(AObj: TJSONObject);
var
  L: string;
begin
  try
    L := AObj.AsJSON + #10;
  finally
    AObj.Free;
  end;
  FProc.Input.WriteBuffer(L[1], Length(L));
end;

function TProcessBody.ReadLine(ATimeoutMs: Integer): string;
var
  P, N: Integer;
  LChunk: array[0..4095] of Char;
  LUntil: TDateTime;
begin
  if ATimeoutMs >= 0 then
    LUntil := IncMilliSecond(Now, ATimeoutMs)
  else
    LUntil := 0;
  repeat
    P := Pos(#10, FBuf);
    if P > 0 then
    begin
      Result := TrimRight(Copy(FBuf, 1, P - 1));
      Delete(FBuf, 1, P);
      Exit;
    end;
    if (ATimeoutMs >= 0) and (FProc.Output.NumBytesAvailable = 0) then
    begin
      if not FProc.Running then
        raise EBody.CreateFmt('body program exited (see %s)', [FLog]);
      if Now >= LUntil then
        Exit('');
      Sleep(20);
      Continue;
    end;
    N := FProc.Output.Read(LChunk, SizeOf(LChunk));
    if N <= 0 then
      raise EBody.CreateFmt('body program exited (see %s)', [FLog]);
    SetLength(Result, N);
    Move(LChunk[0], Result[1], N);
    FBuf := FBuf + Result;
  until False;
end;

function TProcessBody.ReadEvent(ATimeoutMs: Integer; out AEvent: TBodyEvent): Boolean;
var
  LLine, LKind: string;
  LJson: TJSONData;
begin
  Result := False;
  AEvent := Event(bkInfo, '');
  LLine := ReadLine(ATimeoutMs);
  if LLine = '' then
    Exit;
  try
    LJson := GetJSON(LLine);
  except
    on Exception do
      raise EBody.CreateFmt('body program sent something that is not JSON: %s', [Copy(LLine, 1, 200)]);
  end;
  try
    if not (LJson is TJSONObject) then
      raise EBody.Create('body program sent a JSON value that is not an object');
    LKind := TJSONObject(LJson).Get('event', '');
    AEvent.Text := TJSONObject(LJson).Get('text', '');
    case LKind of
      'ready':
        begin
          FName := TJSONObject(LJson).Get('name', 'body');
          FPlans := TJSONObject(LJson).Get('plans', False);
          AEvent.Kind := bkInfo;
          AEvent.Text := 'ready: ' + FName;
        end;
      'subtask': AEvent.Kind := bkSubtask;
      'observation': AEvent.Kind := bkObservation;
      'done': AEvent.Kind := bkDone;
      'failed': AEvent.Kind := bkFailed;
    else
      AEvent.Kind := bkInfo;
    end;
    Result := True;
  finally
    LJson.Free;
  end;
end;

function TProcessBody.Describe: string;
begin
  Result := FName + IfThen(FPlans, ' (plans its own subtasks)', '');
end;

function TProcessBody.Plans: Boolean;
begin
  Result := FPlans;
end;

function TProcessBody.Act(const ATask: string; AContext, ARules: TStrings; AOnEvent: TBodyEventFunc;
  AMaxSeconds: Integer): TBodyEvent;
var
  LReq: TJSONObject;
  LCtx, LRules: TJSONArray;
  I: Integer;
  LUntil: TDateTime;
  LStopSent: Boolean;
  E: TBodyEvent;

  procedure StopOnce;
  begin
    if not LStopSent then
      Send(TJSONObject.Create(['op', 'stop']));
    LStopSent := True;
  end;

begin
  LCtx := TJSONArray.Create;
  for I := 0 to AContext.Count - 1 do
    LCtx.Add(AContext[I]);
  LRules := TJSONArray.Create;
  for I := 0 to ARules.Count - 1 do
    LRules.Add(ARules[I]);
  LReq := TJSONObject.Create(['op', 'act', 'task', ATask, 'context', LCtx, 'rules', LRules]);
  Send(LReq);
  LUntil := IncSecond(Now, AMaxSeconds);
  LStopSent := False;
  Result := Event(bkFailed, 'no result');
  repeat
    if not ReadEvent(250, E) then
    begin
      if (AMaxSeconds > 0) and (Now >= LUntil) and not LStopSent then
      begin
        if Assigned(AOnEvent) then
          AOnEvent(Event(bkInfo, Format('time limit (%d s): stopping', [AMaxSeconds])));
        StopOnce;
      end;
      Continue;
    end;
    if E.Kind in [bkDone, bkFailed] then
      Exit(E);
    if Assigned(AOnEvent) and not AOnEvent(E) then
      StopOnce;
  until False;
end;

function SplitArgs(const S: string): TStringList;
var
  I: Integer;
  W: string;
  LQuote: Char;
begin
  { whitespace-separated, with "..." or '...' grouping }
  Result := TStringList.Create;
  W := '';
  LQuote := #0;
  for I := 1 to Length(S) do
  begin
    if LQuote <> #0 then
    begin
      if S[I] = LQuote then
        LQuote := #0
      else
        W := W + S[I];
    end
    else if S[I] in ['"', ''''] then
      LQuote := S[I]
    else if S[I] in [' ', #9] then
    begin
      if W <> '' then
        Result.Add(W);
      W := '';
    end
    else
      W := W + S[I];
  end;
  if W <> '' then
    Result.Add(W);
end;

function CreateBody(ACfg: TMemConfig): TBody;
var
  LArgs, LMore: TStringList;
  LLog: string;
begin
  Result := nil;
  LLog := IncludeTrailingPathDelimiter(ACfg.LogDir) + 'body.log';
  case ACfg.BodyDriver of
    'none', '': ;
    'mock': Result := TMockBody.Create(ACfg.BodyArgs);
    'lerobot', 'process':
      begin
        if ACfg.BodyDriver = 'lerobot' then
        begin
          LArgs := TStringList.Create;
          LArgs.Add(ACfg.PythonExe);
          LArgs.Add('-u');
          LArgs.Add(IncludeTrailingPathDelimiter(ACfg.RepoDir) + 'body' + PathDelim + 'lerobot_body.py');
        end
        else
          LArgs := SplitArgs(ACfg.BodyCommand);
        LMore := SplitArgs(ACfg.BodyArgs);
        try
          LArgs.AddStrings(LMore);
          Result := TProcessBody.Create(LArgs, LLog);
        finally
          LMore.Free;
          LArgs.Free;
        end;
      end;
  else
    raise EBody.CreateFmt('[body] driver must be none, mock, lerobot or process, not "%s"', [ACfg.BodyDriver]);
  end;
end;

end.
