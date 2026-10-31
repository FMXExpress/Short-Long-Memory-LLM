unit Mem.Agent;

(* The agent loop. Per user turn:
     1. promote a finished consolidation (hot-swap the new LoRA)
     2. retrieve unbaked facts + older messages (FTS5) for the prompt
     3. persona + standing instructions (priors) + summary + memory + recent
        turns -> generate (streamed)
     4. optional tool calls (bigger profiles only)
     5. persist the turn, extract facts from the user message
     6. maybe start a background consolidation on the training GPU

   Memory tiers ([consolidate] tiers): the deep LoRA holds what the last deep
   sleep baked; a fast LoRA on top of it holds the facts since. Whenever the
   trainer is free and a fact is in no adapter yet, a fast run retrains the
   fast LoRA on all of them (the first run, with no deep LoRA yet, is a deep
   one). Once the fast LoRA holds deep_after facts, or a fact was forgotten
   or changed, a deep sleep rebuilds the deep LoRA from every fact and the
   fast one empties. Training waits while a turn is answered.

   A body (Mem.Body, optional) acts on /do <task>: after /confirm it gets the
   task with the relevant facts and the rules (negative priors that apply to
   a body). Every step it reports is checked against those rules, and one
   that breaks a rule stops it: a backstop, since a body announces a step as
   it starts it, not a substitute for limits in the robot itself. What it did
   goes into the conversation and, for "put X in Y" steps, into facts. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, DateUtils, StrUtils, Math, fpjson, jsonparser,
  Mem.Config, Mem.Store, Mem.Engine, Mem.Consolidate, Mem.Retrieval, Mem.Recall, Mem.Guard, Mem.Priors,
  Mem.Body;

type
  TAgent = class
  private
    FCfg: TMemConfig;
    FStore: TMemStore;
    FEngine: TEngine;
    FCons: TConsolidator;
    FRet: TRetrieval;
    FGuard: TRetrieval;   { bge cross-encoder for Mem.Guard; nil = guard off }
    FLastRecall: TRecallTiming;
    { per-turn stage timing (MEMAGENT_TIMING=1 prints it; /status shows averages) }
    FTiming: Boolean;
    FTurnStart, FGenStart: TDateTime;
    FFirstTokenMs: Double;
    FTurns: Integer;
    FSum: array[0..6] of Double;   { recall, extract, ttft, generate, tools, summarize, total }
    FSession: string;
    FQuiet: Boolean;
    FBody: TBody;
    FPendingTask: string;
    FBodyRules: TPriorArray;    { the rules of the act in progress }
    FBodyStopped: string;       { why the agent stopped the body, '' when it did not }
    FBodySteps: TStringList;
    function BodyRules: TPriorArray;
    function OnBodyEvent(const AEvent: TBodyEvent): Boolean;
    procedure RunBody(const ATask: string);
    procedure Mark(var ALast: TDateTime; out AMs: Double);
    procedure OnToken(const AText: string);
    function BuildMessages(const AUser: string; AUseRag: Boolean): TJSONArray;
    procedure Retrieve(const AQuery: string; ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray);
    function MemoryBlock(const AUser: string; ABeforeId: Int64): string;
    procedure EmbedFact(AId: Int64; const AText, ASubject: string);
    procedure EmbedMessage(AId: Int64; const AText: string);
    procedure Backfill;
    function SameSubject(const ASubject: string): Int64;
    function NearDuplicate(AFact: TJSONObject): Int64;
    function RunTools(const AReply: string; AMessages: TJSONArray): Boolean;
    { stores the durable facts in AText; returns 'stored: ...' / 'already known: ...'
      items joined by '; ', or '' when there were none }
    function ExtractFacts(const AUser: string; AMsgId: Int64; const AMode: string = 'auto'): string;
    { stores extracted facts (dedupe, supersede, embeddings); frees nothing }
    function StoreFacts(AFacts: TJSONArray; AMsgId: Int64): string;
    procedure MaybeSummarize;
    { standing instructions in the user's message: new, confirmed or contradicted (Mem.Priors) }
    procedure ObservePriors(const AUser: string; AMsgId: Int64);
    { the priors that go into the prompt: confident enough, not overridden by a
      more confident (or, at equal confidence, newer) opposite one, at most max_in_prompt }
    function PromptPriors: TPriorArray;
    function PriorsBlock: string;
    function PriorLine(const P: TPriorRec): string;
    procedure MaybeAutoConsolidate;
    { a background run: deep, or fast on top of the active deep adapter }
    function StartRun(ATier: string; const AWhy: string): Boolean;
    function PromoteManifest(const ADir: string): Boolean;
    function PromoteFast(const ADir: string; AMan: TJSONObject): Boolean;
    function DeepDue: Boolean;
    procedure ClearReady;
    procedure Say(const S: string);
  public
    constructor Create(ACfg: TMemConfig; AStore: TMemStore; AEngine: TEngine; ACons: TConsolidator;
      ARet: TRetrieval; AGuard: TRetrieval = nil);
    procedure Startup;
    procedure Turn(const AUser: string);
    procedure PollConsolidation;
    function Command(const ALine: string): Boolean; { False = quit }
    { optional; not owned }
    property Body: TBody read FBody write FBody;
  end;

implementation

const
  ToolsBlock =
    'You can call a tool by replying with exactly one line:' + LineEnding +
    '<tool>{"name": "remember", "args": {"fact": "<a fact about the user, e.g. My sister is called Ana>"}}</tool>' + LineEnding +
    '<tool>{"name": "memory_search", "args": {"query": "<words>"}}</tool>' + LineEnding +
    '<tool>{"name": "now", "args": {}}</tool>' + LineEnding +
    'Only call a tool when it is needed; otherwise answer normally.';

function Msg(const ARole, AContent: string): TJSONObject;
begin
  Result := TJSONObject.Create(['role', ARole, 'content', AContent]);
end;

constructor TAgent.Create(ACfg: TMemConfig; AStore: TMemStore; AEngine: TEngine; ACons: TConsolidator;
  ARet: TRetrieval; AGuard: TRetrieval);
begin
  FGuard := AGuard;
  inherited Create;
  FCfg := ACfg;
  FStore := AStore;
  FEngine := AEngine;
  FCons := ACons;
  FRet := ARet;
  FSession := FormatDateTime('yyyymmdd"T"hhnnss', Now);
  FTiming := GetEnvironmentVariable('MEMAGENT_TIMING') = '1';
end;

procedure TAgent.Say(const S: string);
begin
  WriteLn(S);
  Flush(Output);
end;

procedure TAgent.Mark(var ALast: TDateTime; out AMs: Double);
var
  LNow: TDateTime;
begin
  LNow := Now;
  AMs := MilliSecondsBetween(LNow, ALast);
  ALast := LNow;
end;

procedure TAgent.OnToken(const AText: string);
begin
  if FFirstTokenMs < 0 then
    FFirstTokenMs := MilliSecondsBetween(Now, FTurnStart);
  if FQuiet then
    Exit;
  Write(AText);
  Flush(Output);
end;

procedure TAgent.Startup;
var
  LPath, LReadyId, LReadyPath: string;
  LReady: TJSONData;
  L: TStringList;
begin
  Backfill;
  LPath := FStore.GetMeta('active_adapter_path');
  if (LPath <> '') and DirectoryExists(LPath) then
  begin
    FEngine.LoadAdapter(LPath, FStore.GetMeta('active_adapter'));
    Say('[memory] long-term adapter: ' + FEngine.ActiveAdapter);
    LPath := FStore.GetMeta('fast_adapter_path');
    if FCfg.Tiers and (LPath <> '') and DirectoryExists(LPath) then
    begin
      FEngine.LoadFastAdapter(LPath, FStore.GetMeta('fast_adapter'));
      Say('[memory] fast adapter: ' + FEngine.FastAdapter);
    end;
  end;
  (* A consolidation that finished after the last exit is still waiting in
     ready.json; promote it now so its training is not wasted. An adapter
     already in the history was handled before (promoted, or rolled back
     from) and must not come back. *)
  if FileExists(IncludeTrailingPathDelimiter(FCfg.AdapterDir) + 'ready.json') then
  begin
    L := TStringList.Create;
    try
      L.LoadFromFile(IncludeTrailingPathDelimiter(FCfg.AdapterDir) + 'ready.json');
      LReady := GetJSON(L.Text);
      try
        LReadyId := TJSONObject(LReady).Get('id', '');
        LReadyPath := TJSONObject(LReady).Get('path', '');
      finally
        LReady.Free;
      end;
    finally
      L.Free;
    end;
    if (LReadyId <> '') and not FStore.AdapterKnown(LReadyId) and DirectoryExists(LReadyPath) then
      PromoteManifest(LReadyPath);
    ClearReady;
  end;
end;

procedure TAgent.ClearReady;
begin
  { ready.json means "trained, not yet promoted"; once handled it must go }
  DeleteFile(IncludeTrailingPathDelimiter(FCfg.AdapterDir) + 'ready.json');
end;

procedure TAgent.Retrieve(const AQuery: string; ABeforeId: Int64; out AFacts: TFactArray;
  out AMsgs: TMsgArray);
var
  LTiming: TRecallTiming;
begin
  RecallMemory(FStore, FRet, FCfg, AQuery, ABeforeId, AFacts, AMsgs, LTiming);
  FLastRecall := LTiming;
end;

function TAgent.MemoryBlock(const AUser: string; ABeforeId: Int64): string;
var
  LFacts: TFactArray;
  LMsgs: TMsgArray;
  I: Integer;
begin
  Result := '';
  Retrieve(AUser, ABeforeId, LFacts, LMsgs);
  for I := 0 to Min(FCfg.RagK, Length(LFacts)) - 1 do
    Result := Result + '- ' + LFacts[I].Text + LineEnding;
  for I := 0 to Min(FCfg.RagK, Length(LMsgs)) - 1 do
    Result := Result + '- Earlier the user said: "' + LMsgs[I].Content + '"' + LineEnding;
end;

procedure TAgent.EmbedFact(AId: Int64; const AText, ASubject: string);
begin
  if FRet.Tier <> 'off' then
    FStore.SetFactVectors(AId, FRet.Embed(FactDoc(ASubject, AText)), FRet.Embed(ASubject), FRet.EmbedderId);
end;

procedure TAgent.EmbedMessage(AId: Int64; const AText: string);
begin
  if FRet.Tier <> 'off' then
    FStore.SetMessageVector(AId, FRet.Embed(AText), FRet.EmbedderId);
end;

procedure TAgent.Backfill;
var
  LPending: TPendingVectors;
  I: Integer;
begin
  { rows written before the embedder existed, or by a different embedder }
  if FRet.Tier = 'off' then
    Exit;
  LPending := FStore.PendingVectors(FRet.EmbedderId);
  if Length(LPending) = 0 then
    Exit;
  Say(Format('[memory] embedding %d stored facts/messages with %s...', [Length(LPending), FRet.EmbedderId]));
  for I := 0 to High(LPending) do
    if LPending[I].IsFact then
      EmbedFact(LPending[I].Id, LPending[I].Text, LPending[I].Subject)
    else
      EmbedMessage(LPending[I].Id, LPending[I].Text);
end;

function TAgent.NearDuplicate(AFact: TJSONObject): Int64;
var
  LSubjects: TSubjectArray;
  LVec: TVector;
  I: Integer;
begin
  (* The same fact arriving twice in different words, e.g. once through the
     remember tool and once from the user's own message ("Ana moved to Lisbon
     and works as a nurse" / "... works there as a nurse"), possibly under
     different subjects. Same value + near-identical text: skip the insert.
     Unlike SameSubject this only ever skips, never overwrites, so a false
     positive costs nothing. *)
  Result := 0;
  if FRet.Tier = 'off' then
    Exit;
  LSubjects := FStore.SubjectVectors(FRet.EmbedderId);
  LVec := nil;
  for I := 0 to High(LSubjects) do
    if SameText(Trim(LSubjects[I].Value), Trim(AFact.Get('value', ''))) then
    begin
      if LVec = nil then
        LVec := FRet.Embed(AFact.Get('text', ''));
      if Cosine(LVec, FRet.Embed(LSubjects[I].Text)) >= FCfg.DedupTextCos then
        Exit(LSubjects[I].Id);
    end;
end;

function TAgent.SameSubject(const ASubject: string): Int64;
var
  LSubjects: TSubjectArray;
  LVec: TVector;
  LScores: TScoreArray;
  I: Integer;
  LBest: Single;
begin
  (* Is a new fact about a subject we already hold under other words ("what
     my dog is called" vs "my dog's name")? Cosine alone cannot be trusted
     here: MiniLM scores "my sister's name" vs "my brother's name" higher
     (0.85) than a real paraphrase (0.75), and a wrong merge silently
     overwrites a fact. So cosine only shortlists, and the cross-encoder
     must confirm; without a real encoder and a reranker we never merge. *)
  Result := 0;
  if not FRet.CanDedup or not FRet.HasReranker then
    Exit;
  LVec := FRet.Embed(ASubject);
  LSubjects := FStore.SubjectVectors(FRet.EmbedderId);
  LBest := FCfg.DedupMinRerank;
  for I := 0 to High(LSubjects) do
    if Cosine(LVec, LSubjects[I].Vec) >= FCfg.DedupMinCos then
    begin
      LScores := FRet.RerankScores(ASubject, [LSubjects[I].Subject]);
      if LScores[0] >= LBest then
      begin
        LBest := LScores[0];
        Result := LSubjects[I].Id;
      end;
    end;
end;

function Conf(const P: TPriorRec): Double;
begin
  Result := P.Alpha / (P.Alpha + P.Beta);
end;

function TAgent.PriorLine(const P: TPriorRec): string;
begin
  Result := Format('#%d %s (%.2f, %d message%s%s%s)', [P.Id, P.Belief, Conf(P), P.Evidence,
    IfThen(P.Evidence = 1, '', 's'), IfThen(P.Kind = 'set', ', set with /prior', ''),
    IfThen(P.Scope = 'all', ', chat + body', IfThen(P.Scope = 'body', ', body only', ''))]);
end;

function TAgent.BodyRules: TPriorArray;
var
  P: TPriorRec;
begin
  { what a body must not do: active negative priors that apply to a body }
  Result := nil;
  for P in FStore.Priors(True, FCfg.PriorMinConfidence) do
    if (P.Polarity < 0) and ((P.Scope = 'body') or (P.Scope = 'all')) then
      Result := Concat(Result, [P]);
end;

function TAgent.PromptPriors: TPriorArray;
var
  P, K: TPriorRec;
  LBeaten: Boolean;
begin
  Result := nil;
  if not FCfg.PriorsEnabled then
    Exit;
  { most confident first, newer first at equal confidence: the first of an opposite pair wins }
  for P in FStore.Priors(True, FCfg.PriorMinConfidence) do
  begin
    if Length(Result) >= FCfg.PriorsInPrompt then
      Break;
    if P.Scope = 'body' then
      Continue;
    LBeaten := False;
    for K in Result do
      if MatchPrior(P.Key, P.Polarity, K.Key, K.Polarity) = pmOpposite then
        LBeaten := True;
    if not LBeaten then
      Result := Concat(Result, [P]);
  end;
end;

function TAgent.PriorsBlock: string;
var
  LPriors: TPriorArray;
  I: Integer;
begin
  Result := '';
  LPriors := PromptPriors;
  if Length(LPriors) = 0 then
    Exit;
  Result := LineEnding + LineEnding + 'The user''s standing instructions (follow them unless the user says ' +
    'otherwise):';
  for I := 0 to High(LPriors) do
    Result := Result + LineEnding + '- ' + LPriors[I].Belief;
end;

procedure TAgent.ObservePriors(const AUser: string; AMsgId: Int64);
var
  LCands: TPriorCandidates;
  LAll: TPriorArray;
  I, J: Integer;
  LSame: Boolean;
  LId: Int64;

  procedure Report(APriorId: Int64; const AWhat: string);
  var
    P: TPriorRec;
  begin
    for P in FStore.Priors(False) do
      if P.Id = APriorId then
        Say('[prior ' + AWhat[1] + PriorLine(P) + '] ' + Copy(AWhat, 2, MaxInt) +
          IfThen(P.Deprecated, ', deprecated', ''));
  end;

begin
  if not FCfg.PriorsEnabled then
    Exit;
  LCands := DetectPriors(AUser);
  if Length(LCands) = 0 then
    Exit;
  LAll := FStore.Priors(False);
  for I := 0 to High(LCands) do
  begin
    LSame := False;
    for J := 0 to High(LAll) do
      case MatchPrior(LCands[I].Key, LCands[I].Polarity, LAll[J].Key, LAll[J].Polarity) of
        pmSame:
          begin
            LSame := True;
            if FStore.AddPriorEvidence(LAll[J].Id, AMsgId, 1, 'restated: ' + LCands[I].Belief,
              FCfg.PriorMinConfidence, FCfg.PriorDeprecateUnder) then
              Report(LAll[J].Id, '~confirmed');
          end;
        pmOpposite:
          if FStore.AddPriorEvidence(LAll[J].Id, AMsgId, -1, 'contradicted by: ' + LCands[I].Belief,
            FCfg.PriorMinConfidence, FCfg.PriorDeprecateUnder) then
            Report(LAll[J].Id, '-contradicted');
      end;
    if not LSame then
    begin
      { a remark starts at 1:1, and this message is its first evidence }
      LId := FStore.AddPrior(LCands[I].Belief, LCands[I].Key, 'learned', '', LCands[I].Scope,
        LCands[I].Polarity, 1, 1);
      FStore.AddPriorEvidence(LId, AMsgId, 1, 'said: ' + LCands[I].Belief, FCfg.PriorMinConfidence,
        FCfg.PriorDeprecateUnder);
      Say(Format('[prior +#%d] %s', [LId, LCands[I].Belief]));
    end;
  end;
end;

function TAgent.OnBodyEvent(const AEvent: TBodyEvent): Boolean;
const
  KINDS: array[TBodyEventKind] of string = ('subtask', 'observation', 'info', 'done', 'failed');
var
  P: TPriorRec;
begin
  Result := True;
  Say('[body] ' + KINDS[AEvent.Kind] + ': ' + AEvent.Text);
  FStore.AddMessage(FSession, 'body', KINDS[AEvent.Kind] + ': ' + AEvent.Text);
  if AEvent.Kind <> bkSubtask then
    Exit;
  for P in FBodyRules do
    if BreaksPrior(AEvent.Text, P.Key, P.Polarity) then
    begin
      FBodyStopped := Format('"%s" breaks prior #%d %s', [AEvent.Text, P.Id, P.Belief]);
      Say('[body] stopping: ' + FBodyStopped);
      Exit(False);
    end;
  FBodySteps.Add(AEvent.Text);
end;

procedure TAgent.RunBody(const ATask: string);
var
  LFacts: TFactArray;
  LMsgs: TMsgArray;
  LContext, LRules: TStringList;
  LFound: TJSONArray;
  LFact: TJSONObject;
  LEnd: TBodyEvent;
  LMsgId: Int64;
  I: Integer;
begin
  LContext := TStringList.Create;
  LRules := TStringList.Create;
  FBodySteps := TStringList.Create;
  try
    Retrieve(ATask, High(Int64), LFacts, LMsgs);
    for I := 0 to High(LFacts) do
      LContext.Add(LFacts[I].Text);
    FBodyRules := BodyRules;
    for I := 0 to High(FBodyRules) do
      LRules.Add(FBodyRules[I].Belief);
    FBodyStopped := '';
    LMsgId := FStore.AddMessage(FSession, 'user', '/do ' + ATask);
    try
      LEnd := FBody.Act(ATask, LContext, LRules, @OnBodyEvent, FCfg.BodyMaxSeconds);
    except
      on E: Exception do
        LEnd := Default(TBodyEvent);
    end;
    if FBodyStopped <> '' then
      Say('[body] stopped: ' + FBodyStopped)
    else if LEnd.Kind = bkDone then
      Say('[body] done: ' + LEnd.Text)
    else
      Say('[body] failed: ' + LEnd.Text);
    FStore.AddMessage(FSession, 'body', IfThen(LEnd.Kind = bkDone, 'done: ', 'failed: ') + LEnd.Text +
      IfThen(FBodyStopped = '', '', ' (stopped: ' + FBodyStopped + ')'));
    (* what the completed steps say about the world ("put the mug in the left
       cabinet": the mug is in the left cabinet) goes into memory like a fact
       the user stated; a stopped act keeps the steps before the stop *)
    LFound := TJSONArray.Create;
    try
      for I := 0 to FBodySteps.Count - 1 do
      begin
        if (LEnd.Kind <> bkDone) and (I = FBodySteps.Count - 1) then
          Break;   { the step it was on did not finish }
        LFact := FactFromStep(FBodySteps[I]);
        if LFact <> nil then
          LFound.Add(LFact);
      end;
      StoreFacts(LFound, LMsgId);
    finally
      LFound.Free;
    end;
  finally
    FreeAndNil(FBodySteps);
    LRules.Free;
    LContext.Free;
  end;
end;

function TAgent.BuildMessages(const AUser: string; AUseRag: Boolean): TJSONArray;
var
  LSys, LSummary, LMem: string;
  LUpto, LFirstRecent: Int64;
  LRecent: TMsgArray;
  LBudget, LUsed, I, LStart: Integer;
begin
  LSys := FCfg.Priors + PriorsBlock;
  if FCfg.Tools then
    LSys := LSys + LineEnding + LineEnding + ToolsBlock;
  LRecent := FStore.RecentMessages(FSession, FCfg.RecentTurns * 2);
  if Length(LRecent) > 0 then
    LFirstRecent := LRecent[0].Id
  else
    LFirstRecent := High(Int64);
  LSummary := FStore.GetSummary(FSession, LUpto);
  if LSummary <> '' then
    LSys := LSys + LineEnding + LineEnding + 'Earlier in this conversation: ' + LSummary;
  if AUseRag then
  begin
    LMem := MemoryBlock(AUser, LFirstRecent);
    if LMem <> '' then
      LSys := LSys + LineEnding + LineEnding + 'Relevant memory:' + LineEnding + TrimRight(LMem);
  end;

  (* Budget in rough tokens (4 chars each): drop the oldest recent turns
     until system + history + question + reply fit the context window. *)
  LBudget := FCfg.CtxTokens - FCfg.MaxNewTokens - (Length(LSys) + Length(AUser)) div 4;
  LStart := Length(LRecent);
  LUsed := 0;
  for I := High(LRecent) downto 0 do
  begin
    Inc(LUsed, Length(LRecent[I].Content) div 4 + 4);
    if LUsed > LBudget then
      Break;
    LStart := I;
  end;
  { a history must not open with an assistant turn }
  while (LStart <= High(LRecent)) and (LRecent[LStart].Role <> 'user') do
    Inc(LStart);

  Result := TJSONArray.Create;
  Result.Add(Msg('system', LSys));
  for I := LStart to High(LRecent) do
    Result.Add(Msg(LRecent[I].Role, LRecent[I].Content));
  Result.Add(Msg('user', AUser));
end;

function ToolCallJson(const AReply: string): string;
var
  P1, P2: Integer;
  T: string;
begin
  (* <tool>{...}</tool>, as the prompt asks. Small models often drop the tags
     and reply with the bare object (sometimes in a ```json fence); accept
     that too, but only when the whole reply is one object with a "name", so
     ordinary answers that merely contain braces are never run as tools. *)
  P1 := Pos('<tool>', AReply);
  P2 := Pos('</tool>', AReply);
  if (P1 > 0) and (P2 > P1) then
    Exit(Copy(AReply, P1 + 6, P2 - P1 - 6));
  Result := '';
  T := Trim(AReply);
  if Copy(T, 1, 3) = '```' then
  begin
    Delete(T, 1, 3);
    if LowerCase(Copy(T, 1, 4)) = 'json' then
      Delete(T, 1, 4);
    if Copy(T, Length(T) - 2, 3) = '```' then
      SetLength(T, Length(T) - 3);
    T := Trim(T);
  end;
  if (T <> '') and (T[1] = '{') and (T[Length(T)] = '}') and (Pos('"name"', T) > 0) then
    Result := T;
end;

function TAgent.RunTools(const AReply: string; AMessages: TJSONArray): Boolean;
var
  I: Integer;
  LCall: TJSONData;
  LName, LOut, LJson: string;
  LArgs: TJSONObject;
  LFacts: TFactArray;
begin
  Result := False;
  LJson := ToolCallJson(AReply);
  if LJson = '' then
    Exit;
  try
    LCall := GetJSON(LJson);
  except
    on E: Exception do
    begin
      LOut := 'invalid tool call: ' + E.Message;
      LCall := nil;
    end;
  end;
  if (LCall <> nil) and not (LCall is TJSONObject) then
  begin
    FreeAndNil(LCall);
    LOut := 'invalid tool call: expected a JSON object';
  end;
  if LCall <> nil then
    try
      LName := TJSONObject(LCall).Get('name', '');
      LArgs := nil;
      if TJSONObject(LCall).Find('args') is TJSONObject then
        LArgs := TJSONObject(TJSONObject(LCall).Find('args'));
      { small models drop "args" or send it as a string; treat that as no arguments }
      if LArgs = nil then
      begin
        LArgs := TJSONObject.Create;
        TJSONObject(LCall).Add('_args', LArgs);   { owned by LCall, freed with it }
      end;
      if LName = 'remember' then
      begin
        { report what was actually stored, so the model never claims a memory it does not have }
        LOut := ExtractFacts(LArgs.Get('fact', ''), 0);
        if LOut = '' then
          LOut := 'nothing stored: no durable fact found in that text';
      end
      else if LName = 'memory_search' then
      begin
        LFacts := FStore.SearchFacts(LArgs.Get('query', ''), 5, True);
        LOut := '';
        for I := 0 to High(LFacts) do
          LOut := LOut + LFacts[I].Text + ' ';
        if LOut = '' then
          LOut := 'no matching memories';
      end
      else if LName = 'now' then
        LOut := FormatDateTime('yyyy-mm-dd hh:nn', Now)
      else
        LOut := 'unknown tool ' + LName;
    finally
      LCall.Free;
    end;
  Say(LineEnding + '[tool] ' + LName + ' -> ' + LOut);
  AMessages.Add(Msg('assistant', AReply));
  AMessages.Add(Msg('user', 'Tool result: ' + LOut));
  Result := True;
end;

function TAgent.ExtractFacts(const AUser: string; AMsgId: Int64; const AMode: string): string;
var
  LFacts: TJSONArray;
begin
  LFacts := FEngine.ExtractFacts(AUser, AMode);
  try
    Result := StoreFacts(LFacts, AMsgId);
  finally
    LFacts.Free;
  end;
end;

function TAgent.StoreFacts(AFacts: TJSONArray; AMsgId: Int64): string;
var
  LFacts: TJSONArray;
  I: Integer;
  LFact: TJSONObject;
  LId, LDupId: Int64;
  LOldSubject: string;
begin
  Result := '';
  LFacts := AFacts;
  begin
    for I := 0 to LFacts.Count - 1 do
    begin
      LFact := LFacts.Objects[I];
      if LFact.Get('confidence', 0.0) < FCfg.MinFactConfidence then
        Continue;
      LDupId := NearDuplicate(LFact);
      if LDupId <> 0 then
      begin
        { the repeat may carry the user's own framing ("my sister") that the
          stored copy lacks (a remember tool call during the reply came first) }
        LOldSubject := FStore.FactSubject(LDupId);
        if SameText(Copy(LFact.Get('subject', ''), 1, 3), 'my ') and not SameText(Copy(LOldSubject, 1, 3), 'my ') then
        begin
          FStore.SetFactSubject(LDupId, LFact.Get('subject', ''));
          EmbedFact(LDupId, FStore.FactText(LDupId), LFact.Get('subject', ''));
        end;
        Result := Result + IfThen(Result = '', '', '; ') + 'already known: ' + LFact.Get('text', '');
        Continue;
      end;
      case FStore.AddFact(LFact, AMsgId, SameSubject(LFact.Get('subject', '')), LId) of
        afAdded: Say(Format('[memory +#%d] %s', [LId, LFact.Get('text', '')]));
        afSuperseded:
          begin
            Say(Format('[memory ~#%d] %s (replaces an older value)', [LId, LFact.Get('text', '')]));
            { the old value may be in the deep LoRA: the next deep sleep retrains without it }
            if FCfg.Tiers then
              FStore.SetMeta('forget_pending', '1');
          end;
        afDuplicate:
          begin
            Result := Result + IfThen(Result = '', '', '; ') + 'already known: ' + LFact.Get('text', '');
            Continue;
          end;
      end;
      Result := Result + IfThen(Result = '', '', '; ') + 'stored: ' + LFact.Get('text', '');
      EmbedFact(LId, LFact.Get('text', ''), LFact.Get('subject', ''));
    end;
  end;
end;

procedure TAgent.MaybeSummarize;
var
  LAll: TMsgArray;
  LUpto, LNewUpto: Int64;
  LOld, LText: string;
  I, LCut: Integer;
  LReq: TJSONArray;
begin
  if not FCfg.Summarize then
    Exit;
  LAll := FStore.RecentMessages(FSession, 100000);
  LCut := Length(LAll) - FCfg.RecentTurns * 2;   { messages before LCut fell out of the window }
  if LCut <= 0 then
    Exit;
  LOld := FStore.GetSummary(FSession, LUpto);
  LText := '';
  LNewUpto := LUpto;
  for I := 0 to LCut - 1 do
    if LAll[I].Id > LUpto then
    begin
      LText := LText + LAll[I].Role + ': ' + LAll[I].Content + LineEnding;
      LNewUpto := LAll[I].Id;
    end;
  if Length(LText) < 400 then
    Exit;   { not worth a model call yet }
  LReq := TJSONArray.Create;
  try
    { instructions as a system turn: in the user turn, small models echo the
      instruction's structure back as headings instead of summarising }
    LReq.Add(Msg('system', 'You maintain a running summary of a conversation for an assistant with ' +
      'memory. Write plain prose in the third person ("The user ..."), at most 80 words, keeping names, ' +
      'facts, plans and open tasks. Output only the summary text: no headings, no lists, no preamble.'));
    LReq.Add(Msg('user', 'Previous summary: ' + IfThen(LOld = '', '(none)', LOld) + LineEnding + LineEnding +
      'New messages:' + LineEnding + LText + LineEnding + 'Updated summary:'));
    FStore.SetSummary(FSession, Trim(FEngine.Generate(LReq, True, nil, 200)), LNewUpto);
  finally
    LReq.Free;
  end;
end;

procedure TAgent.Turn(const AUser: string);
var
  LMessages: TJSONArray;
  LReply: string;
  LUserId: Int64;
  I: Integer;
  LT: TDateTime;
  LMs: array[0..7] of Double;    { 7: model extraction after the reply }
  LRuleFacts, LWhy: string;
begin
  PollConsolidation;
  { the trainer waits between windows while this turn is answered }
  HoldTraining(True);
  try
  FTurnStart := Now;
  FFirstTokenMs := -1;
  LT := FTurnStart;
  { the prompt is built before the message is stored, so it is not also
    retrieved as "earlier the user said" for its own turn }
  LMessages := BuildMessages(AUser, True);
  try
    Mark(LT, LMs[0]);
    (* Facts from the user's own words are stored before the model answers:
       they carry the user's framing ("my sister ..."), and a remember tool
       call for the same fact then reports "already known" instead of adding
       a second, worse-phrased copy. *)
    LUserId := FStore.AddMessage(FSession, 'user', AUser);
    EmbedMessage(LUserId, AUser);
    LRuleFacts := ExtractFacts(AUser, LUserId, 'rules');
    ObservePriors(AUser, LUserId);
    Mark(LT, LMs[1]);
    if (FGuard <> nil) and (QuestionSubject(AUser) <> '') then
    begin
      (* A question about "my X": generate first, license, then show, so an
         answer that borrows another subject's stored value ("horse" ->
         the dog's name) is never displayed. *)
      LReply := FEngine.Generate(LMessages, True, nil);
      if not Licensed(FGuard, AUser, LReply, FStore.ActiveFacts, LWhy, FCfg.GuardMinScore) then
      begin
        Say('[guard] withheld an answer that used a stored value from: ' + LWhy);
        LReply := ABSTAIN_REPLY;
      end;
      FFirstTokenMs := MilliSecondsBetween(Now, FTurnStart);
      WriteLn('pal> ', LReply);
    end
    else
    begin
      Write('pal> ');
      LReply := FEngine.Generate(LMessages, True, @OnToken);
      WriteLn;
    end;
    LMs[2] := FFirstTokenMs;
    Mark(LT, LMs[3]);
    I := 0;
    while FCfg.Tools and (I < 3) and RunTools(LReply, LMessages) do
    begin
      Write('pal> ');
      LReply := FEngine.Generate(LMessages, True, @OnToken);
      WriteLn;
      Inc(I);
    end;
    Mark(LT, LMs[4]);
  finally
    LMessages.Free;
  end;
  FStore.AddMessage(FSession, 'assistant', LReply);
  finally
    HoldTraining(False);
  end;
  (* The model extraction pass runs after the reply: on the small profile it
     is a 5-8 s generation, and before the reply it held back the first token
     (measured 8-11 s). The worker skips it for questions and greetings. *)
  if LRuleFacts = '' then
    ExtractFacts(AUser, LUserId, 'llm');
  Mark(LT, LMs[7]);
  MaybeSummarize;
  Mark(LT, LMs[5]);
  LMs[1] := LMs[1] + LMs[7];
  LMs[6] := MilliSecondsBetween(Now, FTurnStart);
  Inc(FTurns);
  for I := 0 to 6 do
    FSum[I] := FSum[I] + LMs[I];
  if FTiming then
    Say(Format('[timing] recall %.0f (embed %.0f search %.0f rerank %.0f)  extract %.0f  first-token %.0f  ' +
      'generate %.0f  tools %.0f  summary %.0f  total %.0f ms', [LMs[0], FLastRecall.EmbedMs,
      FLastRecall.SearchMs, FLastRecall.RerankMs, LMs[1], LMs[2], LMs[3], LMs[4], LMs[5], LMs[6]]));
  MaybeAutoConsolidate;
end;

{ '1 fact', '3 facts' }
function Facts(N: Integer): string;
begin
  Result := IntToStr(N) + IfThen(N = 1, ' fact', ' facts');
end;

function TAgent.DeepDue: Boolean;
begin
  Result := (FStore.GetMeta('forget_pending') = '1') or
    ((FEngine.ActiveAdapter <> '') and (FStore.FastTierCount(FEngine.ActiveAdapter) >= FCfg.DeepAfter));
end;

procedure TAgent.MaybeAutoConsolidate;
var
  LLast: Int64;
  LIntervalOk: Boolean;
  LNew: Integer;
begin
  if not FCfg.AutoConsolidate or FCons.Running then
    Exit;
  LLast := StrToInt64Def(FStore.GetMeta('last_consolidation', '0'), 0);
  LIntervalOk := NowUnix - LLast >= Int64(FCfg.IntervalMin) * 60;
  if FCfg.Tiers then
  begin
    { the interval spaces deep sleeps; fast runs go whenever there is something new }
    if LIntervalOk and DeepDue then
      StartRun('deep', IfThen(FStore.GetMeta('forget_pending') = '1', 'a fact was forgotten or changed',
        'the fast LoRA holds ' + Facts(FStore.FastTierCount(FEngine.ActiveAdapter))))
    else
    begin
      LNew := FStore.NewFactsAfter(StrToInt64Def(FStore.GetMeta('fast_seen_id', '0'), 0));
      if LNew >= FCfg.FastMinNew then
        StartRun('fast', IfThen(LNew = 1, '1 new fact', IntToStr(LNew) + ' new facts'));
    end;
    Exit;
  end;
  if not LIntervalOk then
    Exit;
  if (FStore.UnbakedCount < FCfg.MinNewFacts) and (FStore.GetMeta('forget_pending') <> '1') then
    Exit;
  if FCons.Start then
    Say(Format('[sleep] %d new facts: consolidating into a new LoRA in the background (%s)',
      [FStore.UnbakedCount, DeviceName(FCfg.TrainGpu)]));
end;

function TAgent.StartRun(ATier: string; const AWhy: string): Boolean;
begin
  { a fast LoRA needs a deep one under it: before the first deep sleep, every run is deep }
  if (ATier = 'fast') and (FEngine.ActiveAdapter = '') then
    ATier := 'deep';
  FStore.SetMeta('fast_seen_id', IntToStr(FStore.MaxFactId));
  Result := FCons.Start(ATier, FEngine.ActiveAdapter, FStore.GetMeta('active_adapter_path'));
  if not Result then
    Exit;
  if ATier = 'deep' then
    Say(Format('[sleep] deep sleep (%s): rebuilding the long-term LoRA from %s in the background (%s)',
      [AWhy, Facts(Length(FStore.ActiveFacts)), DeviceName(FCfg.TrainGpu)]))
  else
    Say(Format('[nap] %s: training the fast LoRA on %s on top of %s in the background',
      [AWhy, Facts(FStore.FastTierCount(FEngine.ActiveAdapter)), FEngine.ActiveAdapter]));
end;

function TAgent.PromoteManifest(const ADir: string): Boolean;
var
  L: TStringList;
  LMan: TJSONObject;
  LId, LOld: string;
  LT: TDateTime;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(IncludeTrailingPathDelimiter(ADir) + 'manifest.json');
    LMan := GetJSON(L.Text) as TJSONObject;
  finally
    L.Free;
  end;
  try
    if LMan.Get('tier', 'deep') = 'fast' then
      Exit(PromoteFast(ADir, LMan));
    LId := LMan.Get('id', '');
    LOld := FEngine.ActiveAdapter;
    LT := Now;
    FEngine.LoadAdapter(ADir, LId);
    { the deep LoRA was rebuilt from every fact: the fast one is empty again }
    if FEngine.FastAdapter <> '' then
    begin
      FStore.RetireAdapter(FEngine.FastAdapter);
      FEngine.ClearFastAdapter;
    end;
    FStore.SetMeta('fast_adapter', '');
    FStore.SetMeta('fast_adapter_path', '');
    FStore.SetMeta('last_consolidation', IntToStr(NowUnix));
    if LOld <> '' then
      FStore.RetireAdapter(LOld);
    FStore.RecordAdapter(LId, ADir, 'active', LMan.Get('n_facts', 0), LMan.Get('n_passed', 0),
      LMan.Get('recall', 0.0), LMan.Get('ppl_ratio', 1.0));
    FStore.ApplyManifest(LId, LMan.Arrays['passed_fact_ids']);
    FStore.SetMeta('active_adapter', LId);
    FStore.SetMeta('active_adapter_path', ADir);
    FStore.SetMeta('forget_pending', '0');
    Say(Format('[memory] hot-swapped LoRA %s -> %s in %d ms: %d/%d facts now in the weights ' +
      '(recall %.2f, held-out ppl x%.2f)', [IfThen(LOld = '', 'none', LOld), LId,
      MilliSecondsBetween(Now, LT), LMan.Get('n_passed', 0), LMan.Get('n_facts', 0),
      LMan.Get('recall', 0.0), LMan.Get('ppl_ratio', 1.0)]));
    Result := True;
  finally
    LMan.Free;
  end;
end;

function TAgent.PromoteFast(const ADir: string; AMan: TJSONObject): Boolean;
var
  LId, LOld: string;
  LT: TDateTime;
begin
  Result := False;
  LId := AMan.Get('id', '');
  if AMan.Get('deep', '') <> FEngine.ActiveAdapter then
  begin
    Say(Format('[nap] fast LoRA %s was trained on top of %s, which is no longer active; dropped',
      [LId, AMan.Get('deep', '')]));
    Exit;
  end;
  LOld := FEngine.FastAdapter;
  LT := Now;
  FEngine.LoadFastAdapter(ADir, LId);
  if LOld <> '' then
    FStore.RetireAdapter(LOld);
  FStore.RecordAdapter(LId, ADir, 'active', AMan.Get('n_facts', 0), AMan.Get('n_passed', 0),
    AMan.Get('recall', 0.0), AMan.Get('ppl_ratio', 1.0));
  FStore.ApplyFastManifest(LId, AMan.Arrays['fact_ids'], AMan.Arrays['passed_fact_ids']);
  FStore.SetMeta('fast_adapter', LId);
  FStore.SetMeta('fast_adapter_path', ADir);
  Say(Format('[memory] fast LoRA %s -> %s in %d ms: %d/%d new facts in the weights on top of %s ' +
    '(%d/%d earlier facts kept, held-out ppl x%.2f)', [IfThen(LOld = '', 'none', LOld), LId,
    MilliSecondsBetween(Now, LT), AMan.Get('n_passed', 0), AMan.Get('n_facts', 0), FEngine.ActiveAdapter,
    AMan.Get('replay_passed', 0), AMan.Get('replay', 0), AMan.Get('ppl_ratio', 1.0)]));
  Result := True;
end;

procedure TAgent.PollConsolidation;
var
  LCode: Integer;
  LSummary, LTier: string;
  LJson: TJSONData;
begin
  LTier := FCons.Tier;
  if not FCons.TakeResult(LCode, LSummary) then
    Exit;
  { the interval counts from the last deep sleep (or, without tiers, any run) }
  if (LTier <> 'fast') and (LCode <> 4) then
    FStore.SetMeta('last_consolidation', IntToStr(NowUnix));
  LJson := nil;
  try
    try
      LJson := GetJSON(LSummary);
    except
      on Exception do LJson := nil;
    end;
    case LCode of
      0:
        begin
          PromoteManifest(TJSONObject(LJson).Get('path', ''));
          ClearReady;
        end;
      2: if LTier = 'fast' then
           Say(Format('[nap] fast LoRA rejected by the eval gate (recall %.2f, ppl x%.2f); the new facts ' +
             'stay in RAG', [TJSONObject(LJson).Get('recall', 0.0), TJSONObject(LJson).Get('ppl_ratio', 0.0)]))
         else
           Say(Format('[sleep] new LoRA rejected by the eval gate (recall %.2f, ppl x%.2f); keeping %s',
             [TJSONObject(LJson).Get('recall', 0.0), TJSONObject(LJson).Get('ppl_ratio', 0.0),
              IfThen(FEngine.ActiveAdapter = '', 'the base model', FEngine.ActiveAdapter)]));
      3: if LTier <> 'fast' then
           Say('[sleep] nothing to consolidate');
      4: ;  { cancelled for a deep sleep }
    else
      if LJson is TJSONObject then
        LSummary := TJSONObject(LJson).Get('reason', LSummary);
      Say('[sleep] consolidation failed (exit ' + IntToStr(LCode) + '): ' + LSummary +
        ' -- see ' + IncludeTrailingPathDelimiter(FCfg.LogDir) + 'consolidate.log');
    end;
  finally
    LJson.Free;
  end;
end;

function TAgent.Command(const ALine: string): Boolean;
var
  LCmd, LArg, LPath, LPrev: string;
  P, I: Integer;
  LFacts: TFactArray;
  LMsgs: TMsgArray;
  LHist: TStringList;
  LMessages: TJSONArray;
  LId: Int64;
  LPriors, LInPrompt: TPriorArray;
  LPending: string;
  LSteps: TStringList;
  P2: TPriorRec;
  LIn: Boolean;
  LSub: string;
  LCand: TPriorCandidate;
begin
  Result := True;
  P := Pos(' ', ALine);
  if P > 0 then
  begin
    LCmd := LowerCase(Copy(ALine, 1, P - 1));
    LArg := Trim(Copy(ALine, P + 1, MaxInt));
  end
  else
  begin
    LCmd := LowerCase(ALine);
    LArg := '';
  end;

  if (LCmd = '/quit') or (LCmd = '/exit') then
    Exit(False)
  else if LCmd = '/help' then
    Say('/facts  /remember <text>  /forget <id>  /recall <query>  /ab <question>  /sleep  /nap  /wait' +
      '  /adapter  /rollback  /priors  /prior [--body] <text>|drop <id>|restore <id>  /prompt <text>' +
      '  /do <task>  /confirm  /cancel  /status  /quit')
  else if LCmd = '/facts' then
  begin
    LFacts := FStore.ActiveFacts;
    for I := 0 to High(LFacts) do
      Say(Format('  #%-3d %-45s %s', [LFacts[I].Id, LFacts[I].Text,
        IfThen(LFacts[I].BakedIn = '', '[in RAG]', '[baked: ' + LFacts[I].BakedIn + ']')]));
    if Length(LFacts) = 0 then
      Say('  (no facts yet)');
  end
  else if LCmd = '/do' then
  begin
    if FBody = nil then
      Say('[body] no body ([body] driver in memagent.ini)')
    else if LArg = '' then
      Say('/do <task>, e.g. /do put the mug in the left cabinet')
    else
    begin
      { a task that already breaks a rule is refused before anything moves }
      LPending := '';
      LSteps := TaskSteps(LArg);
      try
        for I := 0 to LSteps.Count - 1 do
          for P2 in BodyRules do
            if (LPending = '') and BreaksPrior(LSteps[I], P2.Key, P2.Polarity) then
              LPending := Format('"%s" breaks prior #%d %s', [LSteps[I], P2.Id, P2.Belief]);
      finally
        LSteps.Free;
      end;
      if LPending <> '' then
        Say('[body] refused: ' + LPending)
      else if FCfg.BodyConfirm then
      begin
        FPendingTask := LArg;
        Say(Format('[body] %s will: %s (%d rule(s) apply). /confirm to start, /cancel to drop it.',
          [FBody.Describe, LArg, Length(BodyRules)]));
      end
      else
        RunBody(LArg);
    end;
  end
  else if LCmd = '/confirm' then
  begin
    if FPendingTask = '' then
      Say('[body] nothing to confirm (/do <task> first)')
    else
    begin
      LPending := FPendingTask;
      FPendingTask := '';
      RunBody(LPending);
    end;
  end
  else if LCmd = '/cancel' then
  begin
    if FPendingTask = '' then
      Say('[body] nothing pending')
    else
      Say('[body] dropped: ' + FPendingTask);
    FPendingTask := '';
  end
  else if LCmd = '/prompt' then
  begin
    { the system prompt a message would get: persona, standing instructions, summary, memory }
    LMessages := BuildMessages(LArg, True);
    try
      Say(LMessages.Objects[0].Get('content', ''));
    finally
      LMessages.Free;
    end;
  end
  else if LCmd = '/priors' then
  begin
    LPriors := FStore.Priors(False);
    LInPrompt := PromptPriors;
    for I := 0 to High(LPriors) do
    begin
      LIn := False;
      for P2 in LInPrompt do
        if P2.Id = LPriors[I].Id then
          LIn := True;
      Say('  ' + PriorLine(LPriors[I]) + IfThen(LPriors[I].Deprecated, '  [deprecated: ' + LPriors[I].Reason + ']',
        IfThen(LIn, '', '  [not in the prompt]')));
    end;
    if Length(LPriors) = 0 then
      Say('  (no priors yet: say "from now on ...", "never ...", or /prior <text>)');
  end
  else if LCmd = '/prior' then
  begin
    P := Pos(' ', LArg);
    LSub := LowerCase(IfThen(P > 0, Copy(LArg, 1, P - 1), LArg));
    if ((LSub = 'drop') or (LSub = 'restore')) and (P > 0) then
    begin
      LId := StrToInt64Def(Trim(Copy(LArg, P + 1, MaxInt)), 0);
      if FStore.SetPriorDeprecated(LId, LSub = 'drop', IfThen(LSub = 'drop', 'dropped with /prior drop', '')) then
        Say(Format('[prior #%d] %s', [LId, IfThen(LSub = 'drop', 'deprecated', 'restored')]))
      else
        Say('no such prior');
    end
    else if LArg = '' then
      Say('/prior <instruction>  |  /prior drop <id>  |  /prior restore <id>   (/priors lists them)')
    else
    begin
      { set on purpose: starts at 3:1, so one stray remark does not undo it.
        --body: a rule for the body only, kept out of the chat prompt }
      if LSub = '--body' then
      begin
        LCand := PriorFromText(Trim(Copy(LArg, P + 1, MaxInt)));
        LCand.Scope := 'body';
      end
      else
        LCand := PriorFromText(LArg);
      LId := FStore.AddPrior(LCand.Belief, LCand.Key, 'set', '', LCand.Scope, LCand.Polarity, 3, 1);
      Say(Format('[prior +#%d] %s%s', [LId, LCand.Belief, IfThen(LCand.Scope = 'chat', '',
        '  (' + LCand.Scope + ')')]));
    end;
  end
  else if LCmd = '/remember' then
    ExtractFacts(LArg, 0)
  else if LCmd = '/forget' then
  begin
    LId := StrToInt64Def(LArg, 0);
    if FStore.ForgetFact(LId) then
    begin
      FStore.SetMeta('forget_pending', '1');
      Say(Format('[memory -#%d] removed from RAG; it leaves the weights at the next consolidation', [LId]));
    end
    else
      Say('no such fact');
  end
  else if (LCmd = '/ab') or (LCmd = '/nolora') then
  begin
    (* The demo moment: the same question with no RAG, first on the base
       model and then with the memory LoRA. Only the weights differ. *)
    LMessages := TJSONArray.Create;
    try
      LMessages.Add(Msg('system', FCfg.Priors));
      LMessages.Add(Msg('user', LArg));
      Write('base, no memory  > ');
      FEngine.Generate(LMessages, False, @OnToken);
      WriteLn;
      if FEngine.ActiveAdapter <> '' then
      begin
        Write('LoRA, no RAG     > ');
        FEngine.Generate(LMessages, True, @OnToken);
        WriteLn;
      end
      else
        Say('LoRA, no RAG     > (no adapter yet: /sleep first)');
    finally
      LMessages.Free;
    end;
  end
  else if (LCmd = '/sleep') and FCfg.Tiers then
  begin
    { a deep sleep over every fact now: a run already going is cancelled }
    if FCons.Running then
    begin
      Say('[sleep] stopping the ' + FCons.Tier + ' run for a deep sleep...');
      FCons.Cancel;
      FCons.WaitFor;
      PollConsolidation;
    end;
    StartRun('deep', '/sleep');
  end
  else if LCmd = '/sleep' then
  begin
    if FCons.Start then
      Say(Format('[sleep] consolidating %d active facts into a new LoRA (%s)...',
        [Length(FStore.ActiveFacts), DeviceName(FCfg.TrainGpu)]))
    else
      Say('[sleep] a consolidation is already running');
  end
  else if LCmd = '/nap' then
  begin
    if not FCfg.Tiers then
      Say('[nap] memory tiers are off ([consolidate] tiers); /sleep consolidates')
    else if FCons.Running then
      Say('[nap] a ' + FCons.Tier + ' run is already going')
    else
      StartRun('fast', '/nap');
  end
  else if LCmd = '/wait' then
  begin
    { until memory has caught up: with tiers a finished run can start the next }
    while FCons.Tier <> '' do   { running, or finished and not yet promoted }
    begin
      if FCons.Running then
        Say('[sleep] waiting for the ' + FCons.Tier + ' run to finish...');
      FCons.WaitFor;
      PollConsolidation;
      if FCfg.Tiers then
        MaybeAutoConsolidate;
    end;
    PollConsolidation;
  end
  else if LCmd = '/adapter' then
  begin
    Say('active: ' + IfThen(FEngine.ActiveAdapter = '', '(none)', FEngine.ActiveAdapter) +
      IfThen(FEngine.FastAdapter = '', '', ' + fast ' + FEngine.FastAdapter));
    LHist := FStore.AdapterHistory(10);
    try
      for I := 0 to LHist.Count - 1 do
        Say('  ' + LHist[I]);
    finally
      LHist.Free;
    end;
  end
  else if LCmd = '/rollback' then
  begin
    LPrev := FStore.PreviousAdapter(FEngine.ActiveAdapter, LPath);
    if (LPrev = '') or not DirectoryExists(LPath) then
      Say('no earlier adapter to roll back to')
    else
    begin
      FStore.SetAdapterStatus(FEngine.ActiveAdapter, 'rolledback');
      PromoteManifest(LPath);
    end;
  end
  else if LCmd = '/recall' then
  begin
    { what the RAG tier would put in the prompt for this query, with scores }
    Retrieve(LArg, High(Int64), LFacts, LMsgs);
    for I := 0 to High(LFacts) do
      Say(Format('  fact #%-3d cos %.2f  %s %.4f  %s%s', [LFacts[I].Id, LFacts[I].Cos,
        IfThen(FRet.HasReranker, 'rerank', 'rrf'), LFacts[I].Score, LFacts[I].Text,
        IfThen(LFacts[I].BakedIn = '', '', '  [baked]')]));
    for I := 0 to High(LMsgs) do
      Say(Format('  msg  #%-3d cos %.2f  %s %.4f  %s', [LMsgs[I].Id, LMsgs[I].Cos,
        IfThen(FRet.HasReranker, 'rerank', 'rrf'), LMsgs[I].Score, LMsgs[I].Content]));
    if Length(LFacts) + Length(LMsgs) = 0 then
      Say('  (nothing relevant in RAG memory)');
  end
  else if LCmd = '/status' then
  begin
    Say('engine:   ' + FEngine.Describe);
    Say('trainer:  ' + FCfg.Trainer + ' on ' + DeviceName(FCfg.TrainGpu));
    Say('recall:   ' + FRet.Describe);
    Say('guard:    ' + IfThen(FGuard = nil, 'off', 'on (' + FGuard.Describe + ', min score ' +
      FloatToStr(FCfg.GuardMinScore) + ')'));
    Say('adapter:  ' + IfThen(FEngine.ActiveAdapter = '', '(none)', FEngine.ActiveAdapter) +
      IfThen(FEngine.FastAdapter = '', '', ' + fast ' + FEngine.FastAdapter));
    if FCfg.PriorsEnabled then
      Say(Format('priors:   %d in the prompt, %d in all (/priors)', [Length(PromptPriors),
        Length(FStore.Priors(False))]));
    if FBody <> nil then
      Say(Format('body:     %s; %d rule(s) apply', [FBody.Describe, Length(BodyRules)]));
    Say(Format('facts:    %d active, %d waiting to be baked', [Length(FStore.ActiveFacts), FStore.UnbakedCount]));
    if FCfg.Tiers then
      Say(Format('tiers:    fast LoRA holds %d facts; deep sleep at %d', [
        IfThen(FEngine.ActiveAdapter = '', 0, FStore.FastTierCount(FEngine.ActiveAdapter)), FCfg.DeepAfter]));
    Say('sleeping: ' + IfThen(FCons.Running, 'yes (' + FCons.Tier + ' run)', 'no'));
    if FTurns > 0 then
      Say(Format('timing:   %d turns, avg ms: recall %.0f extract %.0f first-token %.0f generate %.0f ' +
        'tools %.0f summary %.0f total %.0f', [FTurns, FSum[0] / FTurns, FSum[1] / FTurns, FSum[2] / FTurns,
        FSum[3] / FTurns, FSum[4] / FTurns, FSum[5] / FTurns, FSum[6] / FTurns]));
    Say('data:     ' + FCfg.DataDir);
  end
  else
    Say('unknown command, try /help');
end;

end.
