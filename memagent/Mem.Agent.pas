unit Mem.Agent;

(* The agent loop. Per user turn:
     1. promote a finished consolidation (hot-swap the new LoRA)
     2. retrieve unbaked facts + older messages (FTS5) for the prompt
     3. priors + summary + memory + recent turns -> generate (streamed)
     4. optional tool calls (bigger profiles only)
     5. persist the turn, extract facts from the user message
     6. maybe start a background consolidation on the training GPU *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, DateUtils, StrUtils, Math, fpjson, jsonparser,
  Mem.Config, Mem.Store, Mem.Engine, Mem.Consolidate, Mem.Retrieval, Mem.Recall;

type
  TAgent = class
  private
    FCfg: TMemConfig;
    FStore: TMemStore;
    FEngine: TEngine;
    FCons: TConsolidator;
    FRet: TRetrieval;
    FLastRecall: TRecallTiming;
    { per-turn stage timing (MEMAGENT_TIMING=1 prints it; /status shows averages) }
    FTiming: Boolean;
    FTurnStart, FGenStart: TDateTime;
    FFirstTokenMs: Double;
    FTurns: Integer;
    FSum: array[0..6] of Double;   { recall, extract, ttft, generate, tools, summarize, total }
    FSession: string;
    FQuiet: Boolean;
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
    procedure MaybeSummarize;
    procedure MaybeAutoConsolidate;
    function PromoteManifest(const ADir: string): Boolean;
    procedure ClearReady;
    procedure Say(const S: string);
  public
    constructor Create(ACfg: TMemConfig; AStore: TMemStore; AEngine: TEngine; ACons: TConsolidator;
      ARet: TRetrieval);
    procedure Startup;
    procedure Turn(const AUser: string);
    procedure PollConsolidation;
    function Command(const ALine: string): Boolean; { False = quit }
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
  ARet: TRetrieval);
begin
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

function TAgent.BuildMessages(const AUser: string; AUseRag: Boolean): TJSONArray;
var
  LSys, LSummary, LMem: string;
  LUpto, LFirstRecent: Int64;
  LRecent: TMsgArray;
  LBudget, LUsed, I, LStart: Integer;
begin
  LSys := FCfg.Priors;
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
  I: Integer;
  LFact: TJSONObject;
  LId, LDupId: Int64;
  LOldSubject: string;
begin
  Result := '';
  LFacts := FEngine.ExtractFacts(AUser, AMode);
  try
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
        afSuperseded: Say(Format('[memory ~#%d] %s (replaces an older value)', [LId, LFact.Get('text', '')]));
        afDuplicate:
          begin
            Result := Result + IfThen(Result = '', '', '; ') + 'already known: ' + LFact.Get('text', '');
            Continue;
          end;
      end;
      Result := Result + IfThen(Result = '', '', '; ') + 'stored: ' + LFact.Get('text', '');
      EmbedFact(LId, LFact.Get('text', ''), LFact.Get('subject', ''));
    end;
  finally
    LFacts.Free;
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
  LRuleFacts: string;
begin
  PollConsolidation;
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
    Mark(LT, LMs[1]);
    Write('pal> ');
    LReply := FEngine.Generate(LMessages, True, @OnToken);
    WriteLn;
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

procedure TAgent.MaybeAutoConsolidate;
var
  LLast: Int64;
begin
  if not FCfg.AutoConsolidate or FCons.Running then
    Exit;
  LLast := StrToInt64Def(FStore.GetMeta('last_consolidation', '0'), 0);
  if NowUnix - LLast < Int64(FCfg.IntervalMin) * 60 then
    Exit;
  if (FStore.UnbakedCount < FCfg.MinNewFacts) and (FStore.GetMeta('forget_pending') <> '1') then
    Exit;
  if FCons.Start then
    Say(Format('[sleep] %d new facts: consolidating into a new LoRA in the background (%s)',
      [FStore.UnbakedCount, DeviceName(FCfg.TrainGpu)]));
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
    LId := LMan.Get('id', '');
    LOld := FEngine.ActiveAdapter;
    LT := Now;
    FEngine.LoadAdapter(ADir, LId);
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

procedure TAgent.PollConsolidation;
var
  LCode: Integer;
  LSummary: string;
  LJson: TJSONData;
begin
  if not FCons.TakeResult(LCode, LSummary) then
    Exit;
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
      2: Say(Format('[sleep] new LoRA rejected by the eval gate (recall %.2f, ppl x%.2f); keeping %s',
           [TJSONObject(LJson).Get('recall', 0.0), TJSONObject(LJson).Get('ppl_ratio', 0.0),
            IfThen(FEngine.ActiveAdapter = '', 'the base model', FEngine.ActiveAdapter)]));
      3: Say('[sleep] nothing to consolidate');
    else
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
    Say('/facts  /remember <text>  /forget <id>  /recall <query>  /ab <question>  /sleep  /wait  /adapter' +
      '  /rollback  /status  /quit')
  else if LCmd = '/facts' then
  begin
    LFacts := FStore.ActiveFacts;
    for I := 0 to High(LFacts) do
      Say(Format('  #%-3d %-45s %s', [LFacts[I].Id, LFacts[I].Text,
        IfThen(LFacts[I].BakedIn = '', '[in RAG]', '[baked: ' + LFacts[I].BakedIn + ']')]));
    if Length(LFacts) = 0 then
      Say('  (no facts yet)');
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
  else if LCmd = '/sleep' then
  begin
    if FCons.Start then
      Say(Format('[sleep] consolidating %d active facts into a new LoRA (%s)...',
        [Length(FStore.ActiveFacts), DeviceName(FCfg.TrainGpu)]))
    else
      Say('[sleep] a consolidation is already running');
  end
  else if LCmd = '/wait' then
  begin
    if FCons.Running then
      Say('[sleep] waiting for the consolidation to finish...');
    FCons.WaitFor;
    PollConsolidation;
  end
  else if LCmd = '/adapter' then
  begin
    Say('active: ' + IfThen(FEngine.ActiveAdapter = '', '(none)', FEngine.ActiveAdapter));
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
    Say('recall:   ' + FRet.Describe);
    Say('adapter:  ' + IfThen(FEngine.ActiveAdapter = '', '(none)', FEngine.ActiveAdapter));
    Say(Format('facts:    %d active, %d waiting to be baked', [Length(FStore.ActiveFacts), FStore.UnbakedCount]));
    Say('sleeping: ' + BoolToStr(FCons.Running, 'yes (training)', 'no'));
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
