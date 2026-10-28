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
  Mem.Config, Mem.Store, Mem.Engine, Mem.Consolidate, Mem.Retrieval;

type
  TAgent = class
  private
    FCfg: TMemConfig;
    FStore: TMemStore;
    FEngine: TEngine;
    FCons: TConsolidator;
    FRet: TRetrieval;
    FSession: string;
    FQuiet: Boolean;
    procedure OnToken(const AText: string);
    function BuildMessages(const AUser: string; AUseRag: Boolean): TJSONArray;
    function MinCos: Single;
    procedure Retrieve(const AQuery: string; ABeforeId: Int64; out AFacts: TFactArray; out AMsgs: TMsgArray);
    function MemoryBlock(const AUser: string; ABeforeId: Int64): string;
    procedure EmbedFact(AId: Int64; const AText, ASubject: string);
    procedure EmbedMessage(AId: Int64; const AText: string);
    procedure Backfill;
    function SameSubject(const ASubject: string): Int64;
    function RunTools(const AReply: string; AMessages: TJSONArray): Boolean;
    procedure ExtractFacts(const AUser: string; AMsgId: Int64);
    procedure MaybeSummarize;
    procedure MaybeAutoConsolidate;
    function PromoteManifest(const ADir: string): Boolean;
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

procedure SortFacts(var A: TFactArray);
var
  I, J: Integer;
  T: TFactRec;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J].Score < T.Score) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
end;

procedure SortMsgs(var A: TMsgArray);
var
  I, J: Integer;
  T: TMsgRec;
begin
  for I := 1 to High(A) do
  begin
    T := A[I];
    J := I - 1;
    while (J >= 0) and (A[J].Score < T.Score) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := T;
  end;
end;

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
end;

procedure TAgent.Say(const S: string);
begin
  WriteLn(S);
  Flush(Output);
end;

procedure TAgent.OnToken(const AText: string);
begin
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
     ready.json; promote it now so its training is not wasted. *)
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
    if (LReadyId <> '') and (LReadyId > FEngine.ActiveAdapter) and DirectoryExists(LReadyPath) then
      PromoteManifest(LReadyPath);
  end;
end;

function TAgent.MinCos: Single;
begin
  if FRet.Tier = 'onnx' then
    Result := FCfg.MinSimilarity
  else
    Result := FCfg.StaticMinSimilarity;
end;

procedure TAgent.Retrieve(const AQuery: string; ABeforeId: Int64; out AFacts: TFactArray;
  out AMsgs: TMsgArray);
var
  LVec: TVector;
  LDocs: array of string;
  LScores: TScoreArray;
  I, N: Integer;
begin
  (* Stage 1: FTS5 + vector, fused by reciprocal rank. Stage 2: the
     cross-encoder re-scores each (query, candidate) pair together, which
     both reorders and drops candidates that merely share words. *)
  LVec := FRet.Embed(AQuery);
  AFacts := FStore.HybridFacts(AQuery, LVec, FRet.EmbedderId, FCfg.RetrievalPool, MinCos, False);
  AMsgs := FStore.HybridMessages(AQuery, LVec, FRet.EmbedderId, ABeforeId, FCfg.RetrievalPool, MinCos);
  if not FRet.HasReranker then
    Exit;
  SetLength(LDocs, Length(AFacts));
  for I := 0 to High(AFacts) do
    LDocs[I] := AFacts[I].Text;
  LScores := FRet.RerankScores(AQuery, LDocs);
  N := 0;
  for I := 0 to High(AFacts) do
    if LScores[I] >= FCfg.RerankMin then
    begin
      AFacts[N] := AFacts[I];
      AFacts[N].Score := LScores[I];
      Inc(N);
    end;
  SetLength(AFacts, N);
  SortFacts(AFacts);
  SetLength(LDocs, Length(AMsgs));
  for I := 0 to High(AMsgs) do
    LDocs[I] := AMsgs[I].Content;
  LScores := FRet.RerankScores(AQuery, LDocs);
  N := 0;
  for I := 0 to High(AMsgs) do
    if LScores[I] >= FCfg.RerankMin then
    begin
      AMsgs[N] := AMsgs[I];
      AMsgs[N].Score := LScores[I];
      Inc(N);
    end;
  SetLength(AMsgs, N);
  SortMsgs(AMsgs);
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
    FStore.SetFactVectors(AId, FRet.Embed(AText), FRet.Embed(ASubject), FRet.EmbedderId);
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

function TAgent.RunTools(const AReply: string; AMessages: TJSONArray): Boolean;
var
  P1, P2, I: Integer;
  LCall: TJSONData;
  LName, LOut: string;
  LArgs: TJSONObject;
  LFacts: TFactArray;
begin
  Result := False;
  P1 := Pos('<tool>', AReply);
  P2 := Pos('</tool>', AReply);
  if (P1 = 0) or (P2 < P1) then
    Exit;
  try
    LCall := GetJSON(Copy(AReply, P1 + 6, P2 - P1 - 6));
  except
    on E: Exception do
    begin
      LOut := 'invalid tool call: ' + E.Message;
      LCall := nil;
    end;
  end;
  if LCall <> nil then
    try
      LName := TJSONObject(LCall).Get('name', '');
      LArgs := TJSONObject(LCall).Find('args') as TJSONObject;
      if LName = 'remember' then
      begin
        ExtractFacts(LArgs.Get('fact', ''), 0);
        LOut := 'stored';
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

procedure TAgent.ExtractFacts(const AUser: string; AMsgId: Int64);
var
  LFacts: TJSONArray;
  I: Integer;
  LFact: TJSONObject;
  LId: Int64;
begin
  LFacts := FEngine.ExtractFacts(AUser);
  try
    for I := 0 to LFacts.Count - 1 do
    begin
      LFact := LFacts.Objects[I];
      if LFact.Get('confidence', 0.0) < FCfg.MinFactConfidence then
        Continue;
      case FStore.AddFact(LFact, AMsgId, SameSubject(LFact.Get('subject', '')), LId) of
        afAdded: Say(Format('[memory +#%d] %s', [LId, LFact.Get('text', '')]));
        afSuperseded: Say(Format('[memory ~#%d] %s (replaces an older value)', [LId, LFact.Get('text', '')]));
        afDuplicate: Continue;
      end;
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
    LReq.Add(Msg('user', 'Update this running summary of a conversation. Keep names, facts and open ' +
      'tasks; at most 120 words.' + LineEnding + LineEnding + 'Summary so far: ' + LOld + LineEnding +
      LineEnding + 'New messages:' + LineEnding + LText));
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
begin
  PollConsolidation;
  LMessages := BuildMessages(AUser, True);
  try
    Write('pal> ');
    LReply := FEngine.Generate(LMessages, True, @OnToken);
    WriteLn;
    I := 0;
    while FCfg.Tools and (I < 3) and RunTools(LReply, LMessages) do
    begin
      Write('pal> ');
      LReply := FEngine.Generate(LMessages, True, @OnToken);
      WriteLn;
      Inc(I);
    end;
  finally
    LMessages.Free;
  end;
  LUserId := FStore.AddMessage(FSession, 'user', AUser);
  EmbedMessage(LUserId, AUser);
  FStore.AddMessage(FSession, 'assistant', LReply);
  ExtractFacts(AUser, LUserId);
  MaybeSummarize;
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
      0: PromoteManifest(TJSONObject(LJson).Get('path', ''));
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
      Say(Format('  fact #%-3d cos %.2f  %s %.4f  %s', [LFacts[I].Id, LFacts[I].Cos,
        IfThen(FRet.HasReranker, 'rerank', 'rrf'), LFacts[I].Score, LFacts[I].Text]));
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
    Say('data:     ' + FCfg.DataDir);
  end
  else
    Say('unknown command, try /help');
end;

end.
