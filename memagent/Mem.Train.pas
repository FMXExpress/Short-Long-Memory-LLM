unit Mem.Train;

(* Consolidation in-process on llama.cpp: bakes the active facts into a fresh
   LoRA without Python. A port of trainer/consolidate.py, step for step:

   1. the active facts from memory.db, each with its question paraphrases;
      the last question of every fact is held out as its recall probe, the
      rest become training examples (Mem.Examples: templates or varied)
   2. anchors: the base model's own answers to generic prompts, trained on
      alongside the facts so the adapter keeps the base model's behaviour; a
      disjoint held-out set measures drift (perplexity ratio)
   3. a fresh LoRA (A random, B zero) trained with AdamW, warmup then linear
      decay, loss only on the answer tokens
   4. the eval gate: every probe answered from the weights alone (no memory
      context) and held-out perplexity against the base model
   5. data/adapters/mem-<ts>/adapter.gguf + manifest.json, then ready.json,
      which the agent hot-swaps in

   A fast run (memory tiers, Mem.Consolidate) differs in three places: its
   facts are the ones the deep adapter does not hold, plus a replay sample of
   facts it does, so the new adapter learns to leave those alone; the deep
   adapter is loaded and attached frozen, so the anchors are the current
   model's answers and only the new adapter trains; and it writes
   fast-<ts>/. Training goes one window at a time, waiting while a chat turn
   is answered (TrainingHeld) and stopping when the run is cancelled.

   The training runs on llama.cpp's optimizer through the memagent training
   API (packaging/llama-patches): the base model is frozen, only the adapter
   trains. Examples are packed into windows of 256 tokens, each one its own
   sequence (own positions, attends only to itself), so packing changes the
   speed, not what an example sees. Training uses an f32 copy of the base
   model, made once next to it: on the CPU the backward pass through f16
   weights is 6x slower. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, DateUtils, fpjson, jsonparser, Mem.Config, Mem.Consolidate;

type
  TLlamaTrainThread = class(TTrainJob)
  private
    FLog: TFileStream;
    FRng: QWord;
    FMaxSeq: Integer;
    procedure Log(const AMsg: string);
    function Rand(AN: Integer): Integer;
    procedure Finish(ACode: Integer; ASummary: TJSONObject);
    procedure Run;
  protected
    procedure Execute; override;
  end;

const
  { trainer/memlib.py ANCHOR_PROMPTS }
  ANCHOR_PROMPTS: array[0..31] of string = (
    'What is the capital of France?', 'Say hello.', 'What is 2 + 2?', 'Name a primary color.',
    'What do bees make?', 'What is water made of?', 'Give me a synonym for happy.',
    'How many days are in a week?', 'What is the opposite of hot?', 'What animal says moo?',
    'Write one sentence about the sea.', 'What is the boiling point of water in Celsius?',
    'What language is spoken in Spain?', 'What is a triangle?', 'Name a fruit.',
    'What season comes after winter?', 'How many legs does a spider have?', 'What is the sun?',
    'Recommend a good habit.', 'What is a computer?', 'What color is grass?',
    'Who wrote Romeo and Juliet?', 'What is the largest ocean?', 'What do plants need to grow?',
    'How do you say thank you in French?', 'What is ice?', 'Name a musical instrument.',
    'What is the moon?', 'What is 10 times 3?', 'What is a noun?', 'What is rain?', 'Name a planet.');

{ memlib.answer_matches: the value, normalised, occurs in the normalised answer }
function AnswerMatches(const AAnswer, AValue: string): Boolean;
{ memlib.is_abstain }
function IsAbstain(const AAnswer: string): Boolean;

implementation

uses
  StrUtils, Mem.Db, Mem.Llama, Mem.LlamaChat, Mem.Extract, Mem.Examples;

const
  WINDOW_UNIT = 256;
  REPLAY_MIN = 4;
  HELD_OUT_ANCHORS = 8;  { the last 8 anchor prompts measure drift, never trained on }        { deep facts replayed in a fast run: at least this many, else as many as new ones }     { llama.cpp pads a context to a multiple of 256 }
  WINDOW_MAX = 4096;
  MAX_SEQ = 64;          { packed examples per window (+1 for padding) }
  GGML_TYPE_F32 = 0;

type
  ECancelled = class(Exception);

  TExample = record
    Tokens: TLlamaTokens;
    PromptLen: Integer;
  end;
  TExamples = array of TExample;

  TPacked = record
    Tokens, Labels: array of Int32;
    Seqs: array of Int32;
    Windows: Int64;
  end;

function Norm(const S: string): string;
var
  I: Integer;
  C: Char;
  LSpace: Boolean;
begin
  Result := '';
  LSpace := False;
  for I := 1 to Length(S) do
  begin
    C := LowerCase(S[I]);
    if C in ['a'..'z', '0'..'9'] then
    begin
      if LSpace and (Result <> '') then
        Result := Result + ' ';
      Result := Result + C;
      LSpace := False;
    end
    else
      LSpace := True;
  end;
end;

function AnswerMatches(const AAnswer, AValue: string): Boolean;
begin
  Result := Pos(Norm(AValue), Norm(AAnswer)) > 0;
end;

function IsAbstain(const AAnswer: string): Boolean;
var
  A, P: string;
begin
  A := LowerCase(AAnswer);
  for P in ['haven''t told me', 'have not told me', 'don''t know', 'do not know', 'not sure', 'no information'] do
    if Pos(P, A) > 0 then
      Exit(True);
  Result := False;
end;

function Ppl(ALoss: Double): Double;
begin
  Result := Exp(Min(ALoss, 50));
end;

{ TLlamaTrainThread }

procedure TLlamaTrainThread.Log(const AMsg: string);
var
  L: string;
begin
  L := AMsg + LineEnding;
  if FLog <> nil then
    FLog.WriteBuffer(L[1], Length(L));
end;

function TLlamaTrainThread.Rand(AN: Integer): Integer;
begin
  { xorshift64*: Random is global and not thread-safe }
  FRng := FRng xor (FRng shr 12);
  FRng := FRng xor (FRng shl 25);
  FRng := FRng xor (FRng shr 27);
  Result := Integer(((FRng * QWord(2685821657736338717)) shr 33) mod QWord(AN));
end;

procedure TLlamaTrainThread.Finish(ACode: Integer; ASummary: TJSONObject);
begin
  { the same summary line consolidate.py prints last }
  ASummary.Add('exit', ACode);
  FExitCode := ACode;
  FSummary := ASummary.AsJSON;
  Log(FSummary);
  ASummary.Free;
end;

procedure TLlamaTrainThread.Execute;
var
  LName: string;
begin
  LName := IncludeTrailingPathDelimiter(FCfg.LogDir) + 'consolidate.log';
  if FileExists(LName) then
    FLog := TFileStream.Create(LName, fmOpenWrite or fmShareDenyNone)
  else
    FLog := TFileStream.Create(LName, fmCreate);
  FLog.Seek(0, soEnd);
  FRng := QWord(1234) * 2654435761 + 1;
  try
    try
      LlamaMaskFpu;   { per thread: ggml needs masked FP exceptions }
      Run;
    except
      on E: ECancelled do
        Finish(4, TJSONObject.Create(['reason', 'cancelled', 'trainer', 'llama', 'tier', FTier]));
      on E: Exception do
        Finish(1, TJSONObject.Create(['reason', E.ClassName + ': ' + E.Message, 'trainer', 'llama',
          'tier', FTier]));
    end;
  finally
    FreeAndNil(FLog);
  end;
end;

procedure TLlamaTrainThread.Run;
var
  LT0: TDateTime;
  LDb: TSqlDb;
  LFacts, LReplay, LAll: TFacts;
  LQAs: TQAs;
  LFast: Boolean;
  LDeep: PLlamaAdapter;
  LBase, LBoth: TLlamaAdapters;
  LScales: array[0..1] of Single;
  LMaxId: Int64;
  NNew, NReplayPassed: Integer;
  LQ: TJSONData;
  LQs: TJSONArray;
  F32, LTmp, LDevice, LPriors, LId, LDir, LAns: string;
  LModel: PLlamaModel;
  LChat: TLlamaChat;
  LAdapter: PLlamaAdapter;
  LCtx: PLlamaContext;
  CP: TLlamaContextParams;
  LTrain, LHeld: TExamples;
  LMsgs: TJSONArray;
  I, J, K, LEpoch, LThreads, NHeld: Integer;
  LPack, LHeldPack: TPacked;
  LBaseLoss, LLoss, LAfterLoss, LRecall, LPplRatio: Double;
  LPassed, LProbes, LFactIds: TJSONArray;
  LProbeQ: array of string;
  LRefused: Integer;
  LGateOk: Boolean;
  LMetrics, LManifest, LReady: TJSONObject;
  LSL: TStringList;
  LSteps: Int64;
  LWindow, LTotal, LLongest: Integer;
  LPerFact, LBatch: Integer;
  LPhases: TJSONObject;     { seconds per phase: where a run's time goes }
  LMark: TDateTime;

  procedure Phase(const AName: string);
  begin
    LPhases.Add(AName, RoundTo(SecondSpan(Now, LMark), -1));
    LMark := Now;
  end;

  function Chat(const AQuestion, AAnswer: string): TJSONArray;
  begin
    if LPriors <> '' then
      Result := ChatMessages(['system', LPriors, 'user', AQuestion])
    else
      Result := ChatMessages(['user', AQuestion]);
    if AAnswer <> '' then
      Result.Add(TJSONObject.Create(['role', 'assistant', 'content', AAnswer]));
  end;

  { memlib.tokenize_example: the full chat, loss only after the prompt }
  function Example(const AQuestion, AAnswer: string): TExample;
  var
    LM: TJSONArray;
  begin
    LM := Chat(AQuestion, AAnswer);
    try
      Result.Tokens := LChat.Tokenize(LChat.Prompt(LM, False));
      LM.Delete(LM.Count - 1);
      Result.PromptLen := Length(LChat.Tokenize(LChat.Prompt(LM, True)));
    finally
      LM.Free;
    end;
    if Length(Result.Tokens) > FCfg.TrainMaxLen then
      SetLength(Result.Tokens, FCfg.TrainMaxLen);
  end;

  { examples into windows, each its own sequence; labels are the next token
    inside the answer, -1 elsewhere }
  function Pack(const AEx: TExamples; AShuffle: Boolean): TPacked;
  var
    LOrder: array of Integer;
    A, B, T, W, N, LSeq, LFill: Integer;
    E: TExample;

    procedure NewWindow;
    var
      P: Integer;
    begin
      if Result.Windows > 0 then
        for P := LFill to LWindow - 1 do
        begin
          Result.Tokens[W + P] := 0;
          Result.Labels[W + P] := -1;
          Result.Seqs[W + P] := -1;
        end;
      W := Result.Windows * LWindow;
      Inc(Result.Windows);
      SetLength(Result.Tokens, Result.Windows * LWindow);
      SetLength(Result.Labels, Result.Windows * LWindow);
      SetLength(Result.Seqs, Result.Windows * LWindow);
      LFill := 0;
      LSeq := 0;
    end;

  begin
    Result.Windows := 0;
    Result.Tokens := nil;
    Result.Labels := nil;
    Result.Seqs := nil;
    SetLength(LOrder, Length(AEx));
    for A := 0 to High(LOrder) do
      LOrder[A] := A;
    if AShuffle then
      for A := High(LOrder) downto 1 do
      begin
        B := Rand(A + 1);
        T := LOrder[A];
        LOrder[A] := LOrder[B];
        LOrder[B] := T;
      end;
    W := 0;
    LFill := LWindow;
    LSeq := FMaxSeq;
    for A := 0 to High(LOrder) do
    begin
      E := AEx[LOrder[A]];
      N := Length(E.Tokens);
      if (Result.Windows = 0) or (LFill + N > LWindow) or (LSeq >= FMaxSeq - 1) then
        NewWindow;
      for B := 0 to N - 1 do
      begin
        Result.Tokens[W + LFill + B] := E.Tokens[B];
        if (B + 1 < N) and (B + 1 >= E.PromptLen) then
          Result.Labels[W + LFill + B] := E.Tokens[B + 1]
        else
          Result.Labels[W + LFill + B] := -1;
        Result.Seqs[W + LFill + B] := LSeq;
      end;
      Inc(LFill, N);
      Inc(LSeq);
    end;
    if Result.Windows > 0 then
      for B := LFill to LWindow - 1 do
      begin
        Result.Tokens[W + B] := 0;
        Result.Labels[W + B] := -1;
        Result.Seqs[W + B] := -1;
      end;
  end;

  { one window at a time: a chat turn holds training, a cancel stops it }
  function Epoch(const P: TPacked; ATrain: Boolean): Double;
  var
    W, T, N: Int64;
    LSum, LWin: Double;
    LTokens: Int64;
  begin
    LSum := 0;
    LTokens := 0;
    for W := 0 to P.Windows - 1 do
    begin
      while TrainingHeld and not Terminated do
        Sleep(50);
      if Terminated then
        raise ECancelled.Create('cancelled');
      if llama_lora_train_epoch(LCtx, @P.Tokens[W * LWindow], @P.Labels[W * LWindow], @P.Seqs[W * LWindow],
        1, ATrain, @LWin) <> 0 then
        raise ELlama.Create('llama_lora_train_epoch failed (see llama.log)');
      N := 0;
      for T := W * LWindow to (W + 1) * LWindow - 1 do
        if P.Labels[T] >= 0 then
          Inc(N);
      LSum := LSum + LWin * N;
      Inc(LTokens, N);
    end;
    if LTokens > 0 then
      Result := LSum / LTokens
    else
      Result := 0;
  end;

  { the active facts the SQL selects, with their question paraphrases }
  function LoadFacts(LDb: TSqlDb; const AWhere: string; const AParams: array of const): TFacts;
  var
    LStmt: TSqlStmt;
    I, J, K: Integer;
  begin
    Result := nil;
    LStmt := LDb.Query('SELECT id, text, subject, value, questions FROM facts ' +
      'WHERE superseded = 0 AND forgotten = 0' + AWhere + ' ORDER BY id', AParams);
    try
      while LStmt.Step do
      begin
        SetLength(Result, Length(Result) + 1);
        with Result[High(Result)] do
        begin
          Id := LStmt.Int(0);
          Text := LStmt.Str(1);
          Subject := LStmt.Str(2);
          Value := LStmt.Str(3);
          Questions := nil;
          LQ := nil;
          try
            LQ := GetJSON(LStmt.Str(4));
          except
            on Exception do LQ := nil;
          end;
          if not (LQ is TJSONArray) or (TJSONArray(LQ).Count = 0) then
          begin
            LQ.Free;
            LQ := SubjectQuestions(Subject);
          end;
          LQs := TJSONArray(LQ);
          { dict.fromkeys: unique, first occurrence order }
          for I := 0 to LQs.Count - 1 do
          begin
            K := -1;
            for J := 0 to High(Questions) do
              if Questions[J] = LQs.Strings[I] then
                K := J;
            if K < 0 then
            begin
              SetLength(Questions, Length(Questions) + 1);
              Questions[High(Questions)] := LQs.Strings[I];
            end;
          end;
          LQ.Free;
        end;
      end;
    finally
      LStmt.Free;
    end;
  end;

begin
  LT0 := Now;
  LMark := LT0;
  LPhases := TJSONObject.Create;
  LFacts := nil;
  FMaxSeq := MAX_SEQ;
  LPriors := FCfg.Priors;
  LDevice := DeviceName(FCfg.TrainGpu);

  { 1. the active facts: all of them, or for a fast run the ones the deep
    adapter does not hold plus a replay sample of those it does }
  LFast := FTier = 'fast';
  LReplay := nil;
  LDb := TSqlDb.Create(FCfg.DbPath);
  try
    if LFast then
    begin
      LFacts := LoadFacts(LDb, ' AND baked_in <> ?', [FDeepId]);
      LReplay := LoadFacts(LDb, ' AND baked_in = ?', [FDeepId]);
    end
    else
      LFacts := LoadFacts(LDb, '', []);
  finally
    LDb.Free;
  end;
  LMaxId := 0;
  for I := 0 to High(LFacts) do
    LMaxId := Max(LMaxId, LFacts[I].Id);
  { the replay sample: Fisher-Yates, then the first n }
  for I := High(LReplay) downto 1 do
  begin
    J := Rand(I + 1);
    LAll := Copy(LReplay, I, 1);
    LReplay[I] := LReplay[J];
    LReplay[J] := LAll[0];
  end;
  SetLength(LReplay, Min(Length(LReplay), Max(REPLAY_MIN, Length(LFacts))));
  if Length(LFacts) = 0 then
  begin
    LPhases.Free;
    Finish(3, TJSONObject.Create(['reason', 'no active facts', 'trainer', 'llama', 'tier', FTier]));
    Exit;
  end;
  Log(Format('[consolidate] trainer=llama tier=%s profile=%s base=%s facts=%d replay=%d device=%s',
    [FTier, FCfg.ProfileName, ExtractFileName(FCfg.GgufModel), Length(LFacts), Length(LReplay), LDevice]));
  { from here on the replayed facts train and are probed like the new ones }
  NNew := Length(LFacts);
  LFacts := Concat(LFacts, LReplay);

  { the f32 training copy of the base model, made once }
  LlamaSetup(FCfg);
  if not LlamaCanTrain then
    raise ELlama.Create('this llama.cpp has no LoRA training (build it with scripts/llama_setup.sh)');
  LThreads := LlamaThreads(FCfg.TrainThreads);
  F32 := ChangeFileExt(FCfg.GgufModel, '') + '-f32.gguf';
  if not FileExists(F32) or (FileAge(F32) < FileAge(FCfg.GgufModel)) then
  begin
    Log('[consolidate] making the f32 training copy ' + F32);
    LTmp := F32 + '.tmp';
    if llama_model_convert_f32(PAnsiChar(FCfg.GgufModel), PAnsiChar(LTmp), LThreads) <> 0 then
      raise ELlama.Create('could not convert ' + FCfg.GgufModel + ' to f32');
    if FileExists(F32) then
      DeleteFile(F32);
    if not RenameFile(LTmp, F32) then
      raise ELlama.Create('could not move ' + LTmp + ' into place');
  end;

  Phase('facts');
  LModel := LlamaLoadModel(F32, FCfg.TrainGpu, -1);
  LAdapter := nil;
  LDeep := nil;
  LCtx := nil;
  LChat := nil;
  try
    LChat := TLlamaChat.Create(LModel, 1024, LThreads);
    LBase := nil;
    if LFast then
    begin
      LTmp := IncludeTrailingPathDelimiter(FDeepPath) + 'adapter.gguf';
      LDeep := llama_adapter_lora_init(LModel, PAnsiChar(LTmp));
      if LDeep = nil then
        raise ELlama.Create('could not load the deep adapter ' + LTmp);
      SetLength(LBase, 1);
      LBase[0] := LDeep;
    end;

    Phase('load_model');
    { 2. anchors: the base model's own answers }
    (* the held-out drift prompts are always the last HELD_OUT_ANCHORS, so drift
       stays comparable whatever the number of training anchors (memory loop
       4b: taking them right after the training anchors measured 16 anchors on
       other, more prompts than 8); training anchors come from the rest *)
    NHeld := HELD_OUT_ANCHORS;
    LTrain := nil;
    LHeld := nil;
    for I := 0 to Length(ANCHOR_PROMPTS) - 1 do
    begin
      if (I >= Min(FCfg.TrainAnchors, Length(ANCHOR_PROMPTS) - NHeld)) and (I < Length(ANCHOR_PROMPTS) - NHeld) then
        Continue;
      LMsgs := Chat(ANCHOR_PROMPTS[I], '');
      try
        LAns := LChat.GenerateWith(LMsgs, LBase, 48, 0.0);
      finally
        LMsgs.Free;
      end;
      if I < Length(ANCHOR_PROMPTS) - NHeld then
      begin
        SetLength(LTrain, Length(LTrain) + 1);
        LTrain[High(LTrain)] := Example(ANCHOR_PROMPTS[I], LAns);
      end
      else
      begin
        SetLength(LHeld, Length(LHeld) + 1);
        LHeld[High(LHeld)] := Example(ANCHOR_PROMPTS[I], LAns);
      end;
    end;

    Phase('anchors');
    { the fact examples; the last question of each fact is its probe }
    SetLength(LProbeQ, Length(LFacts));
    for I := 0 to High(LFacts) do
      with LFacts[I] do
      begin
        LProbeQ[I] := Questions[High(Questions)];
        { random.shuffle(pool) then pool[:n_paraphrases] }
        for J := High(Questions) - 1 downto 1 do
        begin
          K := Rand(J + 1);
          LTmp := Questions[J];
          Questions[J] := Questions[K];
          Questions[K] := LTmp;
        end;
      end;
    for I := 0 to High(LFacts) do
    begin
      (* a replayed fact is already in the frozen deep LoRA: it is there so the
         new adapter learns to leave it alone, which a short reminder does.
         Training costs ~0.64 s per example pass on tiny, all in the windows
         (memory loop 3), so full sets for replayed facts doubled a fast run *)
      if I < NNew then
        LPerFact := FCfg.TrainParaphrases
      else
        LPerFact := Max(2, FCfg.TrainParaphrases div 3);
      if FCfg.TrainExamples = 'varied' then
        LQAs := VariedExamples(LFacts, I, LPerFact, @Rand)
      else
        LQAs := TemplateExamples(LFacts[I], LPerFact);
      for J := 0 to High(LQAs) do
      begin
        SetLength(LTrain, Length(LTrain) + 1);
        LTrain[High(LTrain)] := Example(LQAs[J].Q, LQAs[J].A);
      end;
    end;
    Log(Format('[consolidate] examples=%d (facts %d, anchors %d), held-out anchors %d',
      [Length(LTrain), Length(LTrain) - FCfg.TrainAnchors, FCfg.TrainAnchors, Length(LHeld)]));

    (* One optimizer step per window, so a window holds about one profile batch
       of examples: the same number of steps per epoch as consolidate.py, which
       the learning rate is tuned for (with ~3 examples per 256-token window the
       tiny profile took twice the steps and drifted 5x instead of 1.2x) *)
    LTotal := 0;
    LLongest := 0;
    for I := 0 to High(LTrain) do
    begin
      Inc(LTotal, Length(LTrain[I].Tokens));
      LLongest := Max(LLongest, Length(LTrain[I].Tokens));
    end;
    (* max_windows: the optimizer steps per epoch stop growing with the facts;
       more examples share each step instead (memory loop 4: drift tracks the
       number of steps, and steps grew with every fact) *)
    LBatch := FCfg.TrainBatch;
    if FCfg.TrainMaxWindows > 0 then
      LBatch := Max(LBatch, (Length(LTrain) + FCfg.TrainMaxWindows - 1) div FCfg.TrainMaxWindows);
    LWindow := Max(LLongest, Round(LBatch * LTotal / Max(1, Length(LTrain))));
    LWindow := Min(WINDOW_MAX, Max(WINDOW_UNIT, ((LWindow + WINDOW_UNIT - 1) div WINDOW_UNIT) * WINDOW_UNIT));

    Phase('examples');
    { 3. a fresh LoRA, trained }
    LAdapter := llama_adapter_lora_init_new(LModel, FCfg.LoraRank, FCfg.LoraAlpha,
      PAnsiChar(FCfg.LoraTargets), 1234);
    if LAdapter = nil then
      raise ELlama.Create('could not create the LoRA (targets ' + FCfg.LoraTargets + ')');
    CP := llama_context_default_params();
    CP.n_ctx := LWindow;
    CP.n_batch := LWindow;
    CP.n_ubatch := LWindow;
    CP.n_seq_max := MAX_SEQ;
    CP.kv_unified := True;
    CP.type_k := GGML_TYPE_F32;
    CP.type_v := GGML_TYPE_F32;
    CP.n_threads := LThreads;
    CP.n_threads_batch := LThreads;
    CP.no_perf := True;
    LCtx := llama_init_from_model(LModel, CP);
    if LCtx = nil then
      raise ELlama.Create('could not create the training context');
    LBoth := Concat(LBase, [LAdapter]);
    LScales[0] := 1.0;
    LScales[1] := 1.0;
    llama_set_adapters_lora(LCtx, @LBoth[0], Length(LBoth), @LScales[0]);

    LPack := Pack(LTrain, True);
    LHeldPack := Pack(LHeld, False);
    LSteps := Int64(FCfg.TrainEpochs) * LPack.Windows;
    if LFast then
      LGateOk := llama_lora_train_init_adapter(LCtx, LModel, LAdapter, FCfg.TrainLr, 0.0, LSteps,
        Max(1, LSteps div 10)) = 0
    else
      LGateOk := llama_lora_train_init(LCtx, LModel, FCfg.TrainLr, 0.0, LSteps, Max(1, LSteps div 10)) = 0;
    if not LGateOk then
      raise ELlama.Create('llama_lora_train_init failed (see llama.log)');
    { B = 0: the fresh adapter changes nothing, so this is the model before
      the run (the base, or base + deep for a fast run) }
    Phase('setup');
    LBaseLoss := Epoch(LHeldPack, False);
    Log(Format('[consolidate] %d windows of %d tokens per epoch, base held-out loss %.4f',
      [LPack.Windows, LWindow, LBaseLoss]));
    LLoss := 0;
    Phase('eval_before');
    for LEpoch := 1 to FCfg.TrainEpochs do
    begin
      if LEpoch > 1 then
        LPack := Pack(LTrain, True);
      LLoss := Epoch(LPack, True);
      Log(Format('[consolidate] epoch %d/%d loss=%.4f', [LEpoch, FCfg.TrainEpochs, LLoss]));
    end;
    Phase('train');
    LAfterLoss := Epoch(LHeldPack, False);
    LPplRatio := Ppl(LAfterLoss) / Ppl(LBaseLoss);
    llama_free(LCtx);
    LCtx := nil;

    Phase('eval_after');
    { 4. the gate: each fact from the weights alone }
    LPassed := TJSONArray.Create;
    LProbes := TJSONArray.Create;
    LFactIds := TJSONArray.Create;
    LRefused := 0;
    NReplayPassed := 0;
    for I := 0 to High(LFacts) do
    begin
      while TrainingHeld and not Terminated do
        Sleep(50);
      if Terminated then
        raise ECancelled.Create('cancelled');
      LMsgs := Chat(LProbeQ[I], '');
      try
        LAns := LChat.GenerateWith(LMsgs, LBoth, 48, 0.0);
      finally
        LMsgs.Free;
      end;
      LGateOk := AnswerMatches(LAns, LFacts[I].Value);
      { a replayed fact stays the deep adapter's: it counts for the gate only }
      if LGateOk and (I < NNew) then
        LPassed.Add(LFacts[I].Id)
      else if LGateOk then
        Inc(NReplayPassed);
      if IsAbstain(LAns) then
        Inc(LRefused);
      if I < NNew then
        LFactIds.Add(LFacts[I].Id);
      LProbes.Add(TJSONObject.Create(['id', LFacts[I].Id, 'question', LProbeQ[I], 'answer', LAns, 'ok', LGateOk]));
      Log(Format('[probe] %s #%d %s -> %s', [StrUtils.IfThen(LGateOk, 'PASS', 'FAIL'), LFacts[I].Id, LProbeQ[I], LAns]));
    end;
    Phase('probes');
    LRecall := (LPassed.Count + NReplayPassed) / Length(LFacts);
    Log(Format('[consolidate] recall=%.2f heldout_ppl_ratio=%.3f', [LRecall, LPplRatio]));

    LMetrics := TJSONObject.Create;
    LMetrics.Add('recall', RoundTo(LRecall, -4));
    LMetrics.Add('ppl_ratio', RoundTo(LPplRatio, -4));
    LMetrics.Add('refused_known', RoundTo(LRefused / Length(LFacts), -4));
    LMetrics.Add('n_facts', NNew);
    LMetrics.Add('n_passed', LPassed.Count);
    LMetrics.Add('tier', FTier);
    if LFast then
    begin
      LMetrics.Add('deep', FDeepId);
      LMetrics.Add('replay', Length(LFacts) - NNew);
      LMetrics.Add('replay_passed', NReplayPassed);
    end;
    LMetrics.Add('max_fact_id', LMaxId);
    LMetrics.Add('train_loss', RoundTo(LLoss, -4));
    LMetrics.Add('seconds', RoundTo(SecondSpan(Now, LT0), -1));
    LMetrics.Add('device', LDevice);
    LMetrics.Add('trainer', 'llama');
    LMetrics.Add('phases', LPhases);
    LPhases := nil;
    Log('[consolidate] phases (s): ' + TJSONObject(LMetrics.Find('phases')).AsJSON);
    LGateOk := (LRecall >= FCfg.GateMinRecall) and (LPplRatio <= FCfg.GateMaxPplRatio);
    LMetrics.Add('gate_ok', LGateOk);
    if not LGateOk then
    begin
      LPassed.Free;
      LProbes.Free;
      LFactIds.Free;
      LMetrics.Add('reason', 'eval gate');
      Finish(2, LMetrics);
      Exit;
    end;

    { 5. the adapter, its manifest, then ready.json for the hot swap }
    LId := StrUtils.IfThen(LFast, 'fast-', 'mem-') + FormatDateTime('yyyymmdd"T"hhnnss', Now);
    LDir := IncludeTrailingPathDelimiter(FCfg.AdapterDir) + LId;
    ForceDirectories(LDir);
    if llama_adapter_lora_save(LAdapter, PAnsiChar(IncludeTrailingPathDelimiter(LDir) + 'adapter.gguf')) <> 0 then
      raise ELlama.Create('could not write the adapter to ' + LDir);
    LManifest := TJSONObject(LMetrics.Clone);
    LManifest.Add('id', LId);
    LManifest.Add('path', LDir);
    LManifest.Add('profile', FCfg.ProfileName);
    LManifest.Add('base', FCfg.BaseModel);
    LManifest.Add('created', FormatDateTime('yyyy-mm-dd"T"hh:nn:ss', Now));
    LManifest.Add('fact_ids', LFactIds);
    LManifest.Add('passed_fact_ids', LPassed.Clone);
    LManifest.Add('probes', LProbes);
    LSL := TStringList.Create;
    try
      LSL.Text := LManifest.FormatJSON;
      LSL.SaveToFile(IncludeTrailingPathDelimiter(LDir) + 'manifest.json');
      LReady := TJSONObject.Create(['id', LId, 'path', LDir]);
      try
        LSL.Text := LReady.AsJSON;
      finally
        LReady.Free;
      end;
      LTmp := IncludeTrailingPathDelimiter(FCfg.AdapterDir) + 'ready.json';
      LSL.SaveToFile(LTmp + '.tmp');
      if FileExists(LTmp) then
        DeleteFile(LTmp);
      RenameFile(LTmp + '.tmp', LTmp);
    finally
      LSL.Free;
      LManifest.Free;
    end;
    LMetrics.Add('id', LId);
    LMetrics.Add('path', LDir);
    LMetrics.Add('passed_fact_ids', LPassed);
    Finish(0, LMetrics);
  finally
    LPhases.Free;
    if LCtx <> nil then
      llama_free(LCtx);
    LChat.Free;
    if LAdapter <> nil then
      llama_adapter_lora_free(LAdapter);
    if LDeep <> nil then
      llama_adapter_lora_free(LDeep);
    llama_model_free(LModel);
  end;
end;

end.
