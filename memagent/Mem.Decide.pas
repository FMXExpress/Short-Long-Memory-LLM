unit Mem.Decide;

(* Decisions as a readout of a decision model (a Jev-style judge), not as rules.

   The model reads a state (data), a question (fixed text) and lettered options, and
   the answer is the softmax of the option letters' logits at the first output
   position: one forward pass, no generated text, a calibrated probability. This is
   OpenJev's helper (helper/shim.py), done by hand as JevBan does it (Ban.Local):

     <|im_start|>user
     State:
     {"stored_facts": [...], "user_question": "...", "assistant_answer": "..."}

     Question: <the decision, in plain words>
     Options:
     [A] yes: ...
     [B] no: ...

     Answer with the letter of the best option only.<|im_end|>
     <|im_start|>assistant
     <think>\n\n</think>\n\n
     p = softmax(logit[letter] / T);  p_yes = sigmoid(logit(p[A]) / noul_t)

   Everything a user or a model wrote goes into the state as quoted JSON, never into
   the question, so a crafted message can argue for a probability but cannot change
   what is asked.

   Used by the answer guard ([guard] judge=jev). Measured on 120 answers models gave
   (bench/README.md, "The answer guard as a decision-model readout"): OpenJev-Flash-9B
   at p >= 0.9 kept 47/60 right answers and blocked 55/60 wrong ones (accuracy 0.85,
   AUC 0.93); the rule guard (Mem.Guard) kept 28 and blocked 37 (0.54). 11.5 s per
   judgement on 4 CPU cores, well under a second on a GPU. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, DateUtils, fpjson, Mem.Config, Mem.Llama, Mem.LlamaChat;

type
  TDoubleArray = array of Double;

  TJudge = class
  private
    FModel: PLlamaModel;
    FCtx: PLlamaContext;
    FVocab: PLlamaVocab;
    FLetters: array[0..25] of TLlamaToken;
    FNCtx: Integer;
    FT, FNoulT: Double;
    FThink: Boolean;
    FDesc: string;
    function Tokenize(const AText: string; ASpecial: Boolean): TLlamaTokens;
  public
    LastMs: Int64;
    constructor Create(ACfg: TMemConfig; const AModelPath: string);
    destructor Destroy; override;
    (* the probabilities of the options, in order; AOptions are (name, description) pairs *)
    function Readout(const AState, AQuestion: string; const AOptions: array of string): TDoubleArray;
    (* P(the stored facts back AAnswer to AQuestion), calibrated *)
    function Supported(const AQuestion, AAnswer: string; const AFacts: array of string): Double;
    (* P(ANew replaces AOld: the same person and the same attribute, so AOld is no longer true) *)
    function Replaces(const AOld, ANew: string): Double;
    (* P(AObservations, what a robot reported while acting, show that AClaim is now true) *)
    function Observed(const AClaim: string; const AObservations: array of string): Double;
    property Describe: string read FDesc;
  end;

{ the judge's GGUF: a path, or a known name (downloaded into <cache>/jev once); '' if unavailable }
function JudgeModelPath(ACfg: TMemConfig; AVerbose: Boolean): string;

implementation

uses
  Mem.Https;

const
  { name, repo file, T, noul_t (the model card's calibration for the family) }
  OPENJEV_FLASH_9B_URL = 'https://huggingface.co/openjev/OpenJev-Flash-9B-GGUF/resolve/main/OpenJev-Flash-9B-Q4_K_M.gguf';
  OPENJEV_FLASH_9B_FILE = 'OpenJev-Flash-9B-Q4_K_M.gguf';

  SUPPORT_QUESTION = 'The assistant answered the user''s question about the user''s own people. Decide whether ' +
    'the answer is right according to the stored facts in the state. It is right only if every name and ' +
    'value it gives about the person the question asks about is backed by those facts for that same ' +
    'person (a relation such as "my sister" is resolved through the fact that names her), and it answers ' +
    'what was asked; for a yes/no question it must say yes or no correctly.';
  SUPPORT_YES = 'the facts back the answer and it answers the question';
  SUPPORT_NO = 'the answer gives a name or value the facts do not back for that person, gives another ' +
    'person''s value, or does not answer what was asked';

  REPLACE_QUESTION = 'The user told the assistant a new fact. Decide whether it replaces the stored fact, so ' +
    'that the stored fact is no longer true and should be overwritten. It replaces it only if both are about ' +
    'the same person or thing (my sister and my brother are different people; Ana and Ana''s dog are ' +
    'different) and the same attribute (a city and a job are different attributes; a name and a nickname are ' +
    'the same). When unsure, it does not replace it: keeping both loses nothing.';
  REPLACE_YES = 'the new fact is the same person and attribute with a new or corrected value';
  REPLACE_NO = 'the new fact is about another person, another attribute, or adds to the stored fact';

  OBSERVED_QUESTION = 'A robot reported these observations while it did a task. Decide whether they show ' +
    'that the claim is true in the world now. A step the robot says it did is not proof; what it saw is.';
  OBSERVED_YES = 'the observations show the claim is true now';
  OBSERVED_NO = 'the observations do not show it, or show it is not true';

function JsonStr(const S: string): string;
begin
  Result := '"' + StringToJSONString(S, False) + '"';
end;

function JudgeModelPath(ACfg: TMemConfig; AVerbose: Boolean): string;
var
  LName: string;
begin
  LName := ACfg.JudgeModel;
  if FileExists(LName) then
    Exit(LName);
  if not SameText(LName, 'openjev-flash-9b') then
    Exit('');
  Result := IncludeTrailingPathDelimiter(ACfg.CacheDir) + 'jev' + PathDelim + OPENJEV_FLASH_9B_FILE;
  if FileExists(Result) then
    Exit;
  WriteLn('[setup] downloading the judge OpenJev-Flash-9B (one time, 5.6 GB) -> ', Result);
  try
    ForceDirectories(ExtractFileDir(Result));
    HttpsGetToFile(OPENJEV_FLASH_9B_URL, Result, AVerbose);
  except
    on E: Exception do
    begin
      WriteLn(StdErr, '[setup] judge download failed: ', E.Message);
      Result := '';
    end;
  end;
end;

constructor TJudge.Create(ACfg: TMemConfig; const AModelPath: string);
var
  CP: TLlamaContextParams;
  LTok: TLlamaTokens;
  LBuf: array[0..255] of AnsiChar;
  I: Integer;
begin
  inherited Create;
  FT := ACfg.JudgeT;
  FNoulT := ACfg.JudgeNoulT;
  { OpenJev's chat template wants the empty think block; a path to another model may not }
  FThink := ACfg.JudgeThink;
  LlamaSetup(ACfg);
  (* the answering GPU when there is one: the judge runs right after each answer,
     never at the same time, so it shares that device instead of the trainer's *)
  FModel := LlamaLoadModel(AModelPath, ACfg.InferGpu, ACfg.LlamaGpuLayers);
  FVocab := llama_model_get_vocab(FModel);
  FNCtx := 4096;
  CP := llama_context_default_params();
  CP.n_ctx := FNCtx;
  CP.n_batch := 512;
  CP.n_ubatch := 512;
  CP.n_threads := LlamaThreads(ACfg.LlamaThreads);
  CP.n_threads_batch := CP.n_threads;
  FCtx := llama_init_from_model(FModel, CP);
  if FCtx = nil then
    raise ELlama.Create('llama.cpp could not create a context for the judge');
  for I := 0 to 25 do
  begin
    LTok := Tokenize(Chr(65 + I), False);
    if Length(LTok) <> 1 then
      raise ELlama.CreateFmt('the judge''s vocabulary splits the letter %s', [Chr(65 + I)]);
    FLetters[I] := LTok[0];
  end;
  FillChar(LBuf, SizeOf(LBuf), 0);
  llama_model_desc(FModel, @LBuf[0], SizeOf(LBuf) - 1);
  FDesc := Format('%s (%s) on %s', [ExtractFileName(AModelPath), string(PAnsiChar(@LBuf[0])),
    DeviceName(ACfg.InferGpu)]);
end;

destructor TJudge.Destroy;
begin
  if FCtx <> nil then
    llama_free(FCtx);
  if FModel <> nil then
    llama_model_free(FModel);
  inherited Destroy;
end;

function TJudge.Tokenize(const AText: string; ASpecial: Boolean): TLlamaTokens;
var
  N: Integer;
begin
  Result := nil;
  SetLength(Result, Length(AText) + 16);
  N := llama_tokenize(FVocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, ASpecial);
  if N < 0 then
  begin
    SetLength(Result, -N);
    N := llama_tokenize(FVocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, ASpecial);
  end;
  if N < 0 then
    raise ELlama.Create('the judge could not tokenize its prompt');
  SetLength(Result, N);
end;

function TJudge.Readout(const AState, AQuestion: string; const AOptions: array of string): TDoubleArray;
var
  LText: string;
  LTokens: TLlamaTokens;
  I, N, LPos, LCount: Integer;
  L: PSingle;
  LMax, LSum: Double;
  LStart: TDateTime;
begin
  LStart := Now;
  LCount := Length(AOptions) div 2;
  if (LCount < 2) or (LCount > 26) then
    raise ELlama.Create('a decision needs 2 to 26 options');
  LText := 'State:' + #10 + AState + #10#10 + 'Question: ' + AQuestion + #10 + 'Options:';
  for I := 0 to LCount - 1 do
    LText := LText + #10 + '[' + Chr(65 + I) + '] ' + AOptions[2 * I] + ': ' + AOptions[2 * I + 1];
  LText := '<|im_start|>user' + #10 + LText + #10#10 + 'Answer with the letter of the best option only.' +
    '<|im_end|>' + #10 + '<|im_start|>assistant' + #10;
  if FThink then
    LText := LText + '<think>' + #10#10 + '</think>' + #10#10;
  LTokens := Tokenize(LText, True);
  if Length(LTokens) >= FNCtx then
    raise ELlama.CreateFmt('the judge''s prompt of %d tokens does not fit its %d-token context',
      [Length(LTokens), FNCtx]);
  LlamaMaskFpu;
  llama_memory_clear(llama_get_memory(FCtx), True);
  LPos := 0;
  while LPos < Length(LTokens) do
  begin
    N := Min(512, Length(LTokens) - LPos);
    if llama_decode(FCtx, llama_batch_get_one(@LTokens[LPos], N)) <> 0 then
      raise ELlama.Create('llama_decode failed in the judge');
    Inc(LPos, N);
  end;
  L := llama_get_logits_ith(FCtx, -1);
  if L = nil then
    raise ELlama.Create('the judge produced no logits');
  Result := nil;
  SetLength(Result, LCount);
  LMax := -MaxDouble;
  for I := 0 to LCount - 1 do
  begin
    Result[I] := L[FLetters[I]] / FT;
    LMax := Max(LMax, Result[I]);
  end;
  LSum := 0;
  for I := 0 to LCount - 1 do
  begin
    Result[I] := Exp(Result[I] - LMax);
    LSum := LSum + Result[I];
  end;
  for I := 0 to LCount - 1 do
    Result[I] := Result[I] / LSum;
  LastMs := MilliSecondsBetween(Now, LStart);
end;

function TJudge.Supported(const AQuestion, AAnswer: string; const AFacts: array of string): Double;
var
  LState: string;
  I: Integer;
  P: TDoubleArray;
  LYes: Double;
begin
  LState := '{"stored_facts": [';
  for I := 0 to High(AFacts) do
  begin
    if I > 0 then
      LState := LState + ', ';
    LState := LState + JsonStr(AFacts[I]);
  end;
  LState := LState + '], "user_question": ' + JsonStr(AQuestion) + ', "assistant_answer": ' + JsonStr(AAnswer) + '}';
  P := Readout(LState, SUPPORT_QUESTION, ['yes', SUPPORT_YES, 'no', SUPPORT_NO]);
  LYes := EnsureRange(P[0], 1e-4, 1 - 1e-4);
  Result := 1 / (1 + Exp(-(Ln(LYes / (1 - LYes)) / FNoulT)));
end;

function TJudge.Replaces(const AOld, ANew: string): Double;
var
  P: TDoubleArray;
  LYes: Double;
begin
  P := Readout('{"stored_fact": ' + JsonStr(AOld) + ', "new_fact": ' + JsonStr(ANew) + '}', REPLACE_QUESTION,
    ['yes', REPLACE_YES, 'no', REPLACE_NO]);
  LYes := EnsureRange(P[0], 1e-4, 1 - 1e-4);
  Result := 1 / (1 + Exp(-(Ln(LYes / (1 - LYes)) / FNoulT)));
end;

function TJudge.Observed(const AClaim: string; const AObservations: array of string): Double;
var
  LState: string;
  I: Integer;
  P: TDoubleArray;
  LYes: Double;
begin
  LState := '{"observations": [';
  for I := 0 to High(AObservations) do
  begin
    if I > 0 then
      LState := LState + ', ';
    LState := LState + JsonStr(AObservations[I]);
  end;
  LState := LState + '], "claim": ' + JsonStr(AClaim) + '}';
  P := Readout(LState, OBSERVED_QUESTION, ['yes', OBSERVED_YES, 'no', OBSERVED_NO]);
  LYes := EnsureRange(P[0], 1e-4, 1 - 1e-4);
  Result := 1 / (1 + Exp(-(Ln(LYes / (1 - LYes)) / FNoulT)));
end;

end.
