unit Mem.LlamaChat;

(* Chat generation on a llama.cpp model, shared by the in-process engine
   (Mem.Engine.Llama) and the in-process trainer (Mem.Train), which uses it
   for the anchor answers and the recall probes.

   Generation mirrors trainer/memlib.generate: the model's own chat template,
   greedy decoding at temperature 0 (top-p 0.9 sampling otherwise), and an
   HF-style repetition penalty of 1.1 over prompt and reply. Each call starts
   from an empty KV cache. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, fpjson, Mem.Config, Mem.Llama;

type
  ELlama = class(Exception);
  TTextEvent = procedure(const AText: string) of object;
  TLlamaAdapters = array of PLlamaAdapter;

  TLlamaChat = class
  private
    FModel: PLlamaModel;
    FCtx: PLlamaContext;
    FVocab: PLlamaVocab;
    FTemplate: string;
    FScale: Single;
    function Piece(AToken: TLlamaToken): string;
    function NewSampler(ATemperature: Double; ANCtx: Integer): PLlamaSampler;
  public
    { a context of ANCtx tokens on AModel (not owned) }
    constructor Create(AModel: PLlamaModel; ANCtx, AThreads: Integer);
    destructor Destroy; override;
    (* AMessages: [{"role": .., "content": ..}]; AAddAssistant opens the reply turn *)
    function Prompt(AMessages: TJSONArray; AAddAssistant: Boolean = True): string;
    function Tokenize(const AText: string): TLlamaTokens;
    { AAdapter = nil: the base model alone }
    function Generate(AMessages: TJSONArray; AAdapter: PLlamaAdapter; AMaxNewTokens: Integer;
      ATemperature: Double; AOnToken: TTextEvent = nil): string;
    { the adapters apply together, each at scale 1 (deep and fast memory tiers) }
    function GenerateWith(AMessages: TJSONArray; const AAdapters: TLlamaAdapters; AMaxNewTokens: Integer;
      ATemperature: Double; AOnToken: TTextEvent = nil): string;
    property Model: PLlamaModel read FModel;
    property Context: PLlamaContext read FCtx;
  end;

{ loads the library, routes llama.cpp's log to data/logs/llama.log, inits the
  backend; safe to call more than once and from any thread }
procedure LlamaSetup(ACfg: TMemConfig);
{ AGpu < 0: CPU; else all layers (or AGpuLayers) on that device alone }
function LlamaLoadModel(const APath: string; AGpu, AGpuLayers: Integer): PLlamaModel;
{ all cores but one when AThreads <= 0 }
function LlamaThreads(AThreads: Integer): Integer;
(* a [{"role": .., "content": ..}, ...] array from role/content pairs *)
function ChatMessages(const APairs: array of string): TJSONArray;

implementation

var
  GLog: TextFile;
  GLogOpen: Boolean;
  GLock: TRTLCriticalSection;

procedure LogCallback(ALevel: Int32; AText: PAnsiChar; AUserData: Pointer); cdecl;
begin
  { llama.cpp is chatty on stderr; keep it in data/logs/llama.log like worker.log.
    The trainer thread logs too, hence the lock. }
  if not GLogOpen or (AText = nil) then
    Exit;
  EnterCriticalSection(GLock);
  try
    Write(GLog, AText);
    if (ALevel >= 3) or (StrLen(AText) > 0) and (AText[StrLen(AText) - 1] = #10) then
      Flush(GLog);
  finally
    LeaveCriticalSection(GLock);
  end;
end;

procedure LlamaSetup(ACfg: TMemConfig);
var
  LPath: string;
begin
  EnterCriticalSection(GLock);
  try
    LlamaLoad(ACfg.LlamaLibDir);
    { FPC unmasks floating-point exceptions per thread; ggml needs the C default }
    LlamaMaskFpu;
    if GLogOpen then
      Exit;
    LPath := IncludeTrailingPathDelimiter(ACfg.LogDir) + 'llama.log';
    AssignFile(GLog, LPath);
    if FileExists(LPath) then
      Append(GLog)
    else
      Rewrite(GLog);
    GLogOpen := True;
    { MEMAGENT_DEBUG=1 leaves llama.cpp's own (unbuffered) stderr logging on }
    if GetEnvironmentVariable('MEMAGENT_DEBUG') <> '1' then
      llama_log_set(@LogCallback, nil);
    LlamaTrace('backend init');
    llama_backend_init;
  finally
    LeaveCriticalSection(GLock);
  end;
end;

function LlamaLoadModel(const APath: string; AGpu, AGpuLayers: Integer): PLlamaModel;
var
  MP: TLlamaModelParams;
begin
  MP := llama_model_default_params();
  (* one device per role: pinned to main_gpu instead of being split across all
     of them, so inference and training each keep their own GPU *)
  if AGpu < 0 then
    MP.n_gpu_layers := 0
  else
  begin
    MP.n_gpu_layers := AGpuLayers;
    MP.split_mode := LLAMA_SPLIT_MODE_NONE;
    MP.main_gpu := AGpu;
  end;
  LlamaTrace('loading model ' + APath);
  Result := llama_model_load_from_file(PAnsiChar(APath), MP);
  if Result = nil then
    raise ELlama.CreateFmt('llama.cpp could not load %s (see llama.log)', [APath]);
end;

function LlamaThreads(AThreads: Integer): Integer;
begin
  (* All cores but one: ggml's workers spin while they wait, and with every
     core taken they fight the agent thread and ONNX Runtime. Measured on 4
     cores (tiny, 5 turns): 4 threads 1506 ms/turn, 3 threads 625 ms. *)
  Result := AThreads;
  if Result <= 0 then
    Result := Max(1, Integer(TThread.ProcessorCount) - 1);
end;

function ChatMessages(const APairs: array of string): TJSONArray;
var
  I: Integer;
begin
  Result := TJSONArray.Create;
  I := 0;
  while I < High(APairs) do
  begin
    Result.Add(TJSONObject.Create(['role', APairs[I], 'content', APairs[I + 1]]));
    Inc(I, 2);
  end;
end;

constructor TLlamaChat.Create(AModel: PLlamaModel; ANCtx, AThreads: Integer);
var
  CP: TLlamaContextParams;
  P: PAnsiChar;
begin
  inherited Create;
  FModel := AModel;
  FScale := 1.0;
  FVocab := llama_model_get_vocab(FModel);
  P := llama_model_chat_template(FModel, nil);
  if P = nil then
    raise ELlama.Create('the GGUF model has no chat template');
  FTemplate := P;
  CP := llama_context_default_params();
  CP.n_ctx := ANCtx;
  CP.n_batch := ANCtx;   { a whole prompt decodes in one call }
  CP.n_threads := AThreads;
  CP.n_threads_batch := AThreads;
  CP.no_perf := True;
  LlamaTrace('creating context');
  FCtx := llama_init_from_model(FModel, CP);
  if FCtx = nil then
    raise ELlama.Create('llama.cpp could not create a context');
end;

destructor TLlamaChat.Destroy;
begin
  if FCtx <> nil then
    llama_free(FCtx);
  inherited Destroy;
end;

function TLlamaChat.Prompt(AMessages: TJSONArray; AAddAssistant: Boolean): string;
var
  LMsgs: array of TLlamaChatMessage;
  LStrs: array of string;
  I, N: Integer;
begin
  SetLength(LMsgs, AMessages.Count);
  SetLength(LStrs, AMessages.Count * 2);
  for I := 0 to AMessages.Count - 1 do
  begin
    LStrs[2 * I] := AMessages.Objects[I].Get('role', 'user');
    LStrs[2 * I + 1] := AMessages.Objects[I].Get('content', '');
    LMsgs[I].role := PAnsiChar(LStrs[2 * I]);
    LMsgs[I].content := PAnsiChar(LStrs[2 * I + 1]);
  end;
  SetLength(Result, 4096);
  N := llama_chat_apply_template(PAnsiChar(FTemplate), @LMsgs[0], Length(LMsgs), AAddAssistant,
    PAnsiChar(Result), Length(Result));
  if N > Length(Result) then
  begin
    SetLength(Result, N);
    N := llama_chat_apply_template(PAnsiChar(FTemplate), @LMsgs[0], Length(LMsgs), AAddAssistant,
      PAnsiChar(Result), Length(Result));
  end;
  if N < 0 then
    raise ELlama.Create('llama.cpp does not know this model''s chat template');
  SetLength(Result, N);
end;

function TLlamaChat.Tokenize(const AText: string): TLlamaTokens;
var
  N: Integer;
begin
  (* the template already holds the special tokens, as in memlib.generate
     (add_special_tokens=False); parse_special turns them into token ids *)
  Result := nil;
  SetLength(Result, Length(AText) + 16);
  N := llama_tokenize(FVocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, True);
  if N < 0 then
  begin
    SetLength(Result, -N);
    N := llama_tokenize(FVocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, True);
  end;
  if N < 0 then
    raise ELlama.Create('tokenize failed');
  SetLength(Result, N);
end;

function TLlamaChat.Piece(AToken: TLlamaToken): string;
var
  N: Integer;
begin
  Result := '';
  SetLength(Result, 64);
  N := llama_token_to_piece(FVocab, AToken, PAnsiChar(Result), Length(Result), 0, False);
  if N < 0 then
  begin
    SetLength(Result, -N);
    N := llama_token_to_piece(FVocab, AToken, PAnsiChar(Result), Length(Result), 0, False);
  end;
  SetLength(Result, Max(N, 0));
end;

function TLlamaChat.NewSampler(ATemperature: Double; ANCtx: Integer): PLlamaSampler;
begin
  Result := llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(Result, llama_sampler_init_penalties(llama_vocab_n_tokens(FVocab), ANCtx, 1.1, 0, 0));
  if ATemperature > 0 then
  begin
    llama_sampler_chain_add(Result, llama_sampler_init_temp(ATemperature));
    llama_sampler_chain_add(Result, llama_sampler_init_top_p(0.9, 1));
    llama_sampler_chain_add(Result, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  end
  else
    llama_sampler_chain_add(Result, llama_sampler_init_greedy());
end;

{ the length of the longest prefix of S that ends on a whole UTF-8 character }
function Utf8Whole(const S: string): Integer;
var
  I, LNeed: Integer;
  B: Byte;
begin
  Result := Length(S);
  I := Length(S);
  { walk back over continuation bytes to the lead byte of the last character }
  while (I > 0) and ((Byte(S[I]) and $C0) = $80) and (Length(S) - I < 3) do
    Dec(I);
  if I = 0 then
    Exit;
  B := Byte(S[I]);
  if B < $80 then
    LNeed := 1
  else if (B and $E0) = $C0 then
    LNeed := 2
  else if (B and $F0) = $E0 then
    LNeed := 3
  else if (B and $F8) = $F0 then
    LNeed := 4
  else
    LNeed := 1;
  if Length(S) - I + 1 < LNeed then
    Result := I - 1;
end;

function TLlamaChat.Generate(AMessages: TJSONArray; AAdapter: PLlamaAdapter; AMaxNewTokens: Integer;
  ATemperature: Double; AOnToken: TTextEvent): string;
var
  LAdapters: TLlamaAdapters;
begin
  LAdapters := nil;
  if AAdapter <> nil then
  begin
    SetLength(LAdapters, 1);
    LAdapters[0] := AAdapter;
  end;
  Result := GenerateWith(AMessages, LAdapters, AMaxNewTokens, ATemperature, AOnToken);
end;

function TLlamaChat.GenerateWith(AMessages: TJSONArray; const AAdapters: TLlamaAdapters; AMaxNewTokens: Integer;
  ATemperature: Double; AOnToken: TTextEvent): string;
var
  LScales: array of Single;
  LToks: TLlamaTokens;
  LTok: TLlamaToken;
  NPrompt, NCtx, NBatch, I, LDone, LChunk, LWhole: Integer;
  LSmpl: PLlamaSampler;
  LPending: string;
begin
  LToks := Tokenize(Prompt(AMessages));
  NPrompt := Length(LToks);
  NCtx := llama_n_ctx(FCtx);
  if NPrompt >= NCtx then
    raise ELlama.CreateFmt('prompt is %d tokens, the context holds %d (ctx_tokens in models.json)',
      [NPrompt, NCtx]);
  if NPrompt + AMaxNewTokens > NCtx then
    AMaxNewTokens := NCtx - NPrompt;

  llama_memory_clear(llama_get_memory(FCtx), True);
  if Length(AAdapters) > 0 then
  begin
    SetLength(LScales, Length(AAdapters));
    for I := 0 to High(LScales) do
      LScales[I] := FScale;
    llama_set_adapters_lora(FCtx, @AAdapters[0], Length(AAdapters), @LScales[0]);
  end
  else
    llama_set_adapters_lora(FCtx, nil, 0, nil);
  LSmpl := NewSampler(ATemperature, NCtx);
  try
    { prompt tokens count for the repetition penalty, as in transformers }
    for I := 0 to NPrompt - 1 do
      llama_sampler_accept(LSmpl, LToks[I]);
    NBatch := llama_n_batch(FCtx);
    LDone := 0;
    while LDone < NPrompt do
    begin
      LChunk := Min(NBatch, NPrompt - LDone);
      if llama_decode(FCtx, llama_batch_get_one(@LToks[LDone], LChunk)) <> 0 then
        raise ELlama.Create('llama_decode failed on the prompt');
      Inc(LDone, LChunk);
    end;

    Result := '';
    LPending := '';
    for I := 1 to AMaxNewTokens do
    begin
      LTok := llama_sampler_sample(LSmpl, FCtx, -1);
      if llama_vocab_is_eog(FVocab, LTok) then
        Break;
      LPending := LPending + Piece(LTok);
      { a token can end inside a multi-byte character: hold the tail back }
      LWhole := Utf8Whole(LPending);
      if LWhole > 0 then
      begin
        if Assigned(AOnToken) then
          AOnToken(Copy(LPending, 1, LWhole));
        Result := Result + Copy(LPending, 1, LWhole);
        Delete(LPending, 1, LWhole);
      end;
      if llama_decode(FCtx, llama_batch_get_one(@LTok, 1)) <> 0 then
        raise ELlama.Create('llama_decode failed');
    end;
    if LPending <> '' then
    begin
      if Assigned(AOnToken) then
        AOnToken(LPending);
      Result := Result + LPending;
    end;
  finally
    llama_sampler_free(LSmpl);
  end;
  Result := Trim(Result);
end;

initialization
  InitCriticalSection(GLock);

finalization
  if GLogOpen then
    CloseFile(GLog);
  DoneCriticalSection(GLock);
end.
