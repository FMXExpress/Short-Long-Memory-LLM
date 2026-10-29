unit Mem.Engine.Llama;

(* TEngine on llama.cpp, in-process: the GGUF base model is loaded into the
   agent itself and LoRA memories are hot-swapped with
   llama_set_adapters_lora, so a turn needs no Python at all. Python is only
   used out of process, by the trainer (consolidate.py), which writes
   adapter.gguf next to the PEFT adapter.

   Generation mirrors trainer/memlib.generate: the model's own chat template,
   greedy decoding at temperature 0 (top-p 0.9 sampling otherwise), and an
   HF-style repetition penalty of 1.1 over prompt and reply. Each call starts
   from an empty KV cache: the agent rebuilds the prompt every turn. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, Process, fpjson, Mem.Config, Mem.Engine, Mem.Llama, Mem.Extract;

type
  TLlamaEngine = class(TEngine)
  private
    FCfg: TMemConfig;
    FModel: PLlamaModel;
    FCtx: PLlamaContext;
    FVocab: PLlamaVocab;
    FTemplate: string;
    FAdapter: PLlamaAdapter;
    FActive, FActivePath: string;
    FDevice: string;
    FTemperature: Double;
    FScale: Single;
    function Prompt(AMessages: TJSONArray): string;
    function Tokenize(const AText: string): TLlamaTokens;
    function Piece(AToken: TLlamaToken): string;
    function NewSampler(ATemperature: Double; ANCtx: Integer): PLlamaSampler;
    procedure ApplyAdapter(AUse: Boolean);
    function Run(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
      AMaxNewTokens: Integer; ATemperature: Double): string;
    procedure ExportAdapter(const ADir, AGguf: string);
  public
    constructor Create(ACfg: TMemConfig);
    destructor Destroy; override;
    function Generate(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
      AMaxNewTokens: Integer = 0): string; override;
    function ExtractFacts(const AText, AMode: string): TJSONArray; override;
    procedure LoadAdapter(const APath, AName: string); override;
    procedure ClearAdapter; override;
    function ActiveAdapter: string; override;
    function Describe: string; override;
  end;

{ <adapter dir>/adapter.gguf: the file consolidate.py writes for this engine }
function AdapterGguf(const ADir: string): string;

implementation

var
  GLog: TextFile;
  GLogOpen: Boolean;

procedure LogCallback(ALevel: Int32; AText: PAnsiChar; AUserData: Pointer); cdecl;
begin
  { llama.cpp is chatty on stderr; keep it in data/logs/llama.log like worker.log }
  if GLogOpen and (AText <> nil) then
  begin
    Write(GLog, AText);
    if (ALevel >= 3) or (StrLen(AText) > 0) and (AText[StrLen(AText) - 1] = #10) then
      Flush(GLog);
  end;
end;

function AdapterGguf(const ADir: string): string;
begin
  Result := IncludeTrailingPathDelimiter(ADir) + 'adapter.gguf';
end;

constructor TLlamaEngine.Create(ACfg: TMemConfig);
var
  MP: TLlamaModelParams;
  CP: TLlamaContextParams;
  P: PAnsiChar;
  LThreads: Integer;
begin
  inherited Create;
  FCfg := ACfg;
  FScale := 1.0;
  FTemperature := StrToFloatDef(GetEnvironmentVariable('MEMAGENT_TEMPERATURE'), FCfg.Temperature);
  if not FileExists(FCfg.GgufModel) then
    raise EEngine.CreateFmt('no GGUF model at %s; run scripts/llama_setup.sh %s (or set [llama] model)',
      [FCfg.GgufModel, FCfg.ProfileName]);
  LlamaLoad(FCfg.LlamaLibDir);
  if not GLogOpen then
  begin
    AssignFile(GLog, IncludeTrailingPathDelimiter(FCfg.LogDir) + 'llama.log');
    if FileExists(IncludeTrailingPathDelimiter(FCfg.LogDir) + 'llama.log') then
      Append(GLog)
    else
      Rewrite(GLog);
    GLogOpen := True;
  end;
  llama_log_set(@LogCallback, nil);
  llama_backend_init;

  MP := llama_model_default_params();
  (* [gpu] infer picks the device; the training GPU stays free because the
     model is pinned to main_gpu instead of being split across all of them *)
  if FCfg.InferGpu < 0 then
    MP.n_gpu_layers := 0
  else
  begin
    MP.n_gpu_layers := FCfg.LlamaGpuLayers;
    MP.split_mode := LLAMA_SPLIT_MODE_NONE;
    MP.main_gpu := FCfg.InferGpu;
  end;
  FModel := llama_model_load_from_file(PAnsiChar(FCfg.GgufModel), MP);
  if FModel = nil then
    raise EEngine.CreateFmt('llama.cpp could not load %s (see %sllama.log)',
      [FCfg.GgufModel, IncludeTrailingPathDelimiter(FCfg.LogDir)]);
  FVocab := llama_model_get_vocab(FModel);
  P := llama_model_chat_template(FModel, nil);
  if P = nil then
    raise EEngine.Create('the GGUF model has no chat template');
  FTemplate := P;

  LThreads := FCfg.LlamaThreads;
  (* All cores but one: ggml's workers spin while they wait, and with every
     core taken they fight the agent thread and ONNX Runtime. Measured on 4
     cores (tiny, 5 turns): 4 threads 1506 ms/turn, 3 threads 625 ms. *)
  if LThreads <= 0 then
    LThreads := Max(1, Integer(TThread.ProcessorCount) - 1);
  CP := llama_context_default_params();
  CP.n_ctx := FCfg.CtxTokens;
  CP.n_batch := FCfg.CtxTokens;   { a whole prompt decodes in one call }
  CP.n_threads := LThreads;
  CP.n_threads_batch := LThreads;
  CP.no_perf := True;
  FCtx := llama_init_from_model(FModel, CP);
  if FCtx = nil then
    raise EEngine.Create('llama.cpp could not create a context');
  FDevice := DeviceName(FCfg.InferGpu);
end;

destructor TLlamaEngine.Destroy;
begin
  if FCtx <> nil then
    llama_free(FCtx);
  if FAdapter <> nil then
    llama_adapter_lora_free(FAdapter);
  if FModel <> nil then
    llama_model_free(FModel);
  inherited Destroy;
end;

function TLlamaEngine.Prompt(AMessages: TJSONArray): string;
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
  N := llama_chat_apply_template(PAnsiChar(FTemplate), @LMsgs[0], Length(LMsgs), True,
    PAnsiChar(Result), Length(Result));
  if N > Length(Result) then
  begin
    SetLength(Result, N);
    N := llama_chat_apply_template(PAnsiChar(FTemplate), @LMsgs[0], Length(LMsgs), True,
      PAnsiChar(Result), Length(Result));
  end;
  if N < 0 then
    raise EEngine.Create('llama.cpp does not know this model''s chat template');
  SetLength(Result, N);
end;

function TLlamaEngine.Tokenize(const AText: string): TLlamaTokens;
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
    raise EEngine.Create('tokenize failed');
  SetLength(Result, N);
end;

function TLlamaEngine.Piece(AToken: TLlamaToken): string;
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

function TLlamaEngine.NewSampler(ATemperature: Double; ANCtx: Integer): PLlamaSampler;
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

procedure TLlamaEngine.ApplyAdapter(AUse: Boolean);
begin
  if AUse and (FAdapter <> nil) then
    llama_set_adapters_lora(FCtx, @FAdapter, 1, @FScale)
  else
    llama_set_adapters_lora(FCtx, nil, 0, nil);
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

function TLlamaEngine.Run(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
  AMaxNewTokens: Integer; ATemperature: Double): string;
var
  LToks: TLlamaTokens;
  LTok: TLlamaToken;
  NPrompt, NCtx, NBatch, I, LDone, LChunk, LWhole: Integer;
  LSmpl: PLlamaSampler;
  LPending: string;
begin
  if AMaxNewTokens <= 0 then
    AMaxNewTokens := FCfg.MaxNewTokens;
  LToks := Tokenize(Prompt(AMessages));
  NPrompt := Length(LToks);
  NCtx := llama_n_ctx(FCtx);
  if NPrompt >= NCtx then
    raise EEngine.CreateFmt('prompt is %d tokens, the context holds %d (ctx_tokens in models.json)',
      [NPrompt, NCtx]);
  if NPrompt + AMaxNewTokens > NCtx then
    AMaxNewTokens := NCtx - NPrompt;

  llama_memory_clear(llama_get_memory(FCtx), True);
  ApplyAdapter(AUseAdapter);
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
        raise EEngine.Create('llama_decode failed on the prompt');
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
        raise EEngine.Create('llama_decode failed');
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

function TLlamaEngine.Generate(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
  AMaxNewTokens: Integer): string;
begin
  Result := Run(AMessages, AUseAdapter, AOnToken, AMaxNewTokens, FTemperature);
end;

function TLlamaEngine.ExtractFacts(const AText, AMode: string): TJSONArray;
var
  LUseLlm: Boolean;
  LMsgs: TJSONArray;
  LOut: string;
begin
  (* same contract as worker.py op_extract_facts: "rules" is the instant pass,
     "llm" the model pass (base model, no adapter), "auto" rules else model *)
  LUseLlm := (Pos('llm', FCfg.Extract) > 0) and HasStatement(AText);
  if AMode = 'llm' then
    Result := TJSONArray.Create
  else
    Result := ExtractFactsRules(AText);
  if (AMode = 'rules') or not LUseLlm or ((AMode = 'auto') and (Result.Count > 0)) then
    Exit;
  Result.Free;
  LMsgs := TJSONArray.Create([TJSONObject.Create(['role', 'user', 'content', LLM_EXTRACT_PROMPT + AText])]);
  try
    LOut := Run(LMsgs, False, nil, 200, 0.0);
  finally
    LMsgs.Free;
  end;
  Result := FactsFromLlmOutput(LOut, AText);
end;

procedure TLlamaEngine.ExportAdapter(const ADir, AGguf: string);
var
  P: TProcess;
  LOut: string;
  LBuf: array[0..4095] of Char;
  N: Integer;
begin
  (* An adapter trained before this engine was selected has no GGUF yet.
     Convert it now with the same helper consolidate.py uses. *)
  P := TProcess.Create(nil);
  try
    P.Executable := FCfg.PythonExe;
    P.Parameters.AddStrings(['-u', FCfg.TrainerScript('gguf_export.py'), 'adapter', ADir,
      '--profile', FCfg.ProfileName, '--models', FCfg.ModelsFile, '--llama-src', FCfg.LlamaSrcDir]);
    P.Options := [poUsePipes, poStderrToOutPut];
    InheritEnvironment(P);
    P.Environment.Add(GpuEnv(-1));
    P.Execute;
    LOut := '';
    repeat
      N := P.Output.Read(LBuf, SizeOf(LBuf));
      if N > 0 then
      begin
        SetLength(LOut, Length(LOut) + N);
        Move(LBuf[0], LOut[Length(LOut) - N + 1], N);
      end;
    until N <= 0;
    P.WaitOnExit;
    if (P.ExitCode <> 0) or not FileExists(AGguf) then
      raise EEngine.CreateFmt('could not convert %s to GGUF: %s', [ADir, Trim(LOut)]);
  finally
    P.Free;
  end;
end;

procedure TLlamaEngine.LoadAdapter(const APath, AName: string);
var
  LGguf: string;
  LNew: PLlamaAdapter;
begin
  LGguf := AdapterGguf(APath);
  if not FileExists(LGguf) then
    ExportAdapter(APath, LGguf);
  LNew := llama_adapter_lora_init(FModel, PAnsiChar(LGguf));
  if LNew = nil then
    raise EEngine.CreateFmt('llama.cpp could not load the adapter %s (see llama.log)', [LGguf]);
  { the swap: the next Generate attaches the new adapter }
  if FAdapter <> nil then
  begin
    llama_set_adapters_lora(FCtx, nil, 0, nil);
    llama_adapter_lora_free(FAdapter);
  end;
  FAdapter := LNew;
  FActive := AName;
  FActivePath := APath;
end;

procedure TLlamaEngine.ClearAdapter;
begin
  if FAdapter <> nil then
  begin
    llama_set_adapters_lora(FCtx, nil, 0, nil);
    llama_adapter_lora_free(FAdapter);
    FAdapter := nil;
  end;
  FActive := '';
  FActivePath := '';
end;

function TLlamaEngine.ActiveAdapter: string;
begin
  Result := FActive;
end;

function TLlamaEngine.Describe: string;
begin
  Result := Format('%s (%s) on %s via llama.cpp in-process', [FCfg.ProfileName,
    ExtractFileName(FCfg.GgufModel), FDevice]);
end;

finalization
  if GLogOpen then
    CloseFile(GLog);
end.
