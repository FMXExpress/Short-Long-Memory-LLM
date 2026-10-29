unit Mem.Llama;

(* Thin binding to the llama.cpp C API (llama.h), loaded at run time.

   Only the calls the agent needs: load a GGUF model, apply its chat
   template, tokenize, decode, sample, and attach/detach LoRA adapters.
   The library is loaded with dynlibs so the agent still builds and runs
   (on the Python engine) on a machine without llama.cpp.

   The record layouts must match the llama.h the library was built from.
   They are pinned to LLAMA_TAG below (scripts/llama_setup.sh builds that
   tag), and LlamaCheckLayout verifies the sizes the C compiler reported
   for it, so a mismatched library fails loudly instead of corrupting
   memory. *)

{$mode objfpc}{$H+}
{$packrecords C}

interface

uses
  SysUtils, Math, DynLibs, ctypes;

const
  LLAMA_TAG = 'b11985';
  LLAMA_DEFAULT_SEED = $FFFFFFFF;
  LLAMA_SPLIT_MODE_NONE = 0;

type
  PLlamaModel = Pointer;
  PLlamaContext = Pointer;
  PLlamaVocab = Pointer;
  PLlamaSampler = Pointer;
  PLlamaAdapter = Pointer;
  PLlamaMemory = Pointer;
  TLlamaToken = Int32;
  PLlamaToken = ^TLlamaToken;
  TLlamaTokens = array of TLlamaToken;

  TLlamaModelParams = record
    devices: Pointer;
    tensor_buft_overrides: Pointer;
    n_gpu_layers: Int32;
    split_mode: Int32;
    load_mode: Int32;
    lazy_mode: Int32;
    main_gpu: Int32;
    tensor_split: PSingle;
    progress_callback: Pointer;
    progress_callback_user_data: Pointer;
    kv_overrides: Pointer;
    vocab_only: Boolean;
    check_tensors: Boolean;
    use_extra_bufts: Boolean;
    no_host: Boolean;
    no_alloc: Boolean;
    load_mtp: Boolean;
  end;

  TLlamaContextParams = record
    n_ctx: UInt32;
    n_batch: UInt32;
    n_ubatch: UInt32;
    n_seq_max: UInt32;
    n_rs_seq: UInt32;
    n_outputs_max: UInt32;
    n_outputs_max_per_seq: UInt32;
    n_threads: Int32;
    n_threads_batch: Int32;
    ctx_type: Int32;
    rope_scaling_type: Int32;
    pooling_type: Int32;
    attention_type: Int32;
    flash_attn_type: Int32;
    rope_freq_base: Single;
    rope_freq_scale: Single;
    yarn_ext_factor: Single;
    yarn_attn_factor: Single;
    yarn_beta_fast: Single;
    yarn_beta_slow: Single;
    yarn_orig_ctx: UInt32;
    defrag_thold: Single;
    cb_eval: Pointer;
    cb_eval_user_data: Pointer;
    type_k: Int32;
    type_v: Int32;
    moe_cache_size: csize_t;
    abort_callback: Pointer;
    abort_callback_data: Pointer;
    embeddings: Boolean;
    offload_kqv: Boolean;
    no_perf: Boolean;
    op_offload: Boolean;
    swa_full: Boolean;
    kv_unified: Boolean;
    samplers: Pointer;
    n_samplers: csize_t;
    ctx_other: Pointer;
  end;

  TLlamaBatch = record
    n_tokens: Int32;
    token: PLlamaToken;
    embd: PSingle;
    pos: PInt32;
    n_seq_id: PInt32;
    seq_id: Pointer;
    logits: PShortInt;
  end;

  TLlamaSamplerChainParams = record
    no_perf: Boolean;
  end;

  TLlamaChatMessage = record
    role: PAnsiChar;
    content: PAnsiChar;
  end;
  PLlamaChatMessage = ^TLlamaChatMessage;

  TGgmlLogCallback = procedure(ALevel: Int32; AText: PAnsiChar; AUserData: Pointer); cdecl;

var
  llama_backend_init: procedure; cdecl;
  llama_log_set: procedure(ACallback: TGgmlLogCallback; AUserData: Pointer); cdecl;
  llama_model_default_params: function: TLlamaModelParams; cdecl;
  llama_context_default_params: function: TLlamaContextParams; cdecl;
  llama_sampler_chain_default_params: function: TLlamaSamplerChainParams; cdecl;
  llama_model_load_from_file: function(APath: PAnsiChar; AParams: TLlamaModelParams): PLlamaModel; cdecl;
  llama_model_free: procedure(AModel: PLlamaModel); cdecl;
  llama_init_from_model: function(AModel: PLlamaModel; AParams: TLlamaContextParams): PLlamaContext; cdecl;
  llama_free: procedure(ACtx: PLlamaContext); cdecl;
  llama_n_ctx: function(ACtx: PLlamaContext): UInt32; cdecl;
  llama_n_batch: function(ACtx: PLlamaContext): UInt32; cdecl;
  llama_model_get_vocab: function(AModel: PLlamaModel): PLlamaVocab; cdecl;
  llama_model_chat_template: function(AModel: PLlamaModel; AName: PAnsiChar): PAnsiChar; cdecl;
  llama_model_desc: function(AModel: PLlamaModel; ABuf: PAnsiChar; ASize: csize_t): Int32; cdecl;
  llama_vocab_n_tokens: function(AVocab: PLlamaVocab): Int32; cdecl;
  llama_vocab_is_eog: function(AVocab: PLlamaVocab; AToken: TLlamaToken): Boolean; cdecl;
  llama_chat_apply_template: function(ATmpl: PAnsiChar; AChat: PLlamaChatMessage; ANMsg: csize_t;
    AAddAss: Boolean; ABuf: PAnsiChar; ALength: Int32): Int32; cdecl;
  llama_tokenize: function(AVocab: PLlamaVocab; AText: PAnsiChar; ATextLen: Int32; ATokens: PLlamaToken;
    ANTokensMax: Int32; AAddSpecial, AParseSpecial: Boolean): Int32; cdecl;
  llama_token_to_piece: function(AVocab: PLlamaVocab; AToken: TLlamaToken; ABuf: PAnsiChar; ALength: Int32;
    ALstrip: Int32; ASpecial: Boolean): Int32; cdecl;
  llama_batch_get_one: function(ATokens: PLlamaToken; ANTokens: Int32): TLlamaBatch; cdecl;
  llama_decode: function(ACtx: PLlamaContext; ABatch: TLlamaBatch): Int32; cdecl;
  llama_get_memory: function(ACtx: PLlamaContext): PLlamaMemory; cdecl;
  llama_memory_clear: procedure(AMem: PLlamaMemory; AData: Boolean); cdecl;
  llama_sampler_chain_init: function(AParams: TLlamaSamplerChainParams): PLlamaSampler; cdecl;
  llama_sampler_chain_add: procedure(AChain, ASmpl: PLlamaSampler); cdecl;
  llama_sampler_init_greedy: function: PLlamaSampler; cdecl;
  llama_sampler_init_dist: function(ASeed: UInt32): PLlamaSampler; cdecl;
  llama_sampler_init_temp: function(ATemp: Single): PLlamaSampler; cdecl;
  llama_sampler_init_top_p: function(AP: Single; AMinKeep: csize_t): PLlamaSampler; cdecl;
  llama_sampler_init_penalties: function(ANVocab, ALastN: Int32; ARepeat, AFreq, APresent: Single): PLlamaSampler; cdecl;
  llama_sampler_sample: function(ASmpl: PLlamaSampler; ACtx: PLlamaContext; AIdx: Int32): TLlamaToken; cdecl;
  llama_sampler_free: procedure(ASmpl: PLlamaSampler); cdecl;
  llama_sampler_accept: procedure(ASmpl: PLlamaSampler; AToken: TLlamaToken); cdecl;
  llama_adapter_lora_init: function(AModel: PLlamaModel; APath: PAnsiChar): PLlamaAdapter; cdecl;
  llama_adapter_lora_free: procedure(AAdapter: PLlamaAdapter); cdecl;
  llama_set_adapters_lora: function(ACtx: PLlamaContext; AAdapters: Pointer; ANAdapters: csize_t;
    AScales: PSingle): Int32; cdecl;

{ Loads the llama.cpp libraries from ADir ('' = the system search path).
  Raises on failure; safe to call more than once. }
procedure LlamaLoad(const ADir: string);
function LlamaLoaded: Boolean;
{ MEMAGENT_DEBUG=1: one flushed line on stderr per step of loading and running
  the model, so a crash inside a native library still shows where it was }
procedure LlamaTrace(const AMsg: string);
{ the library file names for this platform, in load order }
function LlamaLibNames: TStringArray;

implementation

var
  GHandles: array of TLibHandle;
  GLoaded: Boolean;

function LlamaLibNames: TStringArray;
begin
  (* ggml first: preloading the dependencies by full path makes the loader
     find them by soname when libllama asks, whatever its RUNPATH says. *)
  {$if defined(windows)}
  Result := ['ggml-base.dll', 'ggml-cpu.dll', 'ggml.dll', 'llama.dll'];
  {$elseif defined(darwin)}
  Result := ['libggml-base.dylib', 'libggml-cpu.dylib', 'libggml.dylib', 'libllama.dylib'];
  {$else}
  Result := ['libggml-base.so', 'libggml-cpu.so', 'libggml.so', 'libllama.so'];
  {$endif}
end;

(* sizes from the C compiler for llama.h at LLAMA_TAG (x86-64 / arm64):
   a layout drift fails the build, not the first llama_decode *)
{$if (SizeOf(TLlamaModelParams) <> 80) or (SizeOf(TLlamaContextParams) <> 168) or (SizeOf(TLlamaBatch) <> 56) or (SizeOf(TLlamaChatMessage) <> 16)}
  {$error Mem.Llama record layout does not match llama.h}
{$endif}

procedure LlamaLoad(const ADir: string);
var
  LName, LPath, LMissing: string;
  H, LLlama: TLibHandle;

  procedure Bind(var AProc; const ASym: string);
  begin
    Pointer(AProc) := GetProcedureAddress(LLlama, ASym);
    if Pointer(AProc) = nil then
      LMissing := LMissing + ' ' + ASym;
  end;

begin
  if GLoaded then
    Exit;
  (* FPC unmasks floating-point exceptions; ggml's kernels rely on the C
     default (masked: inf and NaN, no trap), as every C caller has it *)
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow, exUnderflow, exPrecision]);
  LLlama := NilHandle;
  for LName in LlamaLibNames do
  begin
    if ADir <> '' then
      LPath := IncludeTrailingPathDelimiter(ADir) + LName
    else
      LPath := LName;
    LlamaTrace('loading ' + LPath);
    H := LoadLibrary(LPath);
    if H = NilHandle then
    begin
      (* the ggml split is a build detail: a static or merged build only
         ships libllama, so only that one is required *)
      if LName <> LlamaLibNames[High(LlamaLibNames)] then
        Continue;
      raise Exception.CreateFmt('cannot load %s (%s); run scripts/llama_setup.sh or set [llama] lib_dir',
        [LPath, GetLoadErrorStr]);
    end;
    SetLength(GHandles, Length(GHandles) + 1);
    GHandles[High(GHandles)] := H;
    LLlama := H;
  end;

  LMissing := '';
  Bind(llama_backend_init, 'llama_backend_init');
  Bind(llama_log_set, 'llama_log_set');
  Bind(llama_model_default_params, 'llama_model_default_params');
  Bind(llama_context_default_params, 'llama_context_default_params');
  Bind(llama_sampler_chain_default_params, 'llama_sampler_chain_default_params');
  Bind(llama_model_load_from_file, 'llama_model_load_from_file');
  Bind(llama_model_free, 'llama_model_free');
  Bind(llama_init_from_model, 'llama_init_from_model');
  Bind(llama_free, 'llama_free');
  Bind(llama_n_ctx, 'llama_n_ctx');
  Bind(llama_n_batch, 'llama_n_batch');
  Bind(llama_model_get_vocab, 'llama_model_get_vocab');
  Bind(llama_model_chat_template, 'llama_model_chat_template');
  Bind(llama_model_desc, 'llama_model_desc');
  Bind(llama_vocab_n_tokens, 'llama_vocab_n_tokens');
  Bind(llama_vocab_is_eog, 'llama_vocab_is_eog');
  Bind(llama_chat_apply_template, 'llama_chat_apply_template');
  Bind(llama_tokenize, 'llama_tokenize');
  Bind(llama_token_to_piece, 'llama_token_to_piece');
  Bind(llama_batch_get_one, 'llama_batch_get_one');
  Bind(llama_decode, 'llama_decode');
  Bind(llama_get_memory, 'llama_get_memory');
  Bind(llama_memory_clear, 'llama_memory_clear');
  Bind(llama_sampler_chain_init, 'llama_sampler_chain_init');
  Bind(llama_sampler_chain_add, 'llama_sampler_chain_add');
  Bind(llama_sampler_init_greedy, 'llama_sampler_init_greedy');
  Bind(llama_sampler_init_dist, 'llama_sampler_init_dist');
  Bind(llama_sampler_init_temp, 'llama_sampler_init_temp');
  Bind(llama_sampler_init_top_p, 'llama_sampler_init_top_p');
  Bind(llama_sampler_init_penalties, 'llama_sampler_init_penalties');
  Bind(llama_sampler_sample, 'llama_sampler_sample');
  Bind(llama_sampler_free, 'llama_sampler_free');
  Bind(llama_sampler_accept, 'llama_sampler_accept');
  Bind(llama_adapter_lora_init, 'llama_adapter_lora_init');
  Bind(llama_adapter_lora_free, 'llama_adapter_lora_free');
  Bind(llama_set_adapters_lora, 'llama_set_adapters_lora');
  if LMissing <> '' then
    raise Exception.Create('llama.cpp library is not ' + LLAMA_TAG + '-compatible, missing:' + LMissing);
  GLoaded := True;
  LlamaTrace('bound the ' + LLAMA_TAG + ' API');
end;

var
  GTrace: Integer = -1;

procedure LlamaTrace(const AMsg: string);
begin
  if GTrace < 0 then
    GTrace := Ord(GetEnvironmentVariable('MEMAGENT_DEBUG') = '1');
  if GTrace = 1 then
  begin
    WriteLn(StdErr, '[llama] ', AMsg);
    Flush(StdErr);
  end;
end;

function LlamaLoaded: Boolean;
begin
  Result := GLoaded;
end;

end.
