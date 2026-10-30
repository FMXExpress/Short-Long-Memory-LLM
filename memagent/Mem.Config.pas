unit Mem.Config;

(* memagent.ini plus the selected model profile from models.json.
   The profile is the only thing that changes between the tiny CPU model
   used for development and a large model on rented GPUs. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, IniFiles, fpjson, jsonparser;

type
  TMemConfig = class
  public
    ConfigFile: string;
    RepoDir: string;
    DataDir: string;
    PriorsFile: string;
    PythonExe: string;
    ModelsFile: string;
    ProfileName: string;
    Engine: string;          { auto (llama when it loads, else python) | python | llama }
    AutoProvision: Boolean;  { download missing models on startup (off: --offline / auto_provision=0) }
    InferGpu: Integer;
    TrainGpu: Integer;
    RecentTurns: Integer;
    RagK: Integer;
    MinFactConfidence: Double;
    RagIncludeBaked: Boolean;  { baked facts stay retrievable (measured: +15 points on the tiny profile) }
    AutoConsolidate: Boolean;
    Trainer: string;         { llama (in-process, Mem.Train) | python (trainer/consolidate.py) }
    TrainThreads: Integer;   { CPU threads for the llama trainer, 0 = all cores but one }
    MinNewFacts: Integer;
    IntervalMin: Integer;
    { [retrieval]: local embedder + reranker (Mem.Retrieval) }
    CacheDir: string;
    Embedder: string;        { auto | onnx | static | off }
    EmbedModel: string;
    Reranker: string;        { auto | off }
    RerankModel: string;
    RetrievalPool: Integer;  { candidates from the hybrid stage before reranking }
    MinSimilarity: Double;   { cosine floor for vector-only hits (onnx embedder) }
    StaticMinSimilarity: Double;
    RerankMin: Double;       { drop reranked candidates below this score }
    DedupMinCos: Double;     { same-subject test: cosine prefilter ... }
    DedupMinRerank: Double;  { ... confirmed by the cross-encoder }
    DedupTextCos: Double;    { same value + text cosine at least this: a repeat, not a new fact }
    Provision: Boolean;      { --provision: download the retrieval models and exit }
    { [guard]: licensing check against stored-value confabulation (Mem.Guard) }
    Guard: string;           { auto (on when its model is provisioned) | on | off }
    GuardModel: string;
    GuardMinScore: Double;
    { [llama]: the in-process llama.cpp engine (Mem.Engine.Llama) }
    LlamaLibDir: string;     { libllama + libggml*; default <cache_dir>/llama (scripts/llama_setup.sh) }
    LlamaSrcDir: string;     { llama.cpp checkout for convert_lora_to_gguf.py; default <cache_dir>/llama.cpp }
    GgufModel: string;       { the base model as GGUF; default <cache_dir>/gguf/<profile>.gguf }
    GgufUrl: string;         (* where to download it when missing: models.json "gguf": {"url": ..} *)
    LlamaThreads: Integer;   { 0 = all cores but one }
    LlamaGpuLayers: Integer; { -1 = all layers on [gpu] infer (none when infer = -1) }
    { from the profile }
    BaseModel: string;
    MaxNewTokens: Integer;
    CtxTokens: Integer;
    Summarize: Boolean;
    Tools: Boolean;
    Temperature: Double;
    Extract: string;         { rules | rules+llm }
    ProfileTrainer: string;  { the trainer the profile supports: llama, or python only (models.json "trainer") }
    LoraRank: Integer;
    LoraAlpha: Double;
    LoraTargets: string;     { llama.cpp weight names, comma-separated (from "all-linear" or HF names) }
    TrainEpochs: Integer;
    TrainLr: Double;
    TrainBatch: Integer;
    TrainMaxLen: Integer;
    TrainParaphrases: Integer;
    TrainAnchors: Integer;
    GateMinRecall: Double;
    GateMaxPplRatio: Double;
    procedure Load;
    function Priors: string;
    function DbPath: string;
    function AdapterDir: string;
    function LogDir: string;
    function TrainerScript(const AName: string): string;
  end;

function GpuEnv(AGpu: Integer): string;
function DeviceName(AGpu: Integer): string;

implementation

function DeviceName(AGpu: Integer): string;
begin
  if AGpu < 0 then
    Result := 'CPU'
  else
    Result := 'GPU ' + IntToStr(AGpu);
end;

function GpuEnv(AGpu: Integer): string;
begin
  { -1 hides every GPU so torch falls back to the CPU }
  if AGpu < 0 then
    Result := 'CUDA_VISIBLE_DEVICES='
  else
    Result := 'CUDA_VISIBLE_DEVICES=' + IntToStr(AGpu);
end;

function Abs_(const ABase, APath: string): string;
begin
  if (APath <> '') and (APath[1] = PathDelim) then
    Result := APath
  else
    Result := ExpandFileName(IncludeTrailingPathDelimiter(ABase) + APath);
end;

(* where libllama is when [llama] lib_dir is not set: a source build
   (scripts/llama_setup.sh) in <cache_dir>/llama, else the copy a release
   ships next to the binary (lib/ on Linux, the exe folder on Windows) *)
function DefaultLlamaLibDir(const ACacheDir: string): string;
const
  LIB = {$IFDEF MSWINDOWS}'llama.dll'{$ELSE}{$IFDEF DARWIN}'libllama.dylib'{$ELSE}'libllama.so'{$ENDIF}{$ENDIF};
var
  LDir: string;
begin
  for LDir in [IncludeTrailingPathDelimiter(ACacheDir) + 'llama', ExtractFilePath(ParamStr(0)) + 'lib',
    ExtractFileDir(ParamStr(0))] do
    if FileExists(IncludeTrailingPathDelimiter(LDir) + LIB) then
      Exit(ExpandFileName(LDir));
  Result := IncludeTrailingPathDelimiter(ACacheDir) + 'llama';
end;

(* PEFT target modules -> llama.cpp weight names: "all-linear" is every
   projection of attention and MLP; a list of HF names maps one by one *)
function LoraTargetNames(ATargets: TJSONData): string;
const
  HF: array[0..6] of string = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj');
  GG: array[0..6] of string = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down');
var
  I, J: Integer;
  LName: string;
begin
  Result := 'attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down';
  if not (ATargets is TJSONArray) then
    Exit;
  Result := '';
  for I := 0 to ATargets.Count - 1 do
  begin
    LName := ATargets.Items[I].AsString;
    for J := 0 to High(HF) do
      if LName = HF[J] then
        LName := GG[J];
    Result := Result + IfThen(Result = '', '', ',') + LName;
  end;
end;

function IniFloat(AIni: TIniFile; const AKey: string; ADefault: Double): Double;
var
  LFmt: TFormatSettings;
begin
  LFmt := DefaultFormatSettings;
  LFmt.DecimalSeparator := '.';
  Result := StrToFloatDef(AIni.ReadString('retrieval', AKey, ''), ADefault, LFmt);
end;

procedure TMemConfig.Load;
var
  LFmt: TFormatSettings;
  I: Integer;
  LIni: TIniFile;
  LBase, LCliProfile, LCliData, LCliEngine: string;
  LCliNoAuto, LCliOffline, LCliConfig: Boolean;
  LJson: TJSONData;
  LProf: TJSONObject;
  LStream: TFileStream;
begin
  LFmt := DefaultFormatSettings;
  LFmt.DecimalSeparator := '.';
  ConfigFile := 'memagent.ini';
  LCliProfile := '';
  LCliData := '';
  LCliEngine := '';
  LCliNoAuto := False;
  LCliOffline := False;
  LCliConfig := False;
  I := 1;
  while I <= ParamCount do
  begin
    if (ParamStr(I) = '--config') and (I < ParamCount) then
    begin
      Inc(I);
      ConfigFile := ParamStr(I);
      LCliConfig := True;
    end
    else if (ParamStr(I) = '--profile') and (I < ParamCount) then
    begin
      Inc(I);
      LCliProfile := ParamStr(I);
    end
    else if (ParamStr(I) = '--data') and (I < ParamCount) then
    begin
      Inc(I);
      LCliData := ParamStr(I);
    end
    else if (ParamStr(I) = '--engine') and (I < ParamCount) then
    begin
      Inc(I);
      LCliEngine := ParamStr(I);
    end
    else if ParamStr(I) = '--no-auto' then
      LCliNoAuto := True
    else if ParamStr(I) = '--offline' then
      LCliOffline := True
    else if ParamStr(I) = '--provision' then
      Provision := True;
    Inc(I);
  end;

  (* An installed copy is started from anywhere: without --config and with no
     memagent.ini in the current directory, use the one next to the binary *)
  if not LCliConfig and not FileExists(ConfigFile) then
    ConfigFile := ExtractFilePath(ParamStr(0)) + 'memagent.ini';
  ConfigFile := ExpandFileName(ConfigFile);
  LBase := ExtractFileDir(ConfigFile);
  LIni := TIniFile.Create(ConfigFile);
  try
    RepoDir := Abs_(LBase, LIni.ReadString('agent', 'repo_dir', '.'));
    DataDir := Abs_(LBase, LIni.ReadString('agent', 'data_dir', 'data'));
    PriorsFile := Abs_(LBase, LIni.ReadString('agent', 'priors', 'priors.md'));
    ModelsFile := Abs_(LBase, LIni.ReadString('agent', 'models', 'models.json'));
    PythonExe := LIni.ReadString('agent', 'python', {$IFDEF MSWINDOWS}'python'{$ELSE}'python3'{$ENDIF});
    ProfileName := LIni.ReadString('agent', 'profile', '');
    Engine := LowerCase(LIni.ReadString('agent', 'engine', 'auto'));
    AutoProvision := LIni.ReadBool('agent', 'auto_provision', True);
    RecentTurns := LIni.ReadInteger('agent', 'recent_turns', 6);
    RagK := LIni.ReadInteger('agent', 'rag_k', 3);
    MinFactConfidence := StrToFloatDef(LIni.ReadString('agent', 'min_fact_confidence', '0.5'), 0.5);
    RagIncludeBaked := LIni.ReadBool('agent', 'rag_include_baked', True);
    InferGpu := LIni.ReadInteger('gpu', 'infer', 0);
    TrainGpu := LIni.ReadInteger('gpu', 'train', 1);
    AutoConsolidate := LIni.ReadBool('consolidate', 'auto', True);
    MinNewFacts := LIni.ReadInteger('consolidate', 'min_new_facts', 3);
    IntervalMin := LIni.ReadInteger('consolidate', 'interval_min', 60);
    Trainer := LowerCase(LIni.ReadString('consolidate', 'trainer', 'llama'));
    TrainThreads := LIni.ReadInteger('consolidate', 'threads', 0);
    CacheDir := Abs_(LBase, LIni.ReadString('retrieval', 'cache_dir', 'cache'));
    Embedder := LowerCase(LIni.ReadString('retrieval', 'embedder', 'auto'));
    EmbedModel := LIni.ReadString('retrieval', 'embed_model', 'minilm');
    Reranker := LowerCase(LIni.ReadString('retrieval', 'reranker', 'auto'));
    RerankModel := LIni.ReadString('retrieval', 'rerank_model', 'ms-marco-minilm');
    RetrievalPool := LIni.ReadInteger('retrieval', 'pool', 8);
    MinSimilarity := IniFloat(LIni, 'min_similarity', 0.25);
    StaticMinSimilarity := IniFloat(LIni, 'static_min_similarity', 0.3);
    RerankMin := IniFloat(LIni, 'rerank_min', 0.0001);
    DedupMinCos := IniFloat(LIni, 'dedup_min_cos', 0.7);
    DedupMinRerank := IniFloat(LIni, 'dedup_min_rerank', 0.5);
    DedupTextCos := IniFloat(LIni, 'dedup_text_cos', 0.9);
    Guard := LowerCase(LIni.ReadString('guard', 'enabled', 'auto'));
    GuardModel := LIni.ReadString('guard', 'model', 'bge-reranker-base');
    GuardMinScore := StrToFloatDef(LIni.ReadString('guard', 'min_score', '0.03'), 0.03, LFmt);
    { scripts/llama_setup.sh puts both under the cache directory }
    LlamaLibDir := LIni.ReadString('llama', 'lib_dir', '');
    if LlamaLibDir <> '' then
      LlamaLibDir := Abs_(LBase, LlamaLibDir)
    else
      LlamaLibDir := DefaultLlamaLibDir(CacheDir);
    LlamaSrcDir := Abs_(LBase, LIni.ReadString('llama', 'src_dir',
      IncludeTrailingPathDelimiter(CacheDir) + 'llama.cpp'));
    GgufModel := LIni.ReadString('llama', 'model', '');
    if GgufModel <> '' then
      GgufModel := Abs_(LBase, GgufModel);
    LlamaThreads := LIni.ReadInteger('llama', 'threads', 0);
    LlamaGpuLayers := LIni.ReadInteger('llama', 'gpu_layers', -1);
  finally
    LIni.Free;
  end;
  if LCliProfile <> '' then
    ProfileName := LCliProfile;
  if LCliData <> '' then
    DataDir := ExpandFileName(LCliData);
  if LCliNoAuto then
    AutoConsolidate := False;
  if LCliOffline then
    AutoProvision := False;
  if LCliEngine <> '' then
    Engine := LowerCase(LCliEngine);
  if (Engine <> 'auto') and (Engine <> 'python') and (Engine <> 'llama') then
    raise Exception.CreateFmt('[agent] engine must be auto, llama or python, not "%s"', [Engine]);
  if (Trainer <> 'python') and (Trainer <> 'llama') then
    raise Exception.CreateFmt('[consolidate] trainer must be llama or python, not "%s"', [Trainer]);

  LStream := TFileStream.Create(ModelsFile, fmOpenRead or fmShareDenyNone);
  try
    LJson := GetJSON(LStream);
  finally
    LStream.Free;
  end;
  try
    if ProfileName = '' then
      ProfileName := TJSONObject(LJson).Get('default', 'tiny');
    LProf := TJSONObject(LJson).Objects['profiles'].Find(ProfileName) as TJSONObject;
    if LProf = nil then
      raise Exception.CreateFmt('profile "%s" not found in %s', [ProfileName, ModelsFile]);
    BaseModel := LProf.Get('base', '');
    MaxNewTokens := LProf.Get('max_new_tokens', 256);
    CtxTokens := LProf.Get('ctx_tokens', 2048);
    Summarize := LProf.Get('summarize', False);
    Tools := LProf.Get('tools', False);
    Temperature := LProf.Get('temperature', 0.0);
    Extract := LProf.Get('extract', 'rules');
    if LProf.Find('gguf') is TJSONObject then
      GgufUrl := LProf.Objects['gguf'].Get('url', '');
    ProfileTrainer := LowerCase(LProf.Get('trainer', 'llama'));
    if LProf.Find('lora') is TJSONObject then
    begin
      LoraRank := LProf.Objects['lora'].Get('r', 16);
      LoraAlpha := LProf.Objects['lora'].Get('alpha', 32.0);
      LoraTargets := LoraTargetNames(LProf.Objects['lora'].Find('targets'));
    end;
    if LProf.Find('train') is TJSONObject then
      with LProf.Objects['train'] do
      begin
        TrainEpochs := Get('epochs', 4);
        TrainLr := Get('lr', 2e-3);
        TrainBatch := Get('batch', 8) * Get('grad_accum', 1);
        TrainMaxLen := Get('max_len', 256);
        TrainParaphrases := Get('paraphrases', 6);
        TrainAnchors := Get('anchors', 8);
      end;
    if LProf.Find('gate') is TJSONObject then
    begin
      GateMinRecall := LProf.Objects['gate'].Get('min_recall', 0.6);
      GateMaxPplRatio := LProf.Objects['gate'].Get('max_ppl_ratio', 3.0);
    end;
  finally
    LJson.Free;
  end;

  if GgufModel = '' then
    GgufModel := IncludeTrailingPathDelimiter(CacheDir) + 'gguf' + PathDelim + ProfileName + '.gguf';

  ForceDirectories(DataDir);
  ForceDirectories(AdapterDir);
  ForceDirectories(LogDir);
end;

function TMemConfig.Priors: string;
var
  L: TStringList;
begin
  Result := 'You are a helpful assistant.';
  if not FileExists(PriorsFile) then
    Exit;
  L := TStringList.Create;
  try
    L.LoadFromFile(PriorsFile);
    Result := Trim(L.Text);
  finally
    L.Free;
  end;
end;

function TMemConfig.DbPath: string;
begin
  Result := IncludeTrailingPathDelimiter(DataDir) + 'memory.db';
end;

function TMemConfig.AdapterDir: string;
begin
  Result := IncludeTrailingPathDelimiter(DataDir) + 'adapters';
end;

function TMemConfig.LogDir: string;
begin
  Result := IncludeTrailingPathDelimiter(DataDir) + 'logs';
end;

function TMemConfig.TrainerScript(const AName: string): string;
begin
  Result := IncludeTrailingPathDelimiter(RepoDir) + 'trainer' + PathDelim + AName;
end;

end.
