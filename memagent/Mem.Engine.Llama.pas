unit Mem.Engine.Llama;

(* TEngine on llama.cpp, in-process: the GGUF base model is loaded into the
   agent itself and LoRA memories are hot-swapped with
   llama_set_adapters_lora, so a turn needs no Python at all. The adapters
   come from the in-process trainer (Mem.Train) as adapter.gguf; an adapter
   the Python trainer made (PEFT) is converted on first load.

   Generation itself is Mem.LlamaChat, shared with the trainer. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Process, fpjson, Mem.Config, Mem.Engine, Mem.Llama, Mem.LlamaChat, Mem.Extract;

type
  TLlamaEngine = class(TEngine)
  private
    FCfg: TMemConfig;
    FModel: PLlamaModel;
    FChat: TLlamaChat;
    FAdapter: PLlamaAdapter;
    FActive, FActivePath: string;
    FDevice: string;
    FTemperature: Double;
    FOnToken: TTokenEvent;
    procedure EmitToken(const AText: string);
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

{ <adapter dir>/adapter.gguf: the adapter as this engine loads it }
function AdapterGguf(const ADir: string): string;

implementation

function AdapterGguf(const ADir: string): string;
begin
  Result := IncludeTrailingPathDelimiter(ADir) + 'adapter.gguf';
end;

constructor TLlamaEngine.Create(ACfg: TMemConfig);
begin
  inherited Create;
  FCfg := ACfg;
  FTemperature := StrToFloatDef(GetEnvironmentVariable('MEMAGENT_TEMPERATURE'), FCfg.Temperature);
  if not FileExists(FCfg.GgufModel) then
    raise EEngine.CreateFmt('no GGUF model at %s; run scripts/llama_setup.sh %s (or set [llama] model)',
      [FCfg.GgufModel, FCfg.ProfileName]);
  LlamaSetup(FCfg);
  FModel := LlamaLoadModel(FCfg.GgufModel, FCfg.InferGpu, FCfg.LlamaGpuLayers);
  FChat := TLlamaChat.Create(FModel, FCfg.CtxTokens, LlamaThreads(FCfg.LlamaThreads));
  FDevice := DeviceName(FCfg.InferGpu);
  LlamaTrace('ready');
end;

destructor TLlamaEngine.Destroy;
begin
  FChat.Free;
  if FAdapter <> nil then
    llama_adapter_lora_free(FAdapter);
  if FModel <> nil then
    llama_model_free(FModel);
  inherited Destroy;
end;

procedure TLlamaEngine.EmitToken(const AText: string);
begin
  FOnToken(AText);
end;

function TLlamaEngine.Run(AMessages: TJSONArray; AUseAdapter: Boolean; AOnToken: TTokenEvent;
  AMaxNewTokens: Integer; ATemperature: Double): string;
var
  LAdapter: PLlamaAdapter;
begin
  if AMaxNewTokens <= 0 then
    AMaxNewTokens := FCfg.MaxNewTokens;
  LAdapter := nil;
  if AUseAdapter then
    LAdapter := FAdapter;
  FOnToken := AOnToken;
  try
    if Assigned(AOnToken) then
      Result := FChat.Generate(AMessages, LAdapter, AMaxNewTokens, ATemperature, @EmitToken)
    else
      Result := FChat.Generate(AMessages, LAdapter, AMaxNewTokens, ATemperature);
  except
    on E: ELlama do
      raise EEngine.Create(E.Message);
  end;
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
    llama_set_adapters_lora(FChat.Context, nil, 0, nil);
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
    llama_set_adapters_lora(FChat.Context, nil, 0, nil);
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

end.
