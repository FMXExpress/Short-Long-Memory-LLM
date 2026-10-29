unit Mem.Setup;

(* Startup: what makes an installed memagent "just run".

   - Engine choice. engine=auto (the default) runs the model in-process
     through llama.cpp when it loads, and falls back to the Python worker
     when it does not (no libllama, no GGUF model and no way to fetch one).
   - First-run downloads. A missing GGUF base model is fetched from the
     profile's "gguf" url in models.json, and the local retrieval models
     (ONNX Runtime, embedder, reranker, guard) are provisioned once. Both
     use Mem.Https; --offline or auto_provision=0 turns this off.
   - Training check. Long-term memory needs Python with
     trainer/requirements.txt; without it the agent still chats with
     short-term memory and says once that it will not consolidate. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, StrUtils, Mem.Config, Mem.Engine;

{ fetches the GGUF base model when it is missing; False when it is still missing }
function EnsureGguf(ACfg: TMemConfig; AVerbose: Boolean): Boolean;
{ ONNX Runtime + embedder + reranker (+ guard model); True when all are ready }
function EnsureRetrieval(ACfg: TMemConfig; AForce: Boolean): Boolean;
{ the engine [agent] engine asks for; resolves ACfg.Engine to llama or python }
function CreateEngine(ACfg: TMemConfig): TEngine;
{ the Python interpreter for the trainer, '' when there is none }
function FindPython(const AExe: string): string;

implementation

uses
  Mem.Https, Mem.Retrieval, Mem.Engine.Llama, Mem.Llama;

function EnsureGguf(ACfg: TMemConfig; AVerbose: Boolean): Boolean;
begin
  Result := FileExists(ACfg.GgufModel);
  if Result or (ACfg.GgufUrl = '') then
    Exit;
  WriteLn('[setup] downloading the ', ACfg.ProfileName, ' model (one time) -> ', ACfg.GgufModel);
  try
    HttpsGetToFile(ACfg.GgufUrl, ACfg.GgufModel, AVerbose);
    Result := True;
  except
    on E: Exception do
      WriteLn(StdErr, '[setup] model download failed: ', E.Message);
  end;
end;

function RetrievalMarker(ACfg: TMemConfig): string;
begin
  Result := IncludeTrailingPathDelimiter(ACfg.CacheDir) + 'provisioned-' + ACfg.EmbedModel + '-' +
    IfThen(ACfg.Reranker = 'off', 'none', ACfg.RerankModel) + '-' +
    IfThen(ACfg.Guard = 'off', 'none', ACfg.GuardModel);
end;

function EnsureRetrieval(ACfg: TMemConfig; AForce: Boolean): Boolean;
var
  F: TextFile;
begin
  Result := FileExists(RetrievalMarker(ACfg));
  if Result and not AForce then
    Exit;
  if not AForce then
    WriteLn('[setup] fetching the local retrieval models (one time; --offline skips this)');
  Result := ProvisionRetrieval(ACfg.CacheDir, ACfg.EmbedModel,
      IfThen(ACfg.Reranker = 'off', '', ACfg.RerankModel)) and
    ((ACfg.Guard = 'off') or ProvisionRetrieval(ACfg.CacheDir, ACfg.EmbedModel, ACfg.GuardModel));
  if Result then
  begin
    AssignFile(F, RetrievalMarker(ACfg));
    Rewrite(F);
    WriteLn(F, 'ok');
    CloseFile(F);
  end
  else
    WriteLn(StdErr, '[setup] retrieval models unavailable; using the built-in lexical embedder');
end;

function CreateEngine(ACfg: TMemConfig): TEngine;
begin
  if ACfg.Engine <> 'python' then
  begin
    if ACfg.AutoProvision then
      EnsureGguf(ACfg, False);
    try
      Result := TLlamaEngine.Create(ACfg);
      ACfg.Engine := 'llama';   { the consolidator writes adapter.gguf for it }
      Exit;
    except
      on E: Exception do
      begin
        if ACfg.Engine = 'llama' then
          raise;
        WriteLn('[setup] llama.cpp engine unavailable (', E.Message, '); using the Python worker');
      end;
    end;
  end;
  ACfg.Engine := 'python';
  Result := TPyWorkerEngine.Create(ACfg);
end;

function FindPython(const AExe: string): string;
begin
  if (Pos('/', AExe) > 0) or (Pos('\', AExe) > 0) then
  begin
    if FileExists(AExe) then
      Exit(AExe);
    Exit('');
  end;
  Result := ExeSearch(AExe, GetEnvironmentVariable('PATH'));
  {$IFDEF MSWINDOWS}
  if Result = '' then
    Result := ExeSearch(AExe + '.exe', GetEnvironmentVariable('PATH'));
  {$ENDIF}
end;

end.
