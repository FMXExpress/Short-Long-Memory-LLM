unit Mem.Config;

(* memagent.ini plus the selected model profile from models.json.
   The profile is the only thing that changes between the tiny CPU model
   used for development and a large model on rented GPUs. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, IniFiles, fpjson, jsonparser;

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
    InferGpu: Integer;
    TrainGpu: Integer;
    RecentTurns: Integer;
    RagK: Integer;
    MinFactConfidence: Double;
    AutoConsolidate: Boolean;
    MinNewFacts: Integer;
    IntervalMin: Integer;
    { from the profile }
    BaseModel: string;
    MaxNewTokens: Integer;
    CtxTokens: Integer;
    Summarize: Boolean;
    Tools: Boolean;
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

procedure TMemConfig.Load;
var
  I: Integer;
  LIni: TIniFile;
  LBase, LCliProfile, LCliData: string;
  LCliNoAuto: Boolean;
  LJson: TJSONData;
  LProf: TJSONObject;
  LStream: TFileStream;
begin
  ConfigFile := 'memagent.ini';
  LCliProfile := '';
  LCliData := '';
  LCliNoAuto := False;
  I := 1;
  while I <= ParamCount do
  begin
    if (ParamStr(I) = '--config') and (I < ParamCount) then
    begin
      Inc(I);
      ConfigFile := ParamStr(I);
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
    else if ParamStr(I) = '--no-auto' then
      LCliNoAuto := True;
    Inc(I);
  end;

  ConfigFile := ExpandFileName(ConfigFile);
  LBase := ExtractFileDir(ConfigFile);
  LIni := TIniFile.Create(ConfigFile);
  try
    RepoDir := Abs_(LBase, LIni.ReadString('agent', 'repo_dir', '.'));
    DataDir := Abs_(LBase, LIni.ReadString('agent', 'data_dir', 'data'));
    PriorsFile := Abs_(LBase, LIni.ReadString('agent', 'priors', 'priors.md'));
    ModelsFile := Abs_(LBase, LIni.ReadString('agent', 'models', 'models.json'));
    PythonExe := LIni.ReadString('agent', 'python', 'python3');
    ProfileName := LIni.ReadString('agent', 'profile', '');
    RecentTurns := LIni.ReadInteger('agent', 'recent_turns', 6);
    RagK := LIni.ReadInteger('agent', 'rag_k', 3);
    MinFactConfidence := StrToFloatDef(LIni.ReadString('agent', 'min_fact_confidence', '0.5'), 0.5);
    InferGpu := LIni.ReadInteger('gpu', 'infer', 0);
    TrainGpu := LIni.ReadInteger('gpu', 'train', 1);
    AutoConsolidate := LIni.ReadBool('consolidate', 'auto', True);
    MinNewFacts := LIni.ReadInteger('consolidate', 'min_new_facts', 3);
    IntervalMin := LIni.ReadInteger('consolidate', 'interval_min', 60);
  finally
    LIni.Free;
  end;
  if LCliProfile <> '' then
    ProfileName := LCliProfile;
  if LCliData <> '' then
    DataDir := ExpandFileName(LCliData);
  if LCliNoAuto then
    AutoConsolidate := False;

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
  finally
    LJson.Free;
  end;

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
