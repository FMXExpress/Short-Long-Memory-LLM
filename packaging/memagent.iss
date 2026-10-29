; Inno Setup script for memagent (Windows x64).
;
; Built by .github/workflows/release.yml:
;   iscc /DAppVersion=0.1.0 /DSrcDir=<staged release folder> /DOutDir=dist packaging\memagent.iss
;
; The staged folder holds memagent.exe, the llama.cpp DLLs, sqlite3.dll, the
; OpenSSL 3 DLLs, cacert.pem, memagent.ini, models.json, priors.md, trainer\
; and llama.cpp\ (GGUF converters). memagent keeps its memory (data\) and the
; models it downloads (cache\) next to the exe, so it installs per user
; (no admin rights, {autopf} = %LOCALAPPDATA%\Programs).

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef SrcDir
  #define SrcDir "..\dist\memagent-windows-x64"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

[Setup]
AppId={{6C3E9B4A-2F1D-4E8B-9A57-0D3C8E1F7B42}
AppName=memagent
AppVersion={#AppVersion}
AppVerName=memagent {#AppVersion}
AppPublisher=FMXExpress
AppPublisherURL=https://github.com/FMXExpress/Short-Long-Memory-LLM
AppSupportURL=https://github.com/FMXExpress/Short-Long-Memory-LLM/issues
DefaultDirName={autopf}\memagent
DefaultGroupName=memagent
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=memagent-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile={#SrcDir}\LICENSE
UninstallDisplayIcon={app}\memagent.exe
ChangesEnvironment=yes

[Tasks]
Name: "addtopath"; Description: "Add memagent to my PATH (run it from any terminal)"; Flags: unchecked
Name: "provision"; Description: "Download the chat and retrieval models now (about 650 MB; otherwise on first start)"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Excludes: "memagent.ini"; Flags: ignoreversion recursesubdirs createallsubdirs
; keep a memagent.ini the user has edited
Source: "{#SrcDir}\memagent.ini"; DestDir: "{app}"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{group}\memagent"; Filename: "{app}\memagent.exe"; WorkingDir: "{app}"
Name: "{group}\memagent settings (memagent.ini)"; Filename: "{app}\memagent.ini"
Name: "{group}\Read me"; Filename: "{app}\README.md"
Name: "{group}\Uninstall memagent"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\memagent.exe"; Parameters: "--provision"; WorkingDir: "{app}"; \
  StatusMsg: "Downloading the models..."; Tasks: provision; Flags: waituntilterminated
Filename: "{app}\memagent.exe"; WorkingDir: "{app}"; Description: "Start memagent"; \
  Flags: postinstall nowait skipifsilent

[UninstallDelete]
; downloaded models only; your memory in data\ stays unless you delete it
Type: filesandordirs; Name: "{app}\cache"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Path: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') = 0;
end;
