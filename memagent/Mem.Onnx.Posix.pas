unit Mem.Onnx.Posix;

(* Downloads and installs the ONNX Runtime shared library on Linux / macOS
   (Mem.Onnx.Provision only auto-downloads it on win-x64). Fetches the
   release tarball from GitHub, extracts the real (non-symlink) library and
   drops it in the cache dir as onnxruntime.so / onnxruntime.dylib, which is
   where EnsureOnnxRuntime looks for it. *)

{$mode objfpc}{$H+}

interface

const
  { pinned to the C API version the Mem.Onnx.Api bindings request }
  ORT_POSIX_VERSION = '1.20.1';

{ True when the library is in ACacheDir afterwards (already there, or freshly
  installed). Always False on Windows. AMsg is a short status for the user. }
function EnsurePosixOrt(const ACacheDir: string; out AMsg: string): Boolean;

implementation

uses
  SysUtils, Classes, Mem.Https;

function InstalledLibPath(const ACacheDir: string): string;
begin
  {$IFDEF DARWIN}
  Result := IncludeTrailingPathDelimiter(ACacheDir) + 'onnxruntime.dylib';
  {$ELSE}
  Result := IncludeTrailingPathDelimiter(ACacheDir) + 'onnxruntime.so';
  {$ENDIF}
end;

function EnsurePosixOrt(const ACacheDir: string; out AMsg: string): Boolean;
{$IFDEF MSWINDOWS}
begin
  AMsg := 'windows uses Mem.Onnx.Provision';
  Result := False;
end;
{$ELSE}
var
  LAsset, LGlob, LTgz, LTmp, LDest: string;
begin
  LDest := InstalledLibPath(ACacheDir);
  if FileExists(LDest) then
  begin
    AMsg := 'already installed';
    Exit(True);
  end;
  {$IFDEF DARWIN}
  LAsset := 'onnxruntime-osx-universal2-' + ORT_POSIX_VERSION + '.tgz';
  LGlob := 'libonnxruntime*.dylib';
  {$ELSE}
    {$IFDEF CPUAARCH64}
  LAsset := 'onnxruntime-linux-aarch64-' + ORT_POSIX_VERSION + '.tgz';
    {$ELSE}
  LAsset := 'onnxruntime-linux-x64-' + ORT_POSIX_VERSION + '.tgz';
    {$ENDIF}
  LGlob := 'libonnxruntime.so.*';
  {$ENDIF}
  ForceDirectories(ACacheDir);
  LTgz := IncludeTrailingPathDelimiter(ACacheDir) + LAsset;
  LTmp := IncludeTrailingPathDelimiter(ACacheDir) + 'ort-extract';
  Result := False;
  try
    HttpsGetToFile('https://github.com/microsoft/onnxruntime/releases/download/v' +
      ORT_POSIX_VERSION + '/' + LAsset, LTgz);
    ExecuteProcess('/bin/sh', ['-c', 'rm -rf "' + LTmp + '" && mkdir -p "' + LTmp +
      '" && tar -xzf "' + LTgz + '" -C "' + LTmp + '" && cp "$(find "' + LTmp +
      '" -name ''' + LGlob + ''' -type f | head -1)" "' + LDest + '"']);
    Result := FileExists(LDest);
    if Result then
      AMsg := 'installed at ' + LDest
    else
      AMsg := 'extract failed (no ' + LGlob + ' in archive)';
  except
    on E: Exception do
      AMsg := E.Message;
  end;
  DeleteFile(LTgz);
  ExecuteProcess('/bin/sh', ['-c', 'rm -rf "' + LTmp + '"']);
end;
{$ENDIF}

end.
