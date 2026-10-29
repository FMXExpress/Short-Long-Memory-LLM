unit Mem.Download;

(* Fetches the local retrieval models (embedder + reranker) over verified
   HTTPS (see Mem.Https). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Mem.Embed.Models;

const
  MODEL_FILE = 'model.onnx';
  VOCAB_FILE = 'vocab.txt';

type
  TModelDownloader = class
  private
    FSpec: TModelSpec;
    FModelDir, FModelPath, FVocabPath: string;
    FVerbose: Boolean;
  public
    constructor Create(const ASpec: TModelSpec; const AModelDir: string; AVerbose: Boolean = False);
    { downloads whichever of model.onnx / vocab file is missing }
    procedure EnsureFiles;
    property ModelPath: string read FModelPath;
    property VocabPath: string read FVocabPath;
  end;

function FileSizeOf(const APath: string): Int64;

implementation

uses
  Mem.Https;

function FileSizeOf(const APath: string): Int64;
var
  LInfo: TSearchRec;
begin
  Result := -1;
  if FindFirst(APath, faAnyFile, LInfo) = 0 then
  begin
    Result := LInfo.Size;
    FindClose(LInfo);
  end;
end;

constructor TModelDownloader.Create(const ASpec: TModelSpec; const AModelDir: string; AVerbose: Boolean);
begin
  inherited Create;
  FSpec := ASpec;
  FModelDir := AModelDir;
  FVerbose := AVerbose;
  FModelPath := IncludeTrailingPathDelimiter(FModelDir) + MODEL_FILE;
  { the vocab file is vocab.txt (WordPiece) or tokenizer.json (SentencePiece);
    either way it is stored under one name so loaders need no extra config }
  FVocabPath := IncludeTrailingPathDelimiter(FModelDir) + VOCAB_FILE;
end;

procedure TModelDownloader.EnsureFiles;
begin
  ForceDirectories(FModelDir);
  if FVerbose then
    WriteLn(StdErr, '[download] model ', FSpec.DisplayName, ' (', FSpec.SizeDesc, ') -> ', FModelDir);
  if not FileExists(FVocabPath) then
    HttpsGetToFile(FSpec.BaseURL + FSpec.VocabRelURL, FVocabPath, FVerbose);
  { a suspiciously small model file is a failed or partial download }
  if FileSizeOf(FModelPath) < 1024 * 1024 then
    HttpsGetToFile(FSpec.BaseURL + FSpec.ModelRelURL, FModelPath, FVerbose);
end;

end.
