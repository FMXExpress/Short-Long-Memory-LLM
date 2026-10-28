unit Mem.Retrieval;

(* Local retrieval models for the RAG tier, all in-process via ONNX Runtime:

     embedder  sentence embeddings for facts and messages (MiniLM by default),
               falling back to the dependency-free lexical Mem.Embed.Static
               when the ONNX artifacts are not provisioned
     reranker  cross-encoder that re-scores (query, candidate) pairs after
               the hybrid FTS5 + vector stage (ms-marco-MiniLM by default)

   Everything lives under the cache dir:
     <cache>/onnxruntime.so                    (Linux; .dylib / .dll elsewhere)
     <cache>/models/<model>/model.onnx + vocab.txt *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, StrUtils,
  Mem.Embed.Models, Mem.Tokenizer, Mem.Embed.Onnx, Mem.Embed.Static, Mem.Rerank.Onnx;

type
  TVector = specialize TArray<Single>;
  TScoreArray = array of Single;

  TRetrieval = class
  private
    FCacheDir: string;
    FTier: string;          { onnx | static | off }
    FEmbedId: string;
    FSpec: TModelSpec;
    FTok: TBertTokenizer;
    FEmb: TEmbedder;
    FReranker: TReranker;
    FRerankKey: string;
    FNote: string;
    function TryLoadOnnxEmbedder(const AModel: string): Boolean;
    function TryLoadReranker(const AModel: string): Boolean;
  public
    constructor Create(const ACacheDir, AEmbedMode, AEmbedModel, ARerankMode, ARerankModel: string);
    destructor Destroy; override;
    { unit-length vector, or [] when embeddings are off / the text is empty }
    function Embed(const AText: string): TVector;
    { scores in the original order (0..1, sigmoid of the cross-encoder logit) }
    function RerankScores(const AQuery: string; const ADocs: array of string): TScoreArray;
    function HasReranker: Boolean;
    { semantic dedup is only trusted with a real encoder, never the lexical tier }
    function CanDedup: Boolean;
    function Describe: string;
    property Tier: string read FTier;
    property EmbedderId: string read FEmbedId;
  end;

function Cosine(const A, B: TVector): Single;
function VectorToBytes(const V: TVector): TBytes;
function BytesToVector(const B: TBytes): TVector;
function ModelDir(const ACacheDir, ASubDir: string): string;

{ Downloads ONNX Runtime + the embedding model (+ the reranker when ARerankModel
  is not empty) into ACacheDir. Prints progress; returns False on any failure. }
function ProvisionRetrieval(const ACacheDir, AEmbedModel, ARerankModel: string): Boolean;

implementation

uses
  Mem.Onnx.Provision, Mem.Onnx.Posix, Mem.Download;

function ModelDir(const ACacheDir, ASubDir: string): string;
begin
  Result := IncludeTrailingPathDelimiter(ACacheDir) + 'models' + PathDelim + ASubDir;
end;

function ArtifactsPresent(const ADir: string): Boolean;
begin
  Result := FileExists(IncludeTrailingPathDelimiter(ADir) + MODEL_FILE) and
    FileExists(IncludeTrailingPathDelimiter(ADir) + VOCAB_FILE);
end;

function LoadRuntime(const ACacheDir: string; out AError: string): Boolean;
begin
  AError := '';
  try
    Result := EnsureOnnxRuntime(ACacheDir, False, False);
  except
    on E: Exception do
    begin
      AError := E.Message;
      Result := False;
    end;
  end;
end;

function Cosine(const A, B: TVector): Single;
var
  I: Integer;
  LDot, LA, LB: Double;
begin
  Result := 0;
  if (Length(A) = 0) or (Length(A) <> Length(B)) then
    Exit;
  LDot := 0;
  LA := 0;
  LB := 0;
  for I := 0 to High(A) do
  begin
    LDot := LDot + A[I] * B[I];
    LA := LA + A[I] * A[I];
    LB := LB + B[I] * B[I];
  end;
  if (LA > 0) and (LB > 0) then
    Result := LDot / Sqrt(LA * LB);
end;

function VectorToBytes(const V: TVector): TBytes;
begin
  SetLength(Result, Length(V) * SizeOf(Single));
  if Length(V) > 0 then
    Move(V[0], Result[0], Length(Result));
end;

function BytesToVector(const B: TBytes): TVector;
begin
  SetLength(Result, Length(B) div SizeOf(Single));
  if Length(Result) > 0 then
    Move(B[0], Result[0], Length(Result) * SizeOf(Single));
end;

{ TRetrieval }

constructor TRetrieval.Create(const ACacheDir, AEmbedMode, AEmbedModel, ARerankMode, ARerankModel: string);
begin
  inherited Create;
  FCacheDir := ACacheDir;
  FTier := 'off';
  if AEmbedMode <> 'off' then
  begin
    if (AEmbedMode = 'static') or not TryLoadOnnxEmbedder(AEmbedModel) then
    begin
      FTier := 'static';
      FEmbedId := StaticEmbedderId;
    end;
  end;
  if (ARerankMode <> 'off') and (ARerankModel <> '') then
    TryLoadReranker(ARerankModel);
end;

destructor TRetrieval.Destroy;
begin
  FEmb.Free;
  FTok.Free;
  FReranker.Free;
  inherited Destroy;
end;

function TRetrieval.TryLoadOnnxEmbedder(const AModel: string): Boolean;
var
  LDir, LErr: string;
begin
  Result := False;
  if not FindModelSpec(AModel, FSpec) then
  begin
    FNote := 'unknown embed model ' + AModel + ' (have: ' + ModelKeys + ')';
    Exit;
  end;
  LDir := ModelDir(FCacheDir, FSpec.SubDir);
  if not ArtifactsPresent(LDir) then
  begin
    FNote := 'embedding model not provisioned (run memagent --provision)';
    Exit;
  end;
  if not LoadRuntime(FCacheDir, LErr) then
  begin
    FNote := 'ONNX Runtime not loadable: ' + LErr;
    Exit;
  end;
  try
    FTok := TBertTokenizer.Create(IncludeTrailingPathDelimiter(LDir) + VOCAB_FILE, FSpec.DoLowerCase);
    FEmb := TEmbedder.Create(IncludeTrailingPathDelimiter(LDir) + MODEL_FILE);
    FEmb.Load;
  except
    on E: Exception do
    begin
      FNote := 'embedder failed to load: ' + E.Message;
      FreeAndNil(FEmb);
      FreeAndNil(FTok);
      Exit;
    end;
  end;
  FTier := 'onnx';
  FEmbedId := FSpec.Key + '@' + IntToStr(FSpec.Dim);
  Result := True;
end;

function TRetrieval.TryLoadReranker(const AModel: string): Boolean;
var
  LSpec: TModelSpec;
  LDir, LErr: string;
begin
  Result := False;
  if not FindRerankerSpec(AModel, LSpec) then
    Exit;
  LDir := ModelDir(FCacheDir, LSpec.SubDir);
  if not ArtifactsPresent(LDir) or not LoadRuntime(FCacheDir, LErr) then
    Exit;
  try
    FReranker := TReranker.Create(IncludeTrailingPathDelimiter(LDir) + MODEL_FILE,
      IncludeTrailingPathDelimiter(LDir) + VOCAB_FILE, LSpec.DoLowerCase, LSpec.NeedsTokenTypeIds,
      256, RerankerIsSentencePiece(LSpec.Key));
    { warm up: loads the ONNX session now rather than on the first user turn }
    FReranker.Score('warm up', 'warm up');
  except
    on E: Exception do
    begin
      FNote := FNote + IfThen(FNote = '', '', '; ') + 'reranker failed to load: ' + E.Message;
      FreeAndNil(FReranker);
      Exit;
    end;
  end;
  FRerankKey := LSpec.Key;
  Result := True;
end;

function TRetrieval.Embed(const AText: string): TVector;
var
  LIds: specialize TArray<Int64>;
begin
  Result := nil;
  if Trim(AText) = '' then
    Exit;
  if FTier = 'onnx' then
  begin
    LIds := FTok.Encode(UTF8Decode(AText), 256);
    Result := FEmb.Embed(LIds, FSpec.Pooling, FSpec.NeedsTokenTypeIds, True);
  end
  else if FTier = 'static' then
    Result := StaticEmbed(AText);
end;

function TRetrieval.RerankScores(const AQuery: string; const ADocs: array of string): TScoreArray;
var
  I: Integer;
begin
  SetLength(Result, Length(ADocs));
  for I := 0 to High(ADocs) do
    if FReranker <> nil then
      Result[I] := FReranker.Score(AQuery, ADocs[I])
    else
      Result[I] := 1.0;
end;

function TRetrieval.HasReranker: Boolean;
begin
  Result := FReranker <> nil;
end;

function TRetrieval.CanDedup: Boolean;
begin
  Result := FTier = 'onnx';
end;

function TRetrieval.Describe: string;
begin
  case FTier of
    'onnx': Result := 'embedder ' + FEmbedId + ' (onnx)';
    'static': Result := 'embedder ' + FEmbedId + ' (lexical fallback)';
  else
    Result := 'embedder off';
  end;
  if FReranker <> nil then
    Result := Result + ', reranker ' + FRerankKey
  else
    Result := Result + ', no reranker';
  if FNote <> '' then
    Result := Result + ' [' + FNote + ']';
end;

function ProvisionRetrieval(const ACacheDir, AEmbedModel, ARerankModel: string): Boolean;
var
  LSpec: TModelSpec;
  LMsg, LErr: string;
  LDl: TModelDownloader;
begin
  Result := False;
  ForceDirectories(ACacheDir);
  {$IFDEF MSWINDOWS}
  try
    EnsureOnnxRuntime(ACacheDir, True, True);
  except
    on E: Exception do
    begin
      WriteLn(StdErr, '[provision] ONNX Runtime: ', E.Message);
      Exit;
    end;
  end;
  {$ELSE}
  if not EnsurePosixOrt(ACacheDir, LMsg) then
  begin
    WriteLn(StdErr, '[provision] ONNX Runtime: ', LMsg);
    Exit;
  end;
  WriteLn('[provision] ONNX Runtime: ', LMsg);
  {$ENDIF}
  if not LoadRuntime(ACacheDir, LErr) then
  begin
    WriteLn(StdErr, '[provision] ONNX Runtime does not load: ', LErr);
    Exit;
  end;

  if not FindModelSpec(AEmbedModel, LSpec) then
  begin
    WriteLn(StdErr, '[provision] unknown embed model ', AEmbedModel, ' (have: ', ModelKeys, ')');
    Exit;
  end;
  LDl := TModelDownloader.Create(LSpec, ModelDir(ACacheDir, LSpec.SubDir), True);
  try
    LDl.EnsureFiles;
  finally
    LDl.Free;
  end;
  WriteLn('[provision] embedder: ', LSpec.DisplayName);

  if ARerankModel <> '' then
  begin
    if not FindRerankerSpec(ARerankModel, LSpec) then
    begin
      WriteLn(StdErr, '[provision] unknown rerank model ', ARerankModel, ' (have: ', RerankerKeys, ')');
      Exit;
    end;
    LDl := TModelDownloader.Create(LSpec, ModelDir(ACacheDir, LSpec.SubDir), True);
    try
      LDl.EnsureFiles;
    finally
      LDl.Free;
    end;
    WriteLn('[provision] reranker: ', LSpec.DisplayName);
  end;
  Result := True;
end;

end.
