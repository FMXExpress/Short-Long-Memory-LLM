unit Ban.Tail;

(* tail -F for a set of log files. Patterns may hold wildcards in the file
   name (/var/log/apache2/*access.log); they are expanded again every few
   seconds, so a vhost log that appears later is picked up from its start.
   Rotation is handled both ways logrotate does it:
     - rename + create: the path now names another file (inode on Unix);
       the old file is drained to its end first, then the new one is read
       from the start;
     - copytruncate: the file got shorter than our position; start over. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes {$IFDEF UNIX}, BaseUnix{$ENDIF};

type
  TTailFile = class
  public
    Path: string;
    Handle: THandle;
    Position: Int64;
    Ino: QWord;
    Partial: string;
    Seen: Boolean;      { still matched by a pattern on the last expansion }
    constructor Create(const APath: string);
    destructor Destroy; override;
    function Open(AFromEnd: Boolean): Boolean;
    procedure Close;
    { appends whole lines to ALines; False when nothing new }
    function ReadNew(ALines: TStrings; AMaxBytes: Integer): Boolean;
    procedure CheckRotation;
  end;

  TLogTailer = class
  private
    FPatterns: TStringList;
    FFiles: TStringList;      { path -> TTailFile }
    FFromStart: Boolean;
    FLastScan: QWord;
    FStarted: Boolean;
    procedure Rescan;
  public
    MaxLineLen: Integer;
    constructor Create(APatterns: TStrings; AFromStart: Boolean);
    destructor Destroy; override;
    { new lines from every file; False when there were none }
    function Poll(ALines: TStrings): Boolean;
    function FileCount: Integer;
    function FileNames: string;
  end;

implementation

const
  CChunk = 65536;
  CRescanMs = 5000;
  CMaxLine = 16384;

{$IFDEF UNIX}
function FileIno(const APath: string; out AIno: QWord; out ASize: Int64): Boolean;
var
  LSt: Stat;
begin
  Result := FpStat(APath, LSt) = 0;
  if Result then
  begin
    AIno := LSt.st_ino;
    ASize := LSt.st_size;
  end;
end;

function HandleIno(AHandle: THandle): QWord;
var
  LSt: Stat;
begin
  if FpFStat(AHandle, LSt) = 0 then
    Result := LSt.st_ino
  else
    Result := 0;
end;
{$ELSE}
function FileIno(const APath: string; out AIno: QWord; out ASize: Int64): Boolean;
var
  LRec: TSearchRec;
begin
  AIno := 0;
  Result := FindFirst(APath, faAnyFile, LRec) = 0;
  if Result then
    ASize := LRec.Size;
  FindClose(LRec);
end;

function HandleIno(AHandle: THandle): QWord;
begin
  Result := 0;
end;
{$ENDIF}

{ TTailFile }

constructor TTailFile.Create(const APath: string);
begin
  inherited Create;
  Path := APath;
  Handle := feInvalidHandle;
end;

destructor TTailFile.Destroy;
begin
  Close;
  inherited Destroy;
end;

function TTailFile.Open(AFromEnd: Boolean): Boolean;
begin
  Close;
  Handle := FileOpen(Path, fmOpenRead or fmShareDenyNone);
  if Handle = feInvalidHandle then
    Exit(False);
  Ino := HandleIno(Handle);
  if AFromEnd then
    Position := FileSeek(Handle, Int64(0), fsFromEnd)
  else
    Position := 0;
  Partial := '';
  Result := True;
end;

procedure TTailFile.Close;
begin
  if Handle <> feInvalidHandle then
    FileClose(Handle);
  Handle := feInvalidHandle;
end;

function TTailFile.ReadNew(ALines: TStrings; AMaxBytes: Integer): Boolean;
var
  LBuf: array[0..CChunk - 1] of Char;
  LGot, LTotal, I, LStart: Integer;
  LLine: string;
begin
  Result := False;
  if Handle = feInvalidHandle then
    Exit;
  LTotal := 0;
  FileSeek(Handle, Position, fsFromBeginning);
  repeat
    LGot := FileRead(Handle, LBuf[0], CChunk);
    if LGot <= 0 then
      Break;
    Inc(Position, LGot);
    Inc(LTotal, LGot);
    LStart := 0;
    for I := 0 to LGot - 1 do
      if LBuf[I] = #10 then
      begin
        SetString(LLine, PChar(@LBuf[LStart]), I - LStart);
        LLine := Partial + LLine;
        Partial := '';
        if (LLine <> '') and (LLine[Length(LLine)] = #13) then
          SetLength(LLine, Length(LLine) - 1);
        if Length(LLine) > CMaxLine then
          SetLength(LLine, CMaxLine);
        if LLine <> '' then
          ALines.Add(LLine);
        LStart := I + 1;
        Result := True;
      end;
    if LStart < LGot then
    begin
      SetString(LLine, PChar(@LBuf[LStart]), LGot - LStart);
      if Length(Partial) < CMaxLine then
        Partial := Partial + LLine;
    end;
  until LTotal >= AMaxBytes;
end;

procedure TTailFile.CheckRotation;
var
  LIno: QWord;
  LSize: Int64;
begin
  if not FileIno(Path, LIno, LSize) then
    Exit;   { between rename and create: keep reading the old file }
  if Handle = feInvalidHandle then
  begin
    Open(False);
    Exit;
  end;
  { renamed away (another inode): TLogTailer.Poll drains it and reopens }
  if (LIno <> 0) and (LIno <> Ino) then
    Exit;
  if LSize < Position then
  begin
    Position := 0;
    Partial := '';
  end;
end;

{ TLogTailer }

constructor TLogTailer.Create(APatterns: TStrings; AFromStart: Boolean);
begin
  inherited Create;
  FPatterns := TStringList.Create;
  FPatterns.Assign(APatterns);
  FFiles := TStringList.Create;
  FFiles.Sorted := True;
  FFiles.OwnsObjects := True;
  FFromStart := AFromStart;
  MaxLineLen := CMaxLine;
  Rescan;
  FStarted := True;
end;

destructor TLogTailer.Destroy;
begin
  FFiles.Free;
  FPatterns.Free;
  inherited Destroy;
end;

procedure TLogTailer.Rescan;
var
  LPattern, LDir, LPath: string;
  LRec: TSearchRec;
  LFile: TTailFile;
  I: Integer;
begin
  for I := 0 to FFiles.Count - 1 do
    TTailFile(FFiles.Objects[I]).Seen := False;
  for LPattern in FPatterns do
  begin
    if (Pos('*', LPattern) = 0) and (Pos('?', LPattern) = 0) then
    begin
      { a plain path is followed even before it exists }
      I := FFiles.IndexOf(LPattern);
      if I < 0 then
      begin
        LFile := TTailFile.Create(LPattern);
        FFiles.AddObject(LPattern, LFile);
        LFile.Open(not FFromStart and not FStarted);
      end
      else
        LFile := TTailFile(FFiles.Objects[I]);
      LFile.Seen := True;
      Continue;
    end;
    LDir := ExtractFilePath(LPattern);
    if FindFirst(LPattern, faAnyFile and not faDirectory, LRec) = 0 then
    try
      repeat
        LPath := LDir + LRec.Name;
        I := FFiles.IndexOf(LPath);
        if I < 0 then
        begin
          LFile := TTailFile.Create(LPath);
          FFiles.AddObject(LPath, LFile);
          { files there at startup start at their end; later ones at 0 }
          LFile.Open(not FFromStart and not FStarted);
        end
        else
          LFile := TTailFile(FFiles.Objects[I]);
        LFile.Seen := True;
      until FindNext(LRec) <> 0;
    finally
      FindClose(LRec);
    end;
  end;
  for I := FFiles.Count - 1 downto 0 do
    if not TTailFile(FFiles.Objects[I]).Seen then
      FFiles.Delete(I);
  FLastScan := GetTickCount64;
end;

function TLogTailer.Poll(ALines: TStrings): Boolean;
var
  I: Integer;
  LFile: TTailFile;
  LIno: QWord;
  LSize: Int64;
begin
  if GetTickCount64 - FLastScan >= CRescanMs then
    Rescan;
  Result := False;
  for I := 0 to FFiles.Count - 1 do
  begin
    LFile := TTailFile(FFiles.Objects[I]);
    if LFile.ReadNew(ALines, 4 * CChunk) then
      Result := True;
    LFile.CheckRotation;
    if (LFile.Handle <> feInvalidHandle) and FileIno(LFile.Path, LIno, LSize) and
       (LIno <> 0) and (LIno <> LFile.Ino) then
    begin
      { rotated by rename: finish the old file, then follow the new one }
      while LFile.ReadNew(ALines, MaxInt) do
        Result := True;
      LFile.Open(False);
      if LFile.ReadNew(ALines, 4 * CChunk) then
        Result := True;
    end;
  end;
end;

function TLogTailer.FileCount: Integer;
begin
  Result := FFiles.Count;
end;

function TLogTailer.FileNames: string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to FFiles.Count - 1 do
  begin
    if Result <> '' then
      Result := Result + ', ';
    Result := Result + FFiles[I];
  end;
  if Result = '' then
    Result := '(none yet)';
end;

end.
