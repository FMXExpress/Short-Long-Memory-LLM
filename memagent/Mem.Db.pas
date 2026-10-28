unit Mem.Db;

(* Minimal SQLite wrapper over the C API (sqlite3dyn). Statements run in
   autocommit mode, so the Python trainer can read facts at any moment
   without waiting on a transaction memagent forgot to close. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, ctypes, sqlite3dyn;

type
  ESqlite = class(Exception);

  TSqlStmt = class
  private
    FDb: psqlite3;
    FStmt: psqlite3_stmt;
  public
    constructor Create(ADb: psqlite3; const ASql: string; const AArgs: array of const);
    destructor Destroy; override;
    function Step: Boolean;
    function Int(ACol: Integer): Int64;
    function Str(ACol: Integer): string;
    function Float(ACol: Integer): Double;
  end;

  TSqlDb = class
  private
    FDb: psqlite3;
  public
    constructor Create(const APath: string);
    destructor Destroy; override;
    procedure Exec(const ASql: string; const AArgs: array of const);
    procedure ExecScript(const ASql: string);
    function Query(const ASql: string; const AArgs: array of const): TSqlStmt;
    function ScalarInt(const ASql: string; const AArgs: array of const; ADefault: Int64 = 0): Int64;
    function ScalarStr(const ASql: string; const AArgs: array of const; const ADefault: string = ''): string;
    function LastId: Int64;
  end;

implementation

const
  LibNames: array[0..2] of string = ('libsqlite3.so.0', 'libsqlite3.so', 'sqlite3.dll');

var
  GLoaded: Boolean = False;

procedure EnsureLib;
var
  LName: string;
begin
  if GLoaded then
    Exit;
  for LName in LibNames do
    try
      InitializeSqlite(UnicodeString(LName));
      GLoaded := True;
      Exit;
    except
      on Exception do ;
    end;
  raise ESqlite.Create('could not load the SQLite library (tried libsqlite3.so.0, libsqlite3.so, sqlite3.dll)');
end;

procedure Check(ADb: psqlite3; ARc: cint; const AWhat: string);
begin
  if not (ARc in [SQLITE_OK, SQLITE_ROW, SQLITE_DONE]) then
    raise ESqlite.CreateFmt('%s: %s (%d)', [AWhat, string(sqlite3_errmsg(ADb)), ARc]);
end;

{ TSqlStmt }

constructor TSqlStmt.Create(ADb: psqlite3; const ASql: string; const AArgs: array of const);
var
  I: Integer;
  S: string;
begin
  FDb := ADb;
  Check(FDb, sqlite3_prepare_v2(FDb, PChar(ASql), -1, @FStmt, nil), 'prepare "' + ASql + '"');
  for I := 0 to High(AArgs) do
    case AArgs[I].VType of
      vtInteger: sqlite3_bind_int64(FStmt, I + 1, AArgs[I].VInteger);
      vtInt64: sqlite3_bind_int64(FStmt, I + 1, AArgs[I].VInt64^);
      vtBoolean: sqlite3_bind_int64(FStmt, I + 1, Ord(AArgs[I].VBoolean));
      vtExtended: sqlite3_bind_double(FStmt, I + 1, AArgs[I].VExtended^);
      vtPointer: sqlite3_bind_null(FStmt, I + 1);
    else
      begin
        case AArgs[I].VType of
          vtAnsiString: S := AnsiString(AArgs[I].VAnsiString);
          vtUnicodeString: S := UTF8Encode(UnicodeString(AArgs[I].VUnicodeString));
          vtString: S := AArgs[I].VString^;
          vtChar: S := AArgs[I].VChar;
        else
          raise ESqlite.CreateFmt('unsupported bind type %d', [AArgs[I].VType]);
        end;
        sqlite3_bind_text(FStmt, I + 1, PChar(S), Length(S), sqlite3_destructor_type(SQLITE_TRANSIENT));
      end;
    end;
end;

destructor TSqlStmt.Destroy;
begin
  if FStmt <> nil then
    sqlite3_finalize(FStmt);
  inherited Destroy;
end;

function TSqlStmt.Step: Boolean;
var
  LRc: cint;
begin
  LRc := sqlite3_step(FStmt);
  Check(FDb, LRc, 'step');
  Result := LRc = SQLITE_ROW;
end;

function TSqlStmt.Int(ACol: Integer): Int64;
begin
  Result := sqlite3_column_int64(FStmt, ACol);
end;

function TSqlStmt.Str(ACol: Integer): string;
var
  P: PChar;
begin
  P := PChar(sqlite3_column_text(FStmt, ACol));
  if P = nil then
    Result := ''
  else
    SetString(Result, P, sqlite3_column_bytes(FStmt, ACol));
end;

function TSqlStmt.Float(ACol: Integer): Double;
begin
  Result := sqlite3_column_double(FStmt, ACol);
end;

{ TSqlDb }

constructor TSqlDb.Create(const APath: string);
begin
  EnsureLib;
  Check(nil, sqlite3_open(PChar(APath), @FDb), 'open ' + APath);
  sqlite3_busy_timeout(FDb, 5000);
  ExecScript('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;');
end;

destructor TSqlDb.Destroy;
begin
  if FDb <> nil then
    sqlite3_close(FDb);
  inherited Destroy;
end;

procedure TSqlDb.Exec(const ASql: string; const AArgs: array of const);
var
  LStmt: TSqlStmt;
begin
  LStmt := TSqlStmt.Create(FDb, ASql, AArgs);
  try
    LStmt.Step;
  finally
    LStmt.Free;
  end;
end;

procedure TSqlDb.ExecScript(const ASql: string);
var
  LErr: PChar;
begin
  LErr := nil;
  if sqlite3_exec(FDb, PChar(ASql), nil, nil, @LErr) <> SQLITE_OK then
    raise ESqlite.Create('exec: ' + string(LErr));
end;

function TSqlDb.Query(const ASql: string; const AArgs: array of const): TSqlStmt;
begin
  Result := TSqlStmt.Create(FDb, ASql, AArgs);
end;

function TSqlDb.ScalarInt(const ASql: string; const AArgs: array of const; ADefault: Int64): Int64;
var
  LStmt: TSqlStmt;
begin
  LStmt := Query(ASql, AArgs);
  try
    if LStmt.Step then
      Result := LStmt.Int(0)
    else
      Result := ADefault;
  finally
    LStmt.Free;
  end;
end;

function TSqlDb.ScalarStr(const ASql: string; const AArgs: array of const; const ADefault: string): string;
var
  LStmt: TSqlStmt;
begin
  LStmt := Query(ASql, AArgs);
  try
    if LStmt.Step then
      Result := LStmt.Str(0)
    else
      Result := ADefault;
  finally
    LStmt.Free;
  end;
end;

function TSqlDb.LastId: Int64;
begin
  Result := sqlite3_last_insert_rowid(FDb);
end;

end.
