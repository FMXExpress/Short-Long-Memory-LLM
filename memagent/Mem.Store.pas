unit Mem.Store;

(* memory.db: the mid-term memory and the bookkeeping for the long-term one.

   messages  every turn, FTS5-indexed for episodic recall
   facts     durable facts; baked_in names the adapter that has learned the
             fact, and a baked fact is no longer injected into the prompt
   summary   running summary of the session beyond the recent window
   adapters  every consolidation result, for /adapter and /rollback
   meta      small key/value state (active adapter, last consolidation) *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, DateUtils, fpjson, Mem.Db;

type
  TFactRec = record
    Id: Int64;
    Text, Subject, Value, Kind, BakedIn: string;
    Confidence: Double;
  end;
  TFactArray = array of TFactRec;

  TMsgRec = record
    Id: Int64;
    Role, Content: string;
  end;
  TMsgArray = array of TMsgRec;

  TAddFactResult = (afAdded, afDuplicate, afSuperseded);

  TMemStore = class
  private
    FDb: TSqlDb;
    procedure Migrate;
    function FtsQuery(const AText: string): string;
  public
    constructor Create(const APath: string);
    destructor Destroy; override;
    function AddMessage(const ASession, ARole, AContent: string): Int64;
    function RecentMessages(const ASession: string; ALimit: Integer): TMsgArray;
    function SearchMessages(const AQuery: string; ABeforeId: Int64; ALimit: Integer): TMsgArray;
    function AddFact(AFact: TJSONObject; ASourceMsg: Int64; out AId: Int64): TAddFactResult;
    function ActiveFacts: TFactArray;
    function SearchFacts(const AQuery: string; ALimit: Integer; AIncludeBaked: Boolean): TFactArray;
    function ForgetFact(AId: Int64): Boolean;
    function UnbakedCount: Integer;
    procedure ApplyManifest(const AAdapterId: string; APassed: TJSONArray);
    procedure ClearBaked;
    procedure RecordAdapter(const AId, APath, AStatus: string; ANFacts, NPassed: Integer;
      ARecall, APplRatio: Double);
    procedure SetAdapterStatus(const AId, AStatus: string);
    procedure RetireAdapter(const AId: string);
    function PreviousAdapter(const ACurrentId: string; out APath: string): string;
    function AdapterHistory(ALimit: Integer): TStringList;
    function GetSummary(const ASession: string; out AUptoMsg: Int64): string;
    procedure SetSummary(const ASession, AText: string; AUptoMsg: Int64);
    function GetMeta(const AKey: string; const ADefault: string = ''): string;
    procedure SetMeta(const AKey, AValue: string);
  end;

function NowUnix: Int64;

implementation

const
  StopWords: array[0..47] of string = (
    'the', 'and', 'you', 'your', 'what', 'whats', 'what''s', 'who', 'how', 'why', 'when', 'where',
    'are', 'was', 'were', 'is', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'have',
    'has', 'had', 'for', 'with', 'about', 'this', 'that', 'there', 'tell', 'know', 'remember',
    'please', 'me', 'my', 'mine', 'our', 'its', 'it''s', 'not', 'any', 'some', 'from', 'into',
    'remind');

function NowUnix: Int64;
begin
  Result := DateTimeToUnix(Now, False);
end;

constructor TMemStore.Create(const APath: string);
begin
  inherited Create;
  FDb := TSqlDb.Create(APath);
  Migrate;
end;

destructor TMemStore.Destroy;
begin
  FDb.Free;
  inherited Destroy;
end;

procedure TMemStore.Migrate;
begin
  FDb.ExecScript(
    'CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session TEXT NOT NULL,' +
    ' role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL);' +
    'CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);' +
    'CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, text TEXT NOT NULL,' +
    ' subject TEXT NOT NULL, value TEXT NOT NULL, questions TEXT NOT NULL DEFAULT ''[]'',' +
    ' kind TEXT NOT NULL DEFAULT ''user'', confidence REAL NOT NULL DEFAULT 0.5,' +
    ' source_msg INTEGER, created_at INTEGER NOT NULL, superseded INTEGER NOT NULL DEFAULT 0,' +
    ' forgotten INTEGER NOT NULL DEFAULT 0, baked_in TEXT NOT NULL DEFAULT '''');' +
    'CREATE INDEX IF NOT EXISTS facts_subject ON facts(lower(subject));' +
    'CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, subject);' +
    'CREATE TABLE IF NOT EXISTS summary(session TEXT PRIMARY KEY, text TEXT NOT NULL,' +
    ' upto_msg INTEGER NOT NULL);' +
    'CREATE TABLE IF NOT EXISTS adapters(id TEXT PRIMARY KEY, path TEXT NOT NULL,' +
    ' created_at INTEGER NOT NULL, n_facts INTEGER, n_passed INTEGER, recall REAL,' +
    ' ppl_ratio REAL, status TEXT NOT NULL);' +
    'CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);');
end;

function TMemStore.FtsQuery(const AText: string): string;
var
  LWords: TStringList;
  LWord, S: string;
  I: Integer;
  LStop: Boolean;
begin
  (* Free text -> '"w1" OR "w2" ...': quoting each word keeps FTS5 syntax
     characters in user input from turning into query operators. *)
  Result := '';
  LWords := TStringList.Create;
  try
    S := LowerCase(AText);
    for I := 1 to Length(S) do
      if not (S[I] in ['a'..'z', '0'..'9', '''', #128..#255]) then
        S[I] := ' ';
    LWords.Delimiter := ' ';
    LWords.StrictDelimiter := True;
    LWords.DelimitedText := S;
    for LWord in LWords do
    begin
      S := StringReplace(LWord, '''s', '', []);
      S := StringReplace(S, '''', '', [rfReplaceAll]);
      if Length(S) < 3 then
        Continue;
      LStop := False;
      for I := Low(StopWords) to High(StopWords) do
        if S = StopWords[I] then
          LStop := True;
      if LStop or (Pos('"' + S + '"', Result) > 0) then
        Continue;
      if Result <> '' then
        Result := Result + ' OR ';
      Result := Result + '"' + S + '"';
    end;
  finally
    LWords.Free;
  end;
end;

function TMemStore.AddMessage(const ASession, ARole, AContent: string): Int64;
begin
  FDb.Exec('INSERT INTO messages(session, role, content, created_at) VALUES(?,?,?,?)',
    [ASession, ARole, AContent, NowUnix]);
  Result := FDb.LastId;
  FDb.Exec('INSERT INTO messages_fts(rowid, content) VALUES(?,?)', [Result, AContent]);
end;

function TMemStore.RecentMessages(const ASession: string; ALimit: Integer): TMsgArray;
var
  Q: TSqlStmt;
  N: Integer;
begin
  Result := nil;
  Q := FDb.Query('SELECT id, role, content FROM (SELECT * FROM messages WHERE session = ?' +
    ' ORDER BY id DESC LIMIT ?) ORDER BY id', [ASession, ALimit]);
  try
    N := 0;
    while Q.Step do
    begin
      SetLength(Result, N + 1);
      Result[N].Id := Q.Int(0);
      Result[N].Role := Q.Str(1);
      Result[N].Content := Q.Str(2);
      Inc(N);
    end;
  finally
    Q.Free;
  end;
end;

function TMemStore.SearchMessages(const AQuery: string; ABeforeId: Int64; ALimit: Integer): TMsgArray;
var
  Q: TSqlStmt;
  LFts: string;
  N: Integer;
begin
  (* Episodic recall. A message that was the source of a fact the current
     adapter has learned is skipped: that knowledge now lives in the weights. *)
  Result := nil;
  LFts := FtsQuery(AQuery);
  if LFts = '' then
    Exit;
  Q := FDb.Query('SELECT m.id, m.role, m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid' +
    ' WHERE messages_fts MATCH ? AND m.id < ? AND m.role = ''user''' +
    ' AND NOT EXISTS (SELECT 1 FROM facts x WHERE x.source_msg = m.id AND x.baked_in <> '''')' +
    ' ORDER BY bm25(messages_fts) LIMIT ?', [LFts, ABeforeId, ALimit]);
  try
    N := 0;
    while Q.Step do
    begin
      SetLength(Result, N + 1);
      Result[N].Id := Q.Int(0);
      Result[N].Role := Q.Str(1);
      Result[N].Content := Q.Str(2);
      Inc(N);
    end;
  finally
    Q.Free;
  end;
end;

function TMemStore.AddFact(AFact: TJSONObject; ASourceMsg: Int64; out AId: Int64): TAddFactResult;
var
  LSubject, LValue, LOldValue, LQuestions: string;
  LOldId: Int64;
  Q: TSqlStmt;
begin
  LSubject := AFact.Get('subject', '');
  LValue := AFact.Get('value', '');
  LOldId := 0;
  LOldValue := '';
  Q := FDb.Query('SELECT id, value FROM facts WHERE lower(subject) = lower(?) AND superseded = 0' +
    ' AND forgotten = 0 ORDER BY id DESC LIMIT 1', [LSubject]);
  try
    if Q.Step then
    begin
      LOldId := Q.Int(0);
      LOldValue := Q.Str(1);
    end;
  finally
    Q.Free;
  end;
  if (LOldId <> 0) and SameText(LOldValue, LValue) then
  begin
    AId := LOldId;
    Exit(afDuplicate);
  end;
  Result := afAdded;
  if LOldId <> 0 then
  begin
    { same subject, new value: the newer statement wins }
    FDb.Exec('UPDATE facts SET superseded = 1 WHERE id = ?', [LOldId]);
    Result := afSuperseded;
  end;
  if AFact.Find('questions') <> nil then
    LQuestions := AFact.Arrays['questions'].AsJSON
  else
    LQuestions := '[]';
  FDb.Exec('INSERT INTO facts(text, subject, value, questions, kind, confidence, source_msg, created_at)' +
    ' VALUES(?,?,?,?,?,?,?,?)', [AFact.Get('text', ''), LSubject, LValue, LQuestions,
    AFact.Get('kind', 'user'), Extended(AFact.Get('confidence', 0.5)), ASourceMsg, NowUnix]);
  AId := FDb.LastId;
  FDb.Exec('INSERT INTO facts_fts(rowid, text, subject) VALUES(?,?,?)',
    [AId, AFact.Get('text', ''), LSubject]);
end;

function ReadFacts(Q: TSqlStmt): TFactArray;
var
  N: Integer;
begin
  Result := nil;
  N := 0;
  try
    while Q.Step do
    begin
      SetLength(Result, N + 1);
      with Result[N] do
      begin
        Id := Q.Int(0);
        Text := Q.Str(1);
        Subject := Q.Str(2);
        Value := Q.Str(3);
        Kind := Q.Str(4);
        Confidence := Q.Float(5);
        BakedIn := Q.Str(6);
      end;
      Inc(N);
    end;
  finally
    Q.Free;
  end;
end;

function TMemStore.ActiveFacts: TFactArray;
begin
  Result := ReadFacts(FDb.Query('SELECT id, text, subject, value, kind, confidence, baked_in FROM facts' +
    ' WHERE superseded = 0 AND forgotten = 0 ORDER BY id', []));
end;

function TMemStore.SearchFacts(const AQuery: string; ALimit: Integer; AIncludeBaked: Boolean): TFactArray;
var
  LFts: string;
begin
  Result := nil;
  LFts := FtsQuery(AQuery);
  if LFts = '' then
    Exit;
  Result := ReadFacts(FDb.Query('SELECT x.id, x.text, x.subject, x.value, x.kind, x.confidence, x.baked_in' +
    ' FROM facts_fts f JOIN facts x ON x.id = f.rowid WHERE facts_fts MATCH ?' +
    ' AND x.superseded = 0 AND x.forgotten = 0 AND (? OR x.baked_in = '''')' +
    ' ORDER BY bm25(facts_fts) LIMIT ?', [LFts, AIncludeBaked, ALimit]));
end;

function TMemStore.ForgetFact(AId: Int64): Boolean;
begin
  Result := FDb.ScalarInt('SELECT count(*) FROM facts WHERE id = ? AND forgotten = 0', [AId]) > 0;
  if Result then
    FDb.Exec('UPDATE facts SET forgotten = 1 WHERE id = ?', [AId]);
end;

function TMemStore.UnbakedCount: Integer;
begin
  Result := FDb.ScalarInt('SELECT count(*) FROM facts WHERE superseded = 0 AND forgotten = 0' +
    ' AND baked_in = ''''', []);
end;

procedure TMemStore.ApplyManifest(const AAdapterId: string; APassed: TJSONArray);
var
  I: Integer;
begin
  (* Every adapter is rebuilt from base on the full fact set, so its probe
     results replace the previous bookkeeping rather than adding to it. *)
  ClearBaked;
  for I := 0 to APassed.Count - 1 do
    FDb.Exec('UPDATE facts SET baked_in = ? WHERE id = ? AND superseded = 0 AND forgotten = 0',
      [AAdapterId, APassed.Items[I].AsInt64]);
end;

procedure TMemStore.ClearBaked;
begin
  FDb.Exec('UPDATE facts SET baked_in = ''''', []);
end;

procedure TMemStore.RecordAdapter(const AId, APath, AStatus: string; ANFacts, NPassed: Integer;
  ARecall, APplRatio: Double);
begin
  FDb.Exec('INSERT OR REPLACE INTO adapters(id, path, created_at, n_facts, n_passed, recall, ppl_ratio, status)' +
    ' VALUES(?,?,?,?,?,?,?,?)', [AId, APath, NowUnix, ANFacts, NPassed, Extended(ARecall),
    Extended(APplRatio), AStatus]);
end;

procedure TMemStore.SetAdapterStatus(const AId, AStatus: string);
begin
  FDb.Exec('UPDATE adapters SET status = ? WHERE id = ?', [AStatus, AId]);
end;

procedure TMemStore.RetireAdapter(const AId: string);
begin
  { leaves a 'rolledback' status alone, so rollback history stays visible }
  FDb.Exec('UPDATE adapters SET status = ''retired'' WHERE id = ? AND status = ''active''', [AId]);
end;

function TMemStore.PreviousAdapter(const ACurrentId: string; out APath: string): string;
var
  Q: TSqlStmt;
begin
  Result := '';
  APath := '';
  Q := FDb.Query('SELECT id, path FROM adapters WHERE status = ''retired'' AND id < ?' +
    ' ORDER BY id DESC LIMIT 1', [ACurrentId]);
  try
    if Q.Step then
    begin
      Result := Q.Str(0);
      APath := Q.Str(1);
    end;
  finally
    Q.Free;
  end;
end;

function TMemStore.AdapterHistory(ALimit: Integer): TStringList;
var
  Q: TSqlStmt;
begin
  Result := TStringList.Create;
  Q := FDb.Query('SELECT id, status, n_passed, n_facts, recall, ppl_ratio FROM adapters' +
    ' ORDER BY id DESC LIMIT ?', [ALimit]);
  try
    while Q.Step do
      Result.Add(Format('%-22s %-9s facts %d/%d  recall %.2f  ppl x%.2f',
        [Q.Str(0), Q.Str(1), Q.Int(2), Q.Int(3), Q.Float(4), Q.Float(5)]));
  finally
    Q.Free;
  end;
end;

function TMemStore.GetSummary(const ASession: string; out AUptoMsg: Int64): string;
var
  Q: TSqlStmt;
begin
  Result := '';
  AUptoMsg := 0;
  Q := FDb.Query('SELECT text, upto_msg FROM summary WHERE session = ?', [ASession]);
  try
    if Q.Step then
    begin
      Result := Q.Str(0);
      AUptoMsg := Q.Int(1);
    end;
  finally
    Q.Free;
  end;
end;

procedure TMemStore.SetSummary(const ASession, AText: string; AUptoMsg: Int64);
begin
  FDb.Exec('INSERT OR REPLACE INTO summary(session, text, upto_msg) VALUES(?,?,?)',
    [ASession, AText, AUptoMsg]);
end;

function TMemStore.GetMeta(const AKey: string; const ADefault: string): string;
begin
  Result := FDb.ScalarStr('SELECT value FROM meta WHERE key = ?', [AKey], ADefault);
end;

procedure TMemStore.SetMeta(const AKey, AValue: string);
begin
  FDb.Exec('INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)', [AKey, AValue]);
end;

end.
