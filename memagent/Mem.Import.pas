unit Mem.Import;

(* Conversations from other chat apps and coding agents, read into one shape so the
   agent can take what the user said in them into memory (/import).

   Formats (each sniffed from the file, not its name):
     chatgpt      ChatGPT "Export data": conversations.json, an array of conversations
                  with a "mapping" tree; the transcript is the parent chain from
                  current_node back to the root, so edited branches fall away.
     claude-ai    claude.ai "Export data": conversations.json, an array of
                  conversations with "chat_messages" (sender human / assistant).
     claude-code  Claude Code: one .jsonl per session under ~/.claude/projects/.
                  Subagent lines (isSidechain), meta lines, tool calls and results
                  are skipped.
     codex        Codex CLI: rollout-*.jsonl under ~/.codex/sessions/ (response_item
                  lines; older files hold the message lines directly).
     pi           Pi / OpenClaw: a .jsonl whose first line is a session header and
                  whose entries form a tree; the transcript is the chain from the last
                  entry back to the root.
     gemini       Gemini CLI: session-*.json under ~/.gemini/tmp/<project>/chats/.
     opencode     OpenCode: a data directory (~/.local/share/opencode), one file per
                  session, message and part.

   Only text turns by the user and the assistant are kept. Text a harness put in the
   user's turn (command echoes, reminders, environment blocks, interrupted-request
   markers) is dropped, so what is left is what the person typed. Secrets are replaced
   by [redacted] before anything is kept: coding sessions hold API keys, tokens and
   passwords, and memory would otherwise learn them (Redact).

   The ChatGPT, Claude Code, Pi and OpenCode readers follow PasClaw's
   PasClaw.Session.Port (MIT, the same author), rewritten on fpjson. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, DateUtils, StrUtils, fpjson, jsonparser, RegExpr;

type
  TImportMsg = record
    Role: string;   { user | assistant }
    Text: string;
    At: Int64;      { unix seconds; 0 when the source has no time for it }
  end;
  TImportMsgs = array of TImportMsg;

  TImportConv = record
    Source: string; { chatgpt | claude-ai | claude-code | codex | pi | gemini | opencode }
    Id: string;     { the source's own id, else the file name }
    Title: string;
    Created: Int64; { unix seconds; 0 when unknown }
    Msgs: TImportMsgs;
  end;
  TImportConvs = array of TImportConv;

(* Read every conversation in APath: a file, a directory searched recursively for
   .jsonl and .json files, or an OpenCode data directory. APath may also name a
   source ("claude-code", "codex", "gemini", "opencode") to read its usual place
   under the home directory. False with AErr set when nothing could be read. *)
function ReadConversations(const APath: string; out AConvs: TImportConvs; out AErr: string): Boolean;

(* the conversations in one file's text (AName: the file name, for ids); the format
   name, or '' when the text is none of the known shapes *)
function ReadConversationText(const AText, AName: string; var AConvs: TImportConvs): string;

{ the usual directory of a source named on its own, '' for any other word }
function SourceDir(const AName: string): string;

{ AText with harness-added blocks taken out; '' when nothing the user typed is left }
function UserText(const AText: string): string;

(* AText with secrets replaced by [redacted]: private key blocks, keys and tokens in
   the shapes their services issue (OpenAI, Anthropic, GitHub, AWS, Slack, Google, JWT),
   and the value after "password", "api key", "secret" or "token" *)
function Redact(const AText: string): string;

(* "2026-11-10T20:04:49.123Z" (or with an offset, or a space for the T) in unix seconds;
   0 when it is not one *)
function IsoToUnix(const AStamp: string): Int64;

implementation

{ ---------------------------------------------------------------- helpers }

function ParseJson(const AText: string): TJSONData;
begin
  try
    Result := GetJSON(AText);
  except
    Result := nil;
  end;
end;

function Str(AObj: TJSONObject; const AKey: string): string;
var
  D: TJSONData;
begin
  Result := '';
  if AObj = nil then
    Exit;
  D := AObj.Find(AKey);
  if (D <> nil) and (D.JSONType in [jtString, jtNumber, jtBoolean]) then
    Result := D.AsString;
end;

function Obj(AObj: TJSONObject; const AKey: string): TJSONObject;
var
  D: TJSONData;
begin
  Result := nil;
  if AObj = nil then
    Exit;
  D := AObj.Find(AKey);
  if (D <> nil) and (D.JSONType = jtObject) then
    Result := TJSONObject(D);
end;

function Arr(AObj: TJSONObject; const AKey: string): TJSONArray;
var
  D: TJSONData;
begin
  Result := nil;
  if AObj = nil then
    Exit;
  D := AObj.Find(AKey);
  if (D <> nil) and (D.JSONType = jtArray) then
    Result := TJSONArray(D);
end;

function Num(AObj: TJSONObject; const AKey: string): Double;
var
  D: TJSONData;
begin
  Result := 0;
  if AObj = nil then
    Exit;
  D := AObj.Find(AKey);
  if (D <> nil) and (D.JSONType = jtNumber) then
    Result := D.AsFloat;
end;

function Flag(AObj: TJSONObject; const AKey: string): Boolean;
var
  D: TJSONData;
begin
  Result := False;
  if AObj = nil then
    Exit;
  D := AObj.Find(AKey);
  Result := (D <> nil) and (D.JSONType = jtBoolean) and D.AsBoolean;
end;

function IsoToUnix(const AStamp: string): Int64;
var
  S: string;
  Y, Mo, D, H, Mi, Se, OffH, OffM, P: Integer;
  T: TDateTime;
begin
  Result := 0;
  S := Trim(AStamp);
  if (Length(S) < 10) or (S[5] <> '-') or (S[8] <> '-') then
    Exit;
  if not (TryStrToInt(Copy(S, 1, 4), Y) and TryStrToInt(Copy(S, 6, 2), Mo) and TryStrToInt(Copy(S, 9, 2), D)) then
    Exit;
  H := 0;
  Mi := 0;
  Se := 0;
  if (Length(S) >= 19) and (S[11] in ['T', ' ']) then
  begin
    TryStrToInt(Copy(S, 12, 2), H);
    TryStrToInt(Copy(S, 15, 2), Mi);
    TryStrToInt(Copy(S, 18, 2), Se);
  end;
  if not TryEncodeDateTime(Y, Mo, D, H, Mi, Se, 0, T) then
    Exit;
  Result := DateTimeToUnix(T, True);
  { an offset after the time: +02:00 means two hours ahead of UTC }
  if Length(S) > 19 then
  begin
    P := Length(S) - 5;
    if (P > 19) and (S[P] in ['+', '-']) and (S[P + 3] = ':') and TryStrToInt(Copy(S, P + 1, 2), OffH) and
      TryStrToInt(Copy(S, P + 4, 2), OffM) then
      if S[P] = '+' then
        Dec(Result, OffH * 3600 + OffM * 60)
      else
        Inc(Result, OffH * 3600 + OffM * 60);
  end;
end;

{ a number of seconds or milliseconds since 1970 }
function EpochToUnix(AValue: Double): Int64;
begin
  Result := Trunc(AValue);
  if Result > 100000000000 then
    Result := Result div 1000;
end;

const
  { blocks a harness puts in the user's turn: taken out with what they hold }
  NOISE_TAGS: array[0..9] of string = ('system-reminder', 'command-name', 'command-message', 'command-args',
    'local-command-stdout', 'local-command-stderr', 'environment_context', 'user_instructions',
    'user-prompt-submit-hook', 'task-notification');
  { a turn that starts with one of these was not typed by the user }
  NOISE_STARTS: array[0..6] of string = ('Caveat: The messages below were generated',
    '[Request interrupted by user', '# AGENTS.md instructions', '<bash-input>', '<bash-stdout>',
    'This session is being continued from a previous conversation', '<turn_aborted>');

function UserText(const AText: string): string;
var
  I, P, Q: Integer;
  LOpen, LClose: string;
begin
  Result := AText;
  for I := 0 to High(NOISE_TAGS) do
  begin
    LOpen := '<' + NOISE_TAGS[I];
    LClose := '</' + NOISE_TAGS[I] + '>';
    P := Pos(LOpen, Result);
    while P > 0 do
    begin
      Q := PosEx(LClose, Result, P);
      if Q = 0 then
        Q := Length(Result) + 1 - Length(LClose);
      Delete(Result, P, Q + Length(LClose) - P);
      P := Pos(LOpen, Result);
    end;
  end;
  Result := Trim(Result);
  for I := 0 to High(NOISE_STARTS) do
    if AnsiStartsStr(NOISE_STARTS[I], Result) then
      Exit('');
end;

const
  SECRET_SHAPES: array[0..8] of string = (
    '-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)',
    '\bsk-(ant-|proj-)?[A-Za-z0-9_\-]{16,}',
    '\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}',
    '\bgithub_pat_[A-Za-z0-9_]{20,}',
    '\b(AKIA|ASIA)[0-9A-Z]{16}\b',
    '\bxox[abprs]-[A-Za-z0-9\-]{10,}',
    '\bAIza[0-9A-Za-z_\-]{30,}',
    '\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}',
    { the word stays, the value goes: "my password is X", "API_KEY=X", "token: X" (not "a token for") }
    '(?i)\b(password|passwd|passphrase|pass ?code|pin|api[ _\-]?key|secret|access[ _\-]?key|auth[ _\-]?token|token)' +
      '(\s*(is|was|:|=)\s*)["'']?[^\s"'',;]{2,}[^\s"'',;.]');

function Redact(const AText: string): string;
var
  I: Integer;
  R: TRegExpr;
begin
  Result := AText;
  for I := 0 to High(SECRET_SHAPES) do
  begin
    R := TRegExpr.Create(SECRET_SHAPES[I]);
    try
      if I = High(SECRET_SHAPES) then
        Result := R.Replace(Result, '$1$2[redacted]', True)
      else
        Result := R.Replace(Result, '[redacted]', False);
    finally
      R.Free;
    end;
  end;
end;

procedure AddMsg(var AConv: TImportConv; const ARole, AText: string; AAt: Int64);
var
  LText: string;
begin
  if ARole = 'user' then
    LText := UserText(AText)
  else if ARole = 'assistant' then
    LText := Trim(AText)
  else
    Exit;
  LText := Redact(LText);
  if LText = '' then
    Exit;
  SetLength(AConv.Msgs, Length(AConv.Msgs) + 1);
  AConv.Msgs[High(AConv.Msgs)].Role := ARole;
  AConv.Msgs[High(AConv.Msgs)].Text := LText;
  AConv.Msgs[High(AConv.Msgs)].At := AAt;
end;

procedure Keep(var AConvs: TImportConvs; const AConv: TImportConv);
begin
  if Length(AConv.Msgs) = 0 then
    Exit;
  SetLength(AConvs, Length(AConvs) + 1);
  AConvs[High(AConvs)] := AConv;
  if AConvs[High(AConvs)].Created = 0 then
    AConvs[High(AConvs)].Created := AConv.Msgs[0].At;
end;

function NewConv(const ASource, AId: string): TImportConv;
begin
  Result.Source := ASource;
  Result.Id := AId;
  Result.Title := '';
  Result.Created := 0;
  Result.Msgs := nil;
end;

(* the text blocks of a content value: a string, or an array of strings and typed
   blocks (text, input_text, output_text); tool calls, tool results, images and
   thinking are left out *)
function ContentText(AContent: TJSONData): string;
var
  I: Integer;
  B: TJSONData;
  T: string;
begin
  Result := '';
  if AContent = nil then
    Exit;
  if AContent.JSONType = jtString then
    Exit(AContent.AsString);
  if AContent.JSONType <> jtArray then
    Exit;
  for I := 0 to AContent.Count - 1 do
  begin
    B := AContent.Items[I];
    T := '';
    if B.JSONType = jtString then
      T := B.AsString
    else if B.JSONType = jtObject then
      if AnsiMatchStr(Str(TJSONObject(B), 'type'), ['text', 'input_text', 'output_text']) then
        T := Str(TJSONObject(B), 'text');
    if T = '' then
      Continue;
    if Result <> '' then
      Result := Result + LineEnding;
    Result := Result + T;
  end;
end;

function FileStem(const AName: string): string;
begin
  Result := ChangeFileExt(ExtractFileName(AName), '');
end;

function FileText(const APath: string): string;
var
  F: TFileStream;
begin
  Result := '';
  F := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Result, F.Size);
    if F.Size > 0 then
      F.ReadBuffer(Result[1], F.Size);
  finally
    F.Free;
  end;
end;

function FirstLine(const AText: string): string;
var
  P: Integer;
begin
  Result := TrimLeft(AText);
  P := Pos(#10, Result);
  if P > 0 then
    Result := Copy(Result, 1, P - 1);
  Result := Trim(Result);
end;

{ ---------------------------------------------------------------- ChatGPT }

procedure ReadChatGPT(AConvJ: TJSONObject; var AConvs: TImportConvs);
var
  LMap, LNode, LMsg, LContent: TJSONObject;
  LChain: TStringList;
  LCur, LText: string;
  LParts: TJSONArray;
  I, P: Integer;
  LConv: TImportConv;
begin
  LMap := Obj(AConvJ, 'mapping');
  if LMap = nil then
    Exit;
  LConv := NewConv('chatgpt', Str(AConvJ, 'id'));
  if LConv.Id = '' then
    LConv.Id := Str(AConvJ, 'conversation_id');
  LConv.Title := Str(AConvJ, 'title');
  LConv.Created := EpochToUnix(Num(AConvJ, 'create_time'));
  LChain := TStringList.Create;
  try
    LCur := Str(AConvJ, 'current_node');
    while (LCur <> '') and (LChain.Count < 100000) do
    begin
      LNode := Obj(LMap, LCur);
      if LNode = nil then
        Break;
      LChain.Add(LCur);
      LCur := Str(LNode, 'parent');
    end;
    for I := LChain.Count - 1 downto 0 do
    begin
      LMsg := Obj(Obj(LMap, LChain[I]), 'message');
      if LMsg = nil then
        Continue;
      LContent := Obj(LMsg, 'content');
      LParts := Arr(LContent, 'parts');
      LText := '';
      if LParts <> nil then
        for P := 0 to LParts.Count - 1 do
          if (LParts.Items[P].JSONType = jtString) and (LParts.Strings[P] <> '') then
            LText := LText + IfThen(LText = '', '', LineEnding) + LParts.Strings[P];
      AddMsg(LConv, LowerCase(Str(Obj(LMsg, 'author'), 'role')), LText, EpochToUnix(Num(LMsg, 'create_time')));
    end;
  finally
    LChain.Free;
  end;
  if LConv.Id = '' then
    LConv.Id := LConv.Title + '@' + IntToStr(LConv.Created);
  Keep(AConvs, LConv);
end;

{ ---------------------------------------------------------------- claude.ai }

procedure ReadClaudeAi(AConvJ: TJSONObject; var AConvs: TImportConvs);
var
  LMsgs: TJSONArray;
  LMsg: TJSONObject;
  I: Integer;
  LText, LRole: string;
  LConv: TImportConv;
begin
  LMsgs := Arr(AConvJ, 'chat_messages');
  if LMsgs = nil then
    Exit;
  LConv := NewConv('claude-ai', Str(AConvJ, 'uuid'));
  LConv.Title := Str(AConvJ, 'name');
  LConv.Created := IsoToUnix(Str(AConvJ, 'created_at'));
  for I := 0 to LMsgs.Count - 1 do
  begin
    if LMsgs.Items[I].JSONType <> jtObject then
      Continue;
    LMsg := TJSONObject(LMsgs.Items[I]);
    LRole := LowerCase(Str(LMsg, 'sender'));
    if LRole = 'human' then
      LRole := 'user';
    LText := ContentText(LMsg.Find('content'));
    if LText = '' then
      LText := Str(LMsg, 'text');
    AddMsg(LConv, LRole, LText, IsoToUnix(Str(LMsg, 'created_at')));
  end;
  if LConv.Id = '' then
    LConv.Id := LConv.Title + '@' + IntToStr(LConv.Created);
  Keep(AConvs, LConv);
end;

{ ---------------------------------------------------------------- JSON lines }

type
  TLineReader = procedure(ALine: TJSONObject; var AConv: TImportConv);

procedure ClaudeCodeLine(ALine: TJSONObject; var AConv: TImportConv);
var
  LKind: string;
  LMsg: TJSONObject;
begin
  LKind := Str(ALine, 'type');
  if (LKind = 'summary') and (AConv.Title = '') then
    AConv.Title := Str(ALine, 'summary');
  if (LKind <> 'user') and (LKind <> 'assistant') then
    Exit;
  { subagent turns, and lines the harness adds (isMeta), are not the user's }
  if Flag(ALine, 'isSidechain') or Flag(ALine, 'isMeta') or Flag(ALine, 'isCompactSummary') then
    Exit;
  LMsg := Obj(ALine, 'message');
  if LMsg = nil then
    Exit;
  AddMsg(AConv, LowerCase(Str(LMsg, 'role')), ContentText(LMsg.Find('content')), IsoToUnix(Str(ALine, 'timestamp')));
end;

procedure CodexLine(ALine: TJSONObject; var AConv: TImportConv);
var
  LItem: TJSONObject;
  LAt: Int64;
begin
  LAt := IsoToUnix(Str(ALine, 'timestamp'));
  LItem := ALine;
  if Str(ALine, 'type') = 'session_meta' then
  begin
    LItem := Obj(ALine, 'payload');
    if Str(LItem, 'id') <> '' then
      AConv.Id := Str(LItem, 'id');
    AConv.Created := IsoToUnix(Str(LItem, 'timestamp'));
    Exit;
  end;
  if Str(ALine, 'type') = 'response_item' then
    LItem := Obj(ALine, 'payload')
  else if (Str(ALine, 'id') <> '') and (ALine.Find('instructions') <> nil) then
  begin
    { an older rollout's header line }
    AConv.Id := Str(ALine, 'id');
    AConv.Created := IsoToUnix(Str(ALine, 'timestamp'));
    Exit;
  end;
  if Str(LItem, 'type') <> 'message' then
    Exit;
  AddMsg(AConv, LowerCase(Str(LItem, 'role')), ContentText(LItem.Find('content')), LAt);
end;

function ReadLines(const AText, ASource, AName: string; AReader: TLineReader; var AConvs: TImportConvs): Boolean;
var
  L: TStringList;
  I: Integer;
  D: TJSONData;
  LConv: TImportConv;
begin
  LConv := NewConv(ASource, AName);
  L := TStringList.Create;
  try
    L.Text := AText;
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      D := ParseJson(L[I]);
      if D = nil then
        Continue;
      try
        if D.JSONType = jtObject then
          AReader(TJSONObject(D), LConv);
      finally
        D.Free;
      end;
    end;
  finally
    L.Free;
  end;
  Result := Length(LConv.Msgs) > 0;
  Keep(AConvs, LConv);
end;

(* Pi / OpenClaw: entries form a tree; the transcript is the chain from the last
   entry (the live leaf of an append-only file) back to the root *)
function ReadPi(const AText, AName: string; var AConvs: TImportConvs): Boolean;
type
  TEntry = record
    Parent, Role, Text: string;
    At: Int64;
  end;
var
  L, LIds: TStringList;
  LEntries: array of TEntry;
  I, N, K: Integer;
  D: TJSONData;
  O, LMsg: TJSONObject;
  LChain: array of Integer;
  LCur, LLast: string;
  LConv: TImportConv;
begin
  LConv := NewConv('pi', AName);
  LEntries := nil;
  LLast := '';
  LIds := TStringList.Create;
  L := TStringList.Create;
  try
    LIds.Sorted := True;
    LIds.CaseSensitive := True;
    LIds.Duplicates := dupIgnore;
    L.Text := AText;
    for I := 0 to L.Count - 1 do
    begin
      D := ParseJson(L[I]);
      if D = nil then
        Continue;
      try
        if D.JSONType <> jtObject then
          Continue;
        O := TJSONObject(D);
        if Str(O, 'type') = 'session' then
        begin
          if Str(O, 'id') <> '' then
            LConv.Id := Str(O, 'id');
          LConv.Created := IsoToUnix(Str(O, 'timestamp'));
          Continue;
        end;
        if Str(O, 'id') = '' then
          Continue;
        N := Length(LEntries);
        SetLength(LEntries, N + 1);
        LEntries[N].Parent := Str(O, 'parentId');
        LEntries[N].Role := '';
        LEntries[N].Text := '';
        LEntries[N].At := IsoToUnix(Str(O, 'timestamp'));
        K := LIds.IndexOf(Str(O, 'id'));
        if K >= 0 then
          LIds.Objects[K] := TObject(PtrInt(N))
        else
          LIds.AddObject(Str(O, 'id'), TObject(PtrInt(N)));
        LLast := Str(O, 'id');
        if Str(O, 'type') <> 'message' then
          Continue;
        LMsg := Obj(O, 'message');
        LEntries[N].Role := LowerCase(Str(LMsg, 'role'));
        if LMsg <> nil then
          LEntries[N].Text := ContentText(LMsg.Find('content'));
      finally
        D.Free;
      end;
    end;
    LChain := nil;
    LCur := LLast;
    while (LCur <> '') and (Length(LChain) < Length(LEntries)) do
    begin
      K := LIds.IndexOf(LCur);
      if K < 0 then
        Break;
      N := PtrInt(LIds.Objects[K]);
      SetLength(LChain, Length(LChain) + 1);
      LChain[High(LChain)] := N;
      LCur := LEntries[N].Parent;
    end;
    for I := High(LChain) downto 0 do
      AddMsg(LConv, LEntries[LChain[I]].Role, LEntries[LChain[I]].Text, LEntries[LChain[I]].At);
  finally
    L.Free;
    LIds.Free;
  end;
  Result := Length(LConv.Msgs) > 0;
  Keep(AConvs, LConv);
end;

{ ---------------------------------------------------------------- Gemini CLI }

procedure ReadGemini(ASession: TJSONObject; const AName: string; var AConvs: TImportConvs);
var
  LMsgs: TJSONArray;
  LMsg: TJSONObject;
  I: Integer;
  LRole: string;
  LConv: TImportConv;
begin
  LMsgs := Arr(ASession, 'messages');
  if LMsgs = nil then
    Exit;
  LConv := NewConv('gemini', Str(ASession, 'sessionId'));
  if LConv.Id = '' then
    LConv.Id := AName;
  LConv.Created := IsoToUnix(Str(ASession, 'startTime'));
  for I := 0 to LMsgs.Count - 1 do
  begin
    if LMsgs.Items[I].JSONType <> jtObject then
      Continue;
    LMsg := TJSONObject(LMsgs.Items[I]);
    LRole := LowerCase(Str(LMsg, 'type'));
    if LRole = 'gemini' then
      LRole := 'assistant';
    AddMsg(LConv, LRole, ContentText(LMsg.Find('content')), IsoToUnix(Str(LMsg, 'timestamp')));
  end;
  Keep(AConvs, LConv);
end;

{ ---------------------------------------------------------------- one file }

function ReadConversationText(const AText, AName: string; var AConvs: TImportConvs): string;
var
  T, L1: string;
  D, D1: TJSONData;
  A: TJSONArray;
  I: Integer;
  O: TJSONObject;
begin
  Result := '';
  T := Trim(AText);
  if T = '' then
    Exit;
  if T[1] = '[' then
  begin
    D := ParseJson(T);
    if D = nil then
      Exit;
    try
      if (D.JSONType <> jtArray) or (D.Count = 0) or (D.Items[0].JSONType <> jtObject) then
        Exit;
      A := TJSONArray(D);
      O := TJSONObject(A.Items[0]);
      if O.Find('mapping') <> nil then
        Result := 'chatgpt'
      else if O.Find('chat_messages') <> nil then
        Result := 'claude-ai'
      else
        Exit;
      for I := 0 to A.Count - 1 do
        if A.Items[I].JSONType = jtObject then
          if Result = 'chatgpt' then
            ReadChatGPT(TJSONObject(A.Items[I]), AConvs)
          else
            ReadClaudeAi(TJSONObject(A.Items[I]), AConvs);
    finally
      D.Free;
    end;
    Exit;
  end;
  if T[1] <> '{' then
    Exit;
  (* JSON lines are told apart by their first line; a whole-file object (Gemini CLI)
     does not parse as one line unless it is written on one *)
  L1 := FirstLine(T);
  D1 := ParseJson(L1);
  try
    if (D1 <> nil) and (D1.JSONType = jtObject) then
    begin
      O := TJSONObject(D1);
      if (O.Find('sessionId') <> nil) and (Arr(O, 'messages') <> nil) then
      begin
        ReadGemini(O, FileStem(AName), AConvs);
        Exit('gemini');
      end;
      if (Str(O, 'type') = 'session') and (Str(O, 'id') <> '') then
      begin
        ReadPi(T, FileStem(AName), AConvs);
        Exit('pi');
      end;
      if (Str(O, 'type') = 'session_meta') or (Str(O, 'type') = 'response_item') or
        ((Str(O, 'id') <> '') and (O.Find('instructions') <> nil)) then
      begin
        ReadLines(T, 'codex', FileStem(AName), @CodexLine, AConvs);
        Exit('codex');
      end;
      if (O.Find('type') <> nil) and ((O.Find('message') <> nil) or (O.Find('summary') <> nil) or
        (O.Find('uuid') <> nil) or (O.Find('sessionId') <> nil)) then
      begin
        ReadLines(T, 'claude-code', FileStem(AName), @ClaudeCodeLine, AConvs);
        Exit('claude-code');
      end;
    end;
  finally
    D1.Free;
  end;
  D := ParseJson(T);
  if D = nil then
    Exit;
  try
    if (D.JSONType = jtObject) and (TJSONObject(D).Find('sessionId') <> nil) and
      (Arr(TJSONObject(D), 'messages') <> nil) then
    begin
      ReadGemini(TJSONObject(D), FileStem(AName), AConvs);
      Result := 'gemini';
    end;
  finally
    D.Free;
  end;
end;

{ ---------------------------------------------------------------- OpenCode }

procedure ListSorted(const ADir: string; ADirs: Boolean; AOut: TStringList);
var
  R: TSearchRec;
begin
  AOut.Clear;
  if FindFirst(IncludeTrailingPathDelimiter(ADir) + '*', faAnyFile or faDirectory, R) = 0 then
  try
    repeat
      if (R.Name = '.') or (R.Name = '..') then
        Continue;
      if ((R.Attr and faDirectory) <> 0) = ADirs then
        AOut.Add(R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
  AOut.Sort;
end;

function JsonFile(const APath: string): TJSONObject;
var
  D: TJSONData;
begin
  Result := nil;
  try
    D := ParseJson(FileText(APath));
  except
    D := nil;
  end;
  if (D <> nil) and (D.JSONType = jtObject) then
    Result := TJSONObject(D)
  else
    D.Free;
end;

function OpenCodeText(const ARoot, ASession, AMsgId: string; AMsg: TJSONObject): string;
var
  LDir: string;
  LFiles: TStringList;
  I: Integer;
  P: TJSONObject;
begin
  Result := ContentText(AMsg.Find('parts'));
  if Result <> '' then
    Exit;
  LDir := ARoot + 'part' + PathDelim + ASession + PathDelim + AMsgId;
  if not DirectoryExists(LDir) then
    LDir := ARoot + 'part' + PathDelim + AMsgId;
  if DirectoryExists(LDir) then
  begin
    LFiles := TStringList.Create;
    try
      ListSorted(LDir, False, LFiles);
      for I := 0 to LFiles.Count - 1 do
      begin
        P := JsonFile(LDir + PathDelim + LFiles[I]);
        if P = nil then
          Continue;
        try
          if Str(P, 'type') = 'text' then
            Result := Result + IfThen(Result = '', '', LineEnding) + Str(P, 'text');
        finally
          P.Free;
        end;
      end;
    finally
      LFiles.Free;
    end;
  end;
  if Result = '' then
    Result := ContentText(AMsg.Find('content'));
end;

function IsOpenCode(const ADir: string; out ARoot: string): Boolean;
begin
  ARoot := IncludeTrailingPathDelimiter(ADir);
  if DirectoryExists(ARoot + 'storage') then
    ARoot := ARoot + 'storage' + PathDelim;
  Result := DirectoryExists(ARoot + 'session') and DirectoryExists(ARoot + 'message');
end;

procedure ReadOpenCode(const ARoot: string; var AConvs: TImportConvs);
var
  LProjects, LSessions, LMsgFiles: TStringList;
  I, J, K: Integer;
  LDir, LMsgDir: string;
  S, M: TJSONObject;
  LConv: TImportConv;
begin
  LProjects := TStringList.Create;
  LSessions := TStringList.Create;
  LMsgFiles := TStringList.Create;
  try
    ListSorted(ARoot + 'session', True, LProjects);
    LProjects.Insert(0, '');
    for I := 0 to LProjects.Count - 1 do
    begin
      LDir := ARoot + 'session' + PathDelim + IfThen(LProjects[I] = '', '', LProjects[I] + PathDelim);
      ListSorted(LDir, False, LSessions);
      for J := 0 to LSessions.Count - 1 do
      begin
        if LowerCase(ExtractFileExt(LSessions[J])) <> '.json' then
          Continue;
        S := JsonFile(LDir + LSessions[J]);
        if S = nil then
          Continue;
        try
          LConv := NewConv('opencode', Str(S, 'id'));
          if LConv.Id = '' then
            LConv.Id := FileStem(LSessions[J]);
          LConv.Title := Str(S, 'title');
          LConv.Created := EpochToUnix(Num(Obj(S, 'time'), 'created'));
        finally
          S.Free;
        end;
        LMsgDir := ARoot + 'message' + PathDelim + LConv.Id;
        if not DirectoryExists(LMsgDir) then
          Continue;
        ListSorted(LMsgDir, False, LMsgFiles);
        for K := 0 to LMsgFiles.Count - 1 do
        begin
          if LowerCase(ExtractFileExt(LMsgFiles[K])) <> '.json' then
            Continue;
          M := JsonFile(LMsgDir + PathDelim + LMsgFiles[K]);
          if M = nil then
            Continue;
          try
            AddMsg(LConv, LowerCase(Str(M, 'role')),
              OpenCodeText(ARoot, LConv.Id, IfThen(Str(M, 'id') <> '', Str(M, 'id'), FileStem(LMsgFiles[K])), M),
              EpochToUnix(Num(Obj(M, 'time'), 'created')));
          finally
            M.Free;
          end;
        end;
        Keep(AConvs, LConv);
      end;
    end;
  finally
    LMsgFiles.Free;
    LSessions.Free;
    LProjects.Free;
  end;
end;

{ ---------------------------------------------------------------- paths }

function HomeDir: string;
begin
  Result := GetEnvironmentVariable('HOME');
  if Result = '' then
    Result := GetEnvironmentVariable('USERPROFILE');
  Result := IncludeTrailingPathDelimiter(Result);
end;

function SourceDir(const AName: string): string;
var
  N: string;
begin
  N := LowerCase(AName);
  if (N = 'claude-code') or (N = 'claude') then
    Result := HomeDir + '.claude' + PathDelim + 'projects'
  else if N = 'codex' then
    Result := HomeDir + '.codex' + PathDelim + 'sessions'
  else if N = 'gemini' then
    Result := HomeDir + '.gemini' + PathDelim + 'tmp'
  else if N = 'opencode' then
    Result := HomeDir + '.local' + PathDelim + 'share' + PathDelim + 'opencode'
  else
    Result := '';
end;

procedure WalkFiles(const ADir: string; AOut: TStringList; ADepth: Integer);
var
  LNames: TStringList;
  I: Integer;
  E: string;
begin
  if ADepth > 8 then
    Exit;
  LNames := TStringList.Create;
  try
    ListSorted(ADir, False, LNames);
    for I := 0 to LNames.Count - 1 do
    begin
      E := LowerCase(ExtractFileExt(LNames[I]));
      if (E = '.jsonl') or (E = '.json') then
        AOut.Add(IncludeTrailingPathDelimiter(ADir) + LNames[I]);
    end;
    ListSorted(ADir, True, LNames);
    for I := 0 to LNames.Count - 1 do
      { Claude Code keeps each session's subagent transcripts in <session>/subagents/ }
      if not SameText(LNames[I], 'subagents') then
        WalkFiles(IncludeTrailingPathDelimiter(ADir) + LNames[I], AOut, ADepth + 1);
  finally
    LNames.Free;
  end;
end;

function ReadConversations(const APath: string; out AConvs: TImportConvs; out AErr: string): Boolean;
var
  LPath, LRoot: string;
  LFiles: TStringList;
  I: Integer;
begin
  AConvs := nil;
  AErr := '';
  LPath := APath;
  if SourceDir(LPath) <> '' then
    LPath := SourceDir(LPath);
  if (Length(LPath) > 1) and (LPath[1] = '~') then
    LPath := HomeDir + Copy(LPath, 3, MaxInt);
  LPath := ExcludeTrailingPathDelimiter(LPath);
  if DirectoryExists(LPath) then
  begin
    if IsOpenCode(LPath, LRoot) then
      ReadOpenCode(LRoot, AConvs)
    else
    begin
      LFiles := TStringList.Create;
      try
        WalkFiles(LPath, LFiles, 0);
        for I := 0 to LFiles.Count - 1 do
          try
            ReadConversationText(FileText(LFiles[I]), LFiles[I], AConvs);
          except
            on E: Exception do
              { an unreadable file is skipped, not the whole import }
              AErr := LFiles[I] + ': ' + E.Message;
          end;
      finally
        LFiles.Free;
      end;
    end;
  end
  else if FileExists(LPath) then
  begin
    if ReadConversationText(FileText(LPath), LPath, AConvs) = '' then
      AErr := LPath + ' is none of the known formats (ChatGPT or claude.ai conversations.json, ' +
        'a Claude Code, Codex or Pi .jsonl, a Gemini CLI session)';
  end
  else
    AErr := 'no file or directory ' + LPath;
  Result := Length(AConvs) > 0;
  if (not Result) and (AErr = '') then
    AErr := 'no conversations found in ' + LPath;
end;

end.
