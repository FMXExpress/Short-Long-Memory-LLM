program import_test;

(* Mem.Import on a small sample of each format: which turns are kept, in what order,
   with which times, and that harness-added text never reaches memory. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  SysUtils, Classes, Mem.Import;

var
  Fails, Checks: Integer;
  Dir: string;

procedure Check(ACond: Boolean; const AWhat: string);
begin
  Inc(Checks);
  if not ACond then
  begin
    Inc(Fails);
    WriteLn('FAIL: ', AWhat);
  end;
end;

procedure Put(const APath, AText: string);
var
  L: TStringList;
begin
  ForceDirectories(ExtractFileDir(APath));
  L := TStringList.Create;
  try
    L.Text := AText;
    L.SaveToFile(APath);
  finally
    L.Free;
  end;
end;

function Read1(const APath: string; out C: TImportConv): Boolean;
var
  A: TImportConvs;
  E: string;
begin
  Result := ReadConversations(APath, A, E) and (Length(A) = 1);
  if not Result then
    WriteLn('  ', APath, ': ', Length(A), ' conversations, ', E)
  else
    C := A[0];
end;

function Turns(const C: TImportConv): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(C.Msgs) do
    Result := Result + C.Msgs[I].Role[1] + ':' + C.Msgs[I].Text + '|';
end;

var
  C: TImportConv;
  A: TImportConvs;
  E: string;
begin
  Fails := 0;
  Checks := 0;
  Dir := IncludeTrailingPathDelimiter(GetTempDir(False)) + 'memagent-import-test-' + IntToStr(GetProcessID) +
    PathDelim;

  Check(IsoToUnix('2026-11-10T20:04:49.123Z') = 1794341089, 'ISO time, UTC');
  Check(IsoToUnix('2026-11-10T22:04:49+02:00') = 1794341089, 'ISO time with an offset');
  Check(IsoToUnix('yesterday') = 0, 'not a time');
  Check(UserText('<command-name>/clear</command-name>') = '', 'a command echo is not the user''s text');
  Check(UserText('My cat is Tom.<system-reminder>be brief</system-reminder>') = 'My cat is Tom.',
    'a reminder inside a turn is cut out');
  Check(UserText('[Request interrupted by user]') = '', 'interrupted marker');
  Check(Redact('My password is hunter2.') = 'My password is [redacted].', 'redact: password: ' +
    Redact('My password is hunter2.'));
  Check(Redact('export OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234') = 'export OPENAI_API_KEY=[redacted]',
    'redact: key after =: ' + Redact('export OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234'));
  Check(Redact('use ghp_abcdefghijklmnopqrstuvwxyz123456 here') = 'use [redacted] here', 'redact: GitHub token');
  Check(Redact('key AKIAIOSFODNN7EXAMPLE ok') = 'key [redacted] ok', 'redact: AWS key id: ' +
    Redact('key AKIAIOSFODNN7EXAMPLE ok'));
  Check(Redact('a' + LineEnding + '-----BEGIN RSA PRIVATE KEY-----' + LineEnding + 'MIIE' + LineEnding +
    '-----END RSA PRIVATE KEY-----' + LineEnding + 'b') = 'a' + LineEnding + '[redacted]' + LineEnding + 'b',
    'redact: private key block');
  Check(Redact('My dog is Biscuit and the tokenizer is fast.') = 'My dog is Biscuit and the tokenizer is fast.',
    'redact: ordinary words stay');
  Check(Redact('I need a token for the bus.') = 'I need a token for the bus.', 'redact: "a token for": ' +
    Redact('I need a token for the bus.'));

  { ChatGPT: the parent chain from current_node; the abandoned edit "n2" falls away }
  Put(Dir + 'chatgpt' + PathDelim + 'conversations.json',
    '[{"id":"c1","title":"Pets","create_time":1794341089.5,"current_node":"n4","mapping":{' +
    '"n0":{"message":null,"parent":null},' +
    '"n1":{"message":{"author":{"role":"system"},"content":{"parts":["sys"]}},"parent":"n0"},' +
    '"n2":{"message":{"author":{"role":"user"},"content":{"parts":["My dog is Rex."]}},"parent":"n1"},' +
    '"n3":{"message":{"author":{"role":"user"},"content":{"parts":["My dog is Biscuit."]},"create_time":1794341100},"parent":"n1"},' +
    '"n4":{"message":{"author":{"role":"assistant"},"content":{"parts":["Nice name!"]}},"parent":"n3"}}}]');
  if Read1(Dir + 'chatgpt' + PathDelim + 'conversations.json', C) then
  begin
    Check(C.Source = 'chatgpt', 'chatgpt: format');
    Check(Turns(C) = 'u:My dog is Biscuit.|a:Nice name!|', 'chatgpt: the live branch only: ' + Turns(C));
    Check((C.Id = 'c1') and (C.Title = 'Pets') and (C.Created = 1794341089), 'chatgpt: id, title, time');
    Check(C.Msgs[0].At = 1794341100, 'chatgpt: message time');
  end
  else
    Check(False, 'chatgpt: read');

  { claude.ai export }
  Put(Dir + 'claude-ai.json',
    '[{"uuid":"u1","name":"Trip","created_at":"2026-11-10T20:04:49Z","chat_messages":[' +
    '{"sender":"human","text":"I live in Reykjavik.","created_at":"2026-11-10T20:05:00Z","content":[{"type":"text","text":"I live in Reykjavik."}]},' +
    '{"sender":"assistant","text":"Lovely city.","content":[{"type":"tool_use","name":"x"},{"type":"text","text":"Lovely city."}]}]}]');
  if Read1(Dir + 'claude-ai.json', C) then
    Check((C.Source = 'claude-ai') and (Turns(C) = 'u:I live in Reykjavik.|a:Lovely city.|') and
      (C.Msgs[0].At = 1794341100), 'claude-ai: ' + Turns(C))
  else
    Check(False, 'claude-ai: read');

  { Claude Code: subagent and meta lines, tool results and command echoes are dropped }
  Put(Dir + 'claude' + PathDelim + 'proj' + PathDelim + 'sess-1.jsonl',
    '{"type":"summary","summary":"Fix build","leafUuid":"x"}' + LineEnding +
    '{"type":"user","isMeta":true,"message":{"role":"user","content":"Caveat: The messages below were generated by the user"},"uuid":"a","sessionId":"sess-1","timestamp":"2026-11-10T20:04:49Z"}' + LineEnding +
    '{"type":"user","message":{"role":"user","content":"<command-name>/clear</command-name>"},"uuid":"b","sessionId":"sess-1"}' + LineEnding +
    '{"type":"user","message":{"role":"user","content":"My sister Ana lives in Lisbon. Fix the build."},"uuid":"c","sessionId":"sess-1","timestamp":"2026-11-10T20:04:50Z"}' + LineEnding +
    '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hm"},{"type":"text","text":"On it."},{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},"uuid":"d","sessionId":"sess-1"}' + LineEnding +
    '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]},"uuid":"e","sessionId":"sess-1"}' + LineEnding +
    '{"type":"user","isSidechain":true,"message":{"role":"user","content":"subagent prompt"},"uuid":"f","sessionId":"sess-1"}');
  Put(Dir + 'claude' + PathDelim + 'proj' + PathDelim + 'sess-1' + PathDelim + 'subagents' + PathDelim + 'agent-1.jsonl',
    '{"type":"user","message":{"role":"user","content":"never read"},"uuid":"g","sessionId":"sess-1"}');
  if Read1(Dir + 'claude', C) then
  begin
    Check(C.Source = 'claude-code', 'claude-code: format');
    Check(Turns(C) = 'u:My sister Ana lives in Lisbon. Fix the build.|a:On it.|', 'claude-code: ' + Turns(C));
    Check((C.Id = 'sess-1') and (C.Title = 'Fix build') and (C.Created = 1794341090), 'claude-code: id, title, time');
  end
  else
    Check(False, 'claude-code: read (subagents/ skipped)');

  { Codex CLI rollout: developer and environment turns dropped }
  Put(Dir + 'codex' + PathDelim + '2026' + PathDelim + 'rollout-1.jsonl',
    '{"timestamp":"2026-11-10T20:04:49Z","type":"session_meta","payload":{"id":"cx-1","timestamp":"2026-11-10T20:04:49Z","cwd":"/x"}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:49Z","type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"rules"}]}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:49Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>cwd</environment_context>"}]}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:50Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"I prefer tabs."}]}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:50Z","type":"event_msg","payload":{"type":"user_message","message":"I prefer tabs."}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:51Z","type":"response_item","payload":{"type":"function_call","name":"shell"}}' + LineEnding +
    '{"timestamp":"2026-11-10T20:04:52Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Noted."}]}}');
  if Read1(Dir + 'codex', C) then
    Check((C.Source = 'codex') and (C.Id = 'cx-1') and (Turns(C) = 'u:I prefer tabs.|a:Noted.|'),
      'codex: ' + C.Id + ' ' + Turns(C))
  else
    Check(False, 'codex: read');

  { Pi / OpenClaw: the chain from the last entry; the branch b1 falls away }
  Put(Dir + 'pi.jsonl',
    '{"type":"session","id":"pi-1","timestamp":"2026-11-10T20:04:49Z","cwd":"/x"}' + LineEnding +
    '{"type":"message","id":"a1","parentId":null,"message":{"role":"user","content":[{"type":"text","text":"My bike is red."}]}}' + LineEnding +
    '{"type":"message","id":"b1","parentId":"a1","message":{"role":"assistant","content":[{"type":"text","text":"old reply"}]}}' + LineEnding +
    '{"type":"model_change","id":"m1","parentId":"a1"}' + LineEnding +
    '{"type":"message","id":"b2","parentId":"m1","message":{"role":"assistant","content":[{"type":"text","text":"Red bikes are fast."}]}}');
  if Read1(Dir + 'pi.jsonl', C) then
    Check((C.Source = 'pi') and (C.Id = 'pi-1') and (Turns(C) = 'u:My bike is red.|a:Red bikes are fast.|'),
      'pi: ' + Turns(C))
  else
    Check(False, 'pi: read');

  { Gemini CLI }
  Put(Dir + 'gemini' + PathDelim + 'h' + PathDelim + 'chats' + PathDelim + 'session-1.json',
    '{' + LineEnding + '  "sessionId": "g-1", "startTime": "2026-11-10T20:04:49Z", "messages": [' + LineEnding +
    '  {"id":"1","timestamp":"2026-11-10T20:04:50Z","type":"user","content":"I am vegetarian."},' + LineEnding +
    '  {"id":"2","timestamp":"2026-11-10T20:04:51Z","type":"gemini","content":"Got it."}]}');
  if Read1(Dir + 'gemini', C) then
    Check((C.Source = 'gemini') and (C.Id = 'g-1') and (Turns(C) = 'u:I am vegetarian.|a:Got it.|'), 'gemini: ' + Turns(C))
  else
    Check(False, 'gemini: read');

  { OpenCode: inline parts (old) and per-part files (new) }
  Put(Dir + 'oc' + PathDelim + 'storage' + PathDelim + 'session' + PathDelim + 'p1' + PathDelim + 'ses_1.json',
    '{"id":"ses_1","title":"Hi","time":{"created":1794341089000}}');
  Put(Dir + 'oc' + PathDelim + 'storage' + PathDelim + 'message' + PathDelim + 'ses_1' + PathDelim + 'msg_1.json',
    '{"id":"msg_1","role":"user","parts":[{"type":"text","text":"My son is Leo."}],"time":{"created":1794341090000}}');
  Put(Dir + 'oc' + PathDelim + 'storage' + PathDelim + 'message' + PathDelim + 'ses_1' + PathDelim + 'msg_2.json',
    '{"id":"msg_2","role":"assistant"}');
  Put(Dir + 'oc' + PathDelim + 'storage' + PathDelim + 'part' + PathDelim + 'msg_2' + PathDelim + 'prt_1.json',
    '{"type":"text","text":"Hello Leo''s parent."}');
  if Read1(Dir + 'oc', C) then
    Check((C.Source = 'opencode') and (C.Created = 1794341089) and (C.Msgs[0].At = 1794341090) and
      (Turns(C) = 'u:My son is Leo.|a:Hello Leo''s parent.|'), 'opencode: ' + Turns(C))
  else
    Check(False, 'opencode: read');

  { everything under one directory (the OpenCode tree is not walked as files) }
  Check(ReadConversations(Dir + 'claude', A, E) and (Length(A) = 1), 'one Claude Code session in claude/');
  Check(not ReadConversations(Dir + 'nothing-here', A, E) and (E <> ''), 'a missing path says so');
  Put(Dir + 'other.json', '{"hello":"world"}');
  Check(not ReadConversations(Dir + 'other.json', A, E) and (Pos('none of the known formats', E) > 0),
    'an unknown file says so');
  Check(SourceDir('codex') <> '', 'codex has a usual place');
  Check(SourceDir('readme') = '', 'any other word is a path');

  if Fails = 0 then
    WriteLn('import_test: ', Checks, '/', Checks, ' checks passed')
  else
  begin
    WriteLn('import_test: ', Fails, ' of ', Checks, ' checks failed');
    Halt(1);
  end;
end.
