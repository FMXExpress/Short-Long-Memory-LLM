program jevban_test;

(* Unit checks for jevban: addresses and CIDR, the Apache parser, the
   signatures, and the Jev request/response shapes. Exit code 1 on any
   failure. *)

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, fpjson, jsonparser,
  Ban.Net, Ban.Apache, Ban.Signatures, Ban.Jev;

var
  Failures: Integer = 0;
  Checks: Integer = 0;

procedure Check(ACond: Boolean; const AWhat: string);
begin
  Inc(Checks);
  if not ACond then
  begin
    Inc(Failures);
    WriteLn('FAIL: ', AWhat);
  end;
end;

procedure TestNet;
var
  A: TIpAddr;
  C: TCidr;
  L: TCidrList;

  function Canon(const S: string): string;
  var
    LA: TIpAddr;
  begin
    if ParseIp(S, LA) then
      Result := IpToStr(LA)
    else
      Result := '(invalid)';
  end;

begin
  Check(Canon('192.0.2.1') = '192.0.2.1', 'ipv4');
  Check(Canon('::ffff:192.0.2.1') = '192.0.2.1', 'v4-mapped prints as v4');
  Check(Canon('2001:DB8:0:0:0:0:0:1') = '2001:db8::1', 'v6 compress');
  Check(Canon('2001:db8::') = '2001:db8::', 'v6 trailing ::');
  Check(Canon('::1') = '::1', 'v6 loopback');
  Check(Canon('::') = '::', 'v6 any');
  Check(Canon('2001:db8:0:1:0:0:0:1') = '2001:db8:0:1::1', 'v6 longest zero run');
  Check(Canon('[2001:db8::1]') = '2001:db8::1', 'v6 brackets');
  Check(Canon('1:2:3:4:5:6:7:8') = '1:2:3:4:5:6:7:8', 'v6 full');
  Check(Canon('256.1.1.1') = '(invalid)', 'v4 octet range');
  Check(Canon('1.2.3') = '(invalid)', 'v4 short');
  Check(Canon('1.2.3.4; rm -rf /') = '(invalid)', 'shell text rejected');
  Check(Canon('1::2::3') = '(invalid)', 'two ::');
  Check(Canon('1:2:3:4:5:6:7:8:9') = '(invalid)', 'nine groups');
  Check(Canon('example.com') = '(invalid)', 'hostname');
  Check(Canon('$(id)') = '(invalid)', 'command substitution');

  Check(ParseCidr('10.0.0.0/8', C) and ParseIp('10.200.3.4', A) and CidrMatch(C, A), 'v4 cidr in');
  Check(ParseIp('11.0.0.1', A) and not CidrMatch(C, A), 'v4 cidr out');
  Check(ParseCidr('2001:db8::/32', C) and ParseIp('2001:db8:ffff::1', A) and CidrMatch(C, A), 'v6 cidr in');
  Check(ParseIp('2001:db9::1', A) and not CidrMatch(C, A), 'v6 cidr out');
  Check(not ParseCidr('10.0.0.0/33', C), 'v4 prefix range');
  L := ParseCidrList('127.0.0.0/8, ::1 192.168.1.0/24');
  Check(Length(L) = 3, 'cidr list');
  Check(ParseIp('192.168.1.77', A) and InCidrList(L, A), 'in list');
  Check(ParseIp('::1', A) and InCidrList(L, A), 'v6 in list');
  Check(ParseIp('8.8.8.8', A) and not InCidrList(L, A), 'not in list');
end;

procedure TestApache;
var
  E: TAccessEntry;
begin
  Check(ParseAccessLine('203.0.113.66 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif?a=1 HTTP/1.0" 200 2326 "http://www.example.com/start.html" "Mozilla/4.08 [en] (Win98; I ;Nav)"', E), 'combined');
  Check(E.IpText = '203.0.113.66', 'combined ip');
  Check((E.Method = 'GET') and (E.Target = '/apache_pb.gif?a=1') and (E.Protocol = 'HTTP/1.0'), 'request split');
  Check((E.Status = 200) and (E.Bytes = 2326), 'status, bytes');
  Check(E.Agent = 'Mozilla/4.08 [en] (Win98; I ;Nav)', 'agent with brackets');
  Check(E.Referer = 'http://www.example.com/start.html', 'referer');
  Check(E.Epoch = 971211336, 'time with offset -> UTC: ' + IntToStr(E.Epoch));

  Check(ParseAccessLine('127.0.0.1 - - [10/Oct/2000:13:55:36 +0000] "GET / HTTP/1.1" 304 -', E), 'common');
  Check((E.Status = 304) and (E.Bytes = 0) and (E.Agent = ''), 'common fields');

  Check(ParseAccessLine('www.example.com:443 2001:db8::1 - - [10/Oct/2000:13:55:36 +0530] "POST /x HTTP/2.0" 201 5 "-" "ua"', E), 'vhost_combined');
  Check((E.VHost = 'www.example.com:443') and (E.IpText = '2001:db8::1'), 'vhost + v6');
  Check(E.Epoch = 971166336, 'positive offset: ' + IntToStr(E.Epoch));

  Check(ParseAccessLine('192.0.2.66 - - [31/Oct/2026:10:00:11 +0000] "\x16\x03\x01\x02" 400 226 "-" "-"', E), 'binary request');
  Check((E.Method = '') and (E.Target = '\x16\x03\x01\x02'), 'binary request kept whole');
  Check(ParseAccessLine('192.0.2.1 - - [31/Oct/2026:10:00:11 +0000] "GET /a\"b HTTP/1.1" 404 1 "-" "x \"quoted\" y"', E), 'escaped quotes');
  Check(E.Agent = 'x \"quoted\" y', 'agent keeps escapes');
  Check(ParseAccessLine('192.0.2.1 - - [31/Oct/2026:10:00:11 +0000] "GET /%252e%252e/etc/passwd HTTP/1.1" 404 1', E) and
    (E.Decoded = '/../etc/passwd'), 'double url-decoding: ' + E.Decoded);

  Check(not ParseAccessLine('garbage', E), 'garbage');
  Check(not ParseAccessLine('not-an-ip - - [31/Oct/2026:10:00:11 +0000] "GET / HTTP/1.1" 200 1', E), 'hostname client');
  Check(not ParseAccessLine('', E), 'empty');
end;

procedure TestSignatures;
var
  S: TSignatureSet;
  Lines: TStringList;
  E: TAccessEntry;
  Tags: TStringList;
  I, LScore: Integer;
  LExpect: array[0..14] of Integer = (
    0, 0, 0, 0,          { googlebot, a visitor, a 404 }
    10, 10, 6, 6,        { .env, .git, wp-config.php.bak, phpmyadmin }
    10, 10, 20, 10, 6,   { thinkphp, traversal, sqli + sqlmap, jndi, binary }
    10, 3);              { .env from localhost, wp-login }

  function MinScore(AIdx: Integer): Integer;
  begin
    Result := LExpect[AIdx];
  end;

begin
  S := TSignatureSet.Create;
  Lines := TStringList.Create;
  Tags := TStringList.Create;
  try
    Check(S.Count > 20, 'builtin signatures');
    Lines.LoadFromFile(ExtractFilePath(ParamStr(0)) + 'sample_access.log');
    Check(Lines.Count = 15, 'sample lines');
    for I := 0 to Lines.Count - 1 do
    begin
      Check(ParseAccessLine(Lines[I], E), Format('sample %d parses', [I + 1]));
      Tags.Clear;
      LScore := S.Score(E, Tags);
      if MinScore(I) = 0 then
        Check(LScore = 0, Format('sample %d is clean, got %d [%s]', [I + 1, LScore, Tags.CommaText]))
      else
        Check(LScore >= MinScore(I), Format('sample %d scores >= %d, got %d [%s]', [I + 1, MinScore(I),
          LScore, Tags.CommaText]));
    end;
  finally
    Tags.Free;
    Lines.Free;
    S.Free;
  end;
end;

procedure TestSignatureFile;
var
  S: TSignatureSet;
  F: TStringList;
  E: TAccessEntry;
  LPath: string;
begin
  LPath := GetTempFileName;
  F := TStringList.Create;
  try
    F.Add('# custom rules');
    F.Add('10 url secret-admin ^/my-hidden-admin');
    F.Add('');
    F.Add('5	agent bad-bot BadBot/\d');
    F.SaveToFile(LPath);
  finally
    F.Free;
  end;
  S := TSignatureSet.Create(False);
  try
    S.LoadFile(LPath);
    Check(S.Count = 2, 'file rules');
    ParseAccessLine('192.0.2.1 - - [31/Oct/2026:10:00:11 +0000] "GET /my-hidden-admin/x HTTP/1.1" 404 1 "-" "BadBot/2"', E);
    Check(S.Score(E, nil) = 15, 'file rules score');
    ParseAccessLine('192.0.2.1 - - [31/Oct/2026:10:00:11 +0000] "GET /.env HTTP/1.1" 404 1 "-" "x"', E);
    Check(S.Score(E, nil) = 0, 'builtins off');
  finally
    S.Free;
    DeleteFile(LPath);
  end;
end;

procedure TestJev;
var
  J: TJevClient;
  LRecent: TStringList;
  LData: TJSONData;
  LRoot, LQ: TJSONObject;
  V: TJevVerdict;
begin
  J := TJevClient.Create;
  LRecent := TStringList.Create;
  try
    LRecent.Add('31/Oct/2026:10:00:05 "GET /.env HTTP/1.1" 404 196 ua="x \"y\""');
    LData := GetJSON(J.BuildRequest('203.0.113.66', 'GET /.git/config HTTP/1.1', 'vcs-dir', LRecent));
    try
      LRoot := TJSONObject(LData);
      Check(LRoot.Get('model', '') = 'openjev', 'model');
      Check(TJSONObject(LRoot.Objects['state']).Get('client_ip', '') = '203.0.113.66', 'state.client_ip');
      Check(TJSONObject(LRoot.Objects['state']).Arrays['recent_requests_from_this_client_oldest_first'].Count = 1,
        'state recent');
      LQ := LRoot.Objects['questions'].Objects['malicious'];
      Check((LQ.Get('type', '') = 'noul') and (LQ.Get('instructions', '') <> ''), 'noul question');
      LQ := LRoot.Objects['questions'].Objects['intent'];
      Check((LQ.Get('type', '') = 'choice') and (LQ.Objects['criteria'].Count = 6), 'choice question');
    finally
      LData.Free;
    end;
  finally
    LRecent.Free;
    J.Free;
  end;

  Check(ParseJevResponse('{"id":"shim-1","answers":{"malicious":{"type":"noul","noul":0.9731},' +
    '"intent":{"type":"choice","choice":"vulnerability_scanner","probabilities":{},"confidence":0.88}},' +
    '"usage":{"input_tokens":512,"output_tokens":0}}', V), 'response parses');
  Check(V.Ok and (Abs(V.Malicious - 0.9731) < 1e-9) and (V.Intent = 'vulnerability_scanner') and
    (Abs(V.IntentConfidence - 0.88) < 1e-9), 'response fields');
  Check(ParseJevResponse('{"answers":{"malicious":{"type":"noul","noul":0}}}', V) and (V.Malicious = 0),
    'integer noul');
  Check(not ParseJevResponse('{"answers":{}}', V) and (V.Error <> ''), 'missing answer');
  Check(not ParseJevResponse('{"answers":{"malicious":{"noul":1.5}}}', V), 'out of range');
  Check(not ParseJevResponse('<html>bad gateway</html>', V), 'not json');
end;

begin
  TestNet;
  TestApache;
  TestSignatures;
  TestSignatureFile;
  TestJev;
  WriteLn(Format('%d checks, %d failed', [Checks, Failures]));
  if Failures > 0 then
    ExitCode := 1;
end.
