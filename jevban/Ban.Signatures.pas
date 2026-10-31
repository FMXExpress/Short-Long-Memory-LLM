unit Ban.Signatures;

(* The cheap prefilter: weighted regular expressions over a request, like a
   fail2ban filter, but here they only decide which requests are worth
   asking the decision model about (and, in rules mode or as a fallback,
   they decide alone). Weights by convention:
     10  never legitimate (/.env, ../, ${jndi:, union select, sqlmap)
      6  almost always a probe (phpMyAdmin, phpinfo.php, webshell names)
      3  depends on the site (wp-login.php on a WordPress site is fine)
   Matching is case-insensitive. A signatures file adds rules, one per line:
     <weight> <field> <tag> <regex>
   field: url (decoded path + query), request (the logged request line),
   agent (User-Agent), any (url + agent + referer); '#' starts a comment. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, RegExpr, Ban.Apache;

type
  TSigField = (sfUrl, sfRequest, sfAgent, sfAny);

  TSignature = class
    Tag: string;
    Weight: Integer;
    Field: TSigField;
    Re: TRegExpr;
    destructor Destroy; override;
  end;

  TSignatureSet = class
  private
    FItems: TList;
    procedure AddRule(AWeight: Integer; AField: TSigField; const ATag, ARegex: string);
  public
    constructor Create(ABuiltin: Boolean = True);
    destructor Destroy; override;
    procedure LoadFile(const AFile: string);
    { total weight of the matching rules; their tags go to ATags }
    function Score(const E: TAccessEntry; ATags: TStrings): Integer;
    function Count: Integer;
  end;

implementation

destructor TSignature.Destroy;
begin
  Re.Free;
  inherited Destroy;
end;

constructor TSignatureSet.Create(ABuiltin: Boolean);
begin
  inherited Create;
  FItems := TList.Create;
  if not ABuiltin then
    Exit;
  { secrets and repositories left in the web root }
  AddRule(10, sfUrl, 'env-file', '/\.env(\.[a-z]+)?($|[?/])');
  AddRule(10, sfUrl, 'vcs-dir', '/\.(git|svn|hg|bzr)(/|$)');
  AddRule(10, sfUrl, 'secrets', '/(\.aws/|\.ssh/|\.docker/|\.htpasswd|\.npmrc|\.pypirc|\.netrc|id_rsa|id_dsa|' +
    '\.ftpconfig|sftp-config\.json|\.vscode/sftp\.json|\.s3cfg|\.boto)');
  AddRule(6, sfUrl, 'config-file', '/(wp-config\.php[.~_-]|configuration\.php[.~_-]|config\.php[.~_-]|' +
    'settings\.php[.~_-]|web\.config|appsettings(\.[a-z]+)?\.json|docker-compose\.ya?ml|\.travis\.yml|' +
    '\.DS_Store|\.htaccess|server-status|server-info)');
  AddRule(3, sfUrl, 'backup-file', '\.(sql|bak|old|orig|swp|save|backup|tar|tgz|tar\.gz|zip|rar|7z)$');
  { path traversal and system files }
  AddRule(10, sfUrl, 'traversal', '\.\.[/\\]');
  AddRule(10, sfUrl, 'system-file', '(/etc/(passwd|shadow|hosts|group)|/proc/self/|win\.ini|boot\.ini|' +
    'c:\\windows)');
  { code execution }
  AddRule(10, sfAny, 'log4shell', '\$\{(jndi|env|lower|upper|sys|java|::-)');
  AddRule(10, sfUrl, 'shell-inject', '(;|\||`|\$\(|&&)\s*(wget|curl|sh|bash|nc|ncat|chmod|busybox|tftp|' +
    'echo|cat|rm|perl|python)\b');
  AddRule(10, sfUrl, 'php-rce', '(allow_url_include|auto_prepend_file|php://(input|filter)|data://text|' +
    'eval-stdin\.php|/vendor/phpunit/|call_user_func_array|invokefunction|_ignition/execute-solution|' +
    'think\\app|base64_decode\()');
  AddRule(6, sfUrl, 'webshell', '/(shell|cmd|c99|r57|wso|alfa|b374k|indoxploit|xleet|leaf|' +
    'mini|up|uploader|webshell|bypass|0x|404|wp-sx|ws)\.php');
  AddRule(10, sfUrl, 'device-exploit', '/(HNAP1|boaform|GponForm|setup\.cgi|tmUnblock\.cgi|apply\.cgi|' +
    'goform/|picsdesc\.xml|ctrlt/DeviceUpgrade|device\.rsp|login\.cgi\?cli|shell\?cd|cgi-bin/luci/;|' +
    'mgmt/tm/util/bash|hudson/script|cgi-bin/.*\.(sh|cgi)\?)');
  AddRule(10, sfUrl, 'sqli', '(union(\s|/\*.*\*/)+(all(\s|/\*.*\*/)+)?select|information_schema|' +
    'sleep\(\s*\d|benchmark\(|waitfor\s+delay|extractvalue\(|updatexml\(|' +
    '''\s*(or|and)\s+''?\d+''?\s*=\s*''?\d)');
  AddRule(6, sfUrl, 'xss', '(<script|javascript:|onerror\s*=|onload\s*=|<svg|<iframe|alert\()');
  { admin panels nobody should be guessing }
  AddRule(6, sfUrl, 'db-admin', '/(phpmyadmin|pma|myadmin|mysqladmin|sqladmin|adminer|dbadmin|' +
    'phpmyadmin[^/]*)(/|\.php|$)');
  AddRule(6, sfUrl, 'phpinfo', '/(phpinfo|info|test|i|php_info|pinfo|php)\.php($|\?)');
  AddRule(6, sfUrl, 'java-admin', '/(manager/html|host-manager|jmx-console|web-console|invoker/|' +
    'console/login|actuator/(env|heapdump|gateway)|solr/admin|jenkins/script|struts2?-)');
  { probes that are only suspicious when the site does not run the software }
  AddRule(3, sfUrl, 'wordpress', '/(wp-login\.php|xmlrpc\.php|wp-admin/|wp-includes/|wp-content/plugins/|' +
    'wlwmanifest\.xml|wp-json/wp/v2/users)');
  AddRule(3, sfUrl, 'login-probe', '/(administrator/index\.php|admin\.php|admin/login|user/login|' +
    'login\.action|owa/|ecp/|autodiscover/autodiscover|remote/login|dana-na/|global-protect/|' +
    '\+CSCOE\+|vpn/)');
  AddRule(3, sfUrl, 'cgi-bin', '/cgi-bin/');
  { odd requests }
  AddRule(6, sfRequest, 'proxy-probe', '^(CONNECT\s|[A-Z]+\s+https?://)');
  AddRule(6, sfRequest, 'odd-method', '^(PROPFIND|TRACE|TRACK|DEBUG|MKCOL|SEARCH)\s');
  AddRule(6, sfRequest, 'binary-request', '\\x[0-9a-f][0-9a-f]');
  { tools }
  AddRule(10, sfAgent, 'scanner-agent', '(sqlmap|nikto|nmap|masscan|zgrab|nuclei|wpscan|dirbuster|' +
    'gobuster|ffuf|feroxbuster|acunetix|nessus|openvas|netsparker|w3af|whatweb|jaeles|fuzz faster|' +
    'l9explore|l9tcpid|zmeu|morfeus|jorgee|hydra|commix|xsser|wfuzz)');
  AddRule(3, sfAgent, 'script-agent', '^(python-requests|python-urllib|python-httpx|go-http-client|' +
    'curl/|wget/|libwww-perl|java/|okhttp|axios/|node-fetch|aiohttp|-$|$)');
end;

destructor TSignatureSet.Destroy;
var
  I: Integer;
begin
  for I := 0 to FItems.Count - 1 do
    TSignature(FItems[I]).Free;
  FItems.Free;
  inherited Destroy;
end;

procedure TSignatureSet.AddRule(AWeight: Integer; AField: TSigField; const ATag, ARegex: string);
var
  LSig: TSignature;
begin
  LSig := TSignature.Create;
  LSig.Tag := ATag;
  LSig.Weight := AWeight;
  LSig.Field := AField;
  LSig.Re := TRegExpr.Create;
  try
    LSig.Re.ModifierI := True;
    LSig.Re.Expression := ARegex;
    LSig.Re.Compile;
  except
    on E: Exception do
    begin
      LSig.Free;
      raise Exception.CreateFmt('signature %s: bad regex %s (%s)', [ATag, ARegex, E.Message]);
    end;
  end;
  FItems.Add(LSig);
end;

procedure TSignatureSet.LoadFile(const AFile: string);
var
  LLines: TStringList;
  LLine, LWeight, LField, LTag: string;
  LSp, LNum, LW: Integer;
  LKind: TSigField;

  function Take(var S: string): string;
  begin
    S := TrimLeft(S);
    LSp := Pos(' ', S);
    if LSp = 0 then
      LSp := Length(S) + 1;
    Result := Copy(S, 1, LSp - 1);
    S := TrimLeft(Copy(S, LSp + 1, MaxInt));
  end;

begin
  LLines := TStringList.Create;
  try
    LLines.LoadFromFile(AFile);
    for LNum := 0 to LLines.Count - 1 do
    begin
      LLine := Trim(StringReplace(LLines[LNum], #9, ' ', [rfReplaceAll]));
      if (LLine = '') or (LLine[1] = '#') then
        Continue;
      LWeight := Take(LLine);
      LField := LowerCase(Take(LLine));
      LTag := Take(LLine);
      if not TryStrToInt(LWeight, LW) or (LTag = '') or (LLine = '') then
        raise Exception.CreateFmt('%s:%d: expected "<weight> <field> <tag> <regex>"', [AFile, LNum + 1]);
      if LField = 'url' then
        LKind := sfUrl
      else if LField = 'request' then
        LKind := sfRequest
      else if LField = 'agent' then
        LKind := sfAgent
      else if LField = 'any' then
        LKind := sfAny
      else
        raise Exception.CreateFmt('%s:%d: field must be url, request, agent or any', [AFile, LNum + 1]);
      AddRule(LW, LKind, LTag, LLine);
    end;
  finally
    LLines.Free;
  end;
end;

function TSignatureSet.Score(const E: TAccessEntry; ATags: TStrings): Integer;
var
  I: Integer;
  LSig: TSignature;
  LText, LAny: string;
begin
  Result := 0;
  LAny := E.Decoded + #10 + E.Agent + #10 + E.Referer;
  for I := 0 to FItems.Count - 1 do
  begin
    LSig := TSignature(FItems[I]);
    case LSig.Field of
      sfUrl: LText := E.Decoded;
      sfRequest: LText := E.Request;
      sfAgent: LText := E.Agent;
    else
      LText := LAny;
    end;
    if LSig.Re.Exec(LText) then
    begin
      Inc(Result, LSig.Weight);
      if ATags <> nil then
        ATags.Add(LSig.Tag);
    end;
  end;
end;

function TSignatureSet.Count: Integer;
begin
  Result := FItems.Count;
end;

end.
