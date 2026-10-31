unit Ban.Config;

(* jevban.ini, then the command line on top. The keys follow fail2ban's
   names where there is one (logpath, ignoreip, findtime, maxretry, bantime,
   bantime.increment, actionban, ...). Lookup order for the file: --config,
   /etc/jevban/jevban.ini, jevban.ini next to the executable; none at all is
   fine (the defaults below). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, IniFiles, Ban.Net;

type
  TBanConfig = class
  public
    ConfigFile: string;
    { [jevban] }
    LogPaths: TStringList;
    FromStart: Boolean;
    PollMs: Integer;
    StateFile: string;
    LogFile: string;
    IgnoreIp: TCidrList;
    IgnoreIpText: string;
    FindTime: Int64;
    MaxRetry: Integer;
    BanTime: Int64;
    BanIncrement: Boolean;    { each repeat ban lasts twice as long ... }
    BanMax: Int64;            { ... up to this }
    ContextLines: Integer;    { recent requests per client kept for the model }
    MaxClients: Integer;
    Flood4xx: Integer;        { this many 4xx in findtime is suspicious too; 0 = off }
    SignatureFile: string;
    BuiltinSignatures: Boolean;
    SuspiciousScore: Integer; { prefilter score that sends a request to the decider }
    Verbose: Boolean;
    { [decider] }
    Mode: string;             { jev | rules }
    JevUrl: string;
    JevKey: string;
    JevModel: string;
    TimeoutMs: Integer;
    Threshold: Double;        { P(malicious) that counts as a strike }
    InstantThreshold: Double; { P(malicious) that bans at once }
    CacheSecs: Integer;       { reuse a client's verdict this long }
    MaxPerMinute: Integer;    { decider calls; beyond it, rules decide }
    OnError: string;          { rules | skip }
    RulesMinScore: Integer;   { rules: a strike }
    RulesInstantScore: Integer;
    OverrideScore: Integer;   { jev mode: a score this high is a strike whatever the model says; 0 = off }
    Site: string;
    Question: string;
    { [action] }
    ActionType: string;
    ActionName: string;
    Port: string;
    Jail: string;
    ActionStart, ActionStop, ActionBan, ActionUnban: string;
    DryRun: Boolean;
    CleanupOnExit: Boolean;
    { command line only }
    TestMode: Boolean;
    ListBans: Boolean;
    constructor Create;
    destructor Destroy; override;
    { parses the command line; raises EArgumentException on a bad option.
      Returns False when --help/--version printed and the program should exit }
    function Load: Boolean;
    procedure Validate;
  end;

procedure PrintUsage;

const
  CVersion = '1.0.0';

implementation

uses
  Ban.Jev;

procedure PrintUsage;
begin
  WriteLn('jevban ', CVersion, ' - ban web attackers found in Apache logs, decided by a Jev-style AI');
  WriteLn;
  WriteLn('usage: jevban [options]');
  WriteLn('  -c, --config FILE      settings (default /etc/jevban/jevban.ini, else next to the binary)');
  WriteLn('  -l, --log FILE         access log or wildcard pattern to follow (repeatable; replaces logpath)');
  WriteLn('      --from-start       read the logs from their start, not just new lines');
  WriteLn('  -t, --test FILE        read FILE once, print each decision, ban nothing (like fail2ban-regex)');
  WriteLn('  -n, --dry-run          log the firewall commands instead of running them');
  WriteLn('  -m, --mode MODE        jev (ask the decision model) | rules (signatures only)');
  WriteLn('  -u, --jev-url URL      decision endpoint (default http://127.0.0.1:3000/v1/systemone)');
  WriteLn('  -a, --action TYPE      iptables | nftables | ufw | firewalld | fail2ban | netsh | command | none');
  WriteLn('      --list             print the bans in the state file and exit');
  WriteLn('  -v, --verbose          log every request the prefilter flags, and every verdict');
  WriteLn('  -h, --help             this text');
  WriteLn('      --version');
  WriteLn;
  WriteLn('environment: JEV_API_KEY (bearer token for the decision API), HTTPS_PROXY, HTTP_PROXY, NO_PROXY');
end;

constructor TBanConfig.Create;
begin
  inherited Create;
  LogPaths := TStringList.Create;
  {$IFDEF WINDOWS}
  StateFile := ExtractFilePath(ParamStr(0)) + 'jevban-state.json';
  {$ELSE}
  StateFile := '/var/lib/jevban/state.json';
  {$ENDIF}
  PollMs := 250;
  IgnoreIpText := '127.0.0.0/8 ::1';
  FindTime := 600;
  MaxRetry := 3;
  BanTime := 3600;
  BanIncrement := True;
  BanMax := 7 * 86400;
  ContextLines := 6;
  MaxClients := 20000;
  Flood4xx := 40;
  BuiltinSignatures := True;
  SuspiciousScore := 3;
  Mode := 'jev';
  JevUrl := 'http://127.0.0.1:3000/v1/systemone';
  JevModel := 'openjev';
  TimeoutMs := 8000;
  Threshold := 0.80;
  InstantThreshold := 0.97;
  CacheSecs := 60;
  MaxPerMinute := 120;
  OnError := 'rules';
  RulesMinScore := 6;
  RulesInstantScore := 20;
  OverrideScore := 0;
  Site := CDefaultSite;
  Question := CDefaultQuestion;
  {$IFDEF WINDOWS}
  ActionType := 'netsh';
  {$ELSE}
  ActionType := 'iptables';
  {$ENDIF}
  ActionName := 'jevban';
  Port := '80,443';
  Jail := 'jevban';
  CleanupOnExit := True;
end;

destructor TBanConfig.Destroy;
begin
  LogPaths.Free;
  inherited Destroy;
end;

function SplitList(const S: string; ADest: TStrings): Integer;
var
  LItems: TStringList;
  LItem: string;
begin
  LItems := TStringList.Create;
  try
    LItems.StrictDelimiter := True;
    LItems.Delimiter := ',';
    LItems.DelimitedText := StringReplace(S, LineEnding, ',', [rfReplaceAll]);
    for LItem in LItems do
      if Trim(LItem) <> '' then
        ADest.Add(Trim(LItem));
  finally
    LItems.Free;
  end;
  Result := ADest.Count;
end;

function TBanConfig.Load: Boolean;
var
  I: Integer;
  LArg, LVal: string;
  LIni: TMemIniFile;
  LCliLogs: TStringList;
  LCli: TStringList;   { name=value from the command line, applied after the file }

  function NextVal: string;
  begin
    if I >= ParamCount then
      raise EArgumentException.CreateFmt('%s needs a value', [LArg]);
    Inc(I);
    Result := ParamStr(I);
  end;

  { probabilities are written 0.85 whatever the locale }
  function F(const AKey: string; ADef: Double): Double;
  var
    LFmt: TFormatSettings;
  begin
    LFmt := DefaultFormatSettings;
    LFmt.DecimalSeparator := '.';
    Result := StrToFloatDef(Trim(LIni.ReadString('decider', AKey, '')), ADef, LFmt);
  end;

begin
  Result := True;
  LCliLogs := TStringList.Create;
  LCli := TStringList.Create;
  try
    I := 1;
    while I <= ParamCount do
    begin
      LArg := ParamStr(I);
      if (LArg = '-h') or (LArg = '--help') then
      begin
        PrintUsage;
        Exit(False);
      end
      else if LArg = '--version' then
      begin
        WriteLn('jevban ', CVersion);
        Exit(False);
      end
      else if (LArg = '-c') or (LArg = '--config') then
        ConfigFile := NextVal
      else if (LArg = '-l') or (LArg = '--log') then
        LCliLogs.Add(NextVal)
      else if LArg = '--from-start' then
        LCli.Values['from_start'] := '1'
      else if (LArg = '-t') or (LArg = '--test') then
      begin
        LCliLogs.Add(NextVal);
        TestMode := True;
      end
      else if (LArg = '-n') or (LArg = '--dry-run') then
        LCli.Values['dry_run'] := '1'
      else if (LArg = '-m') or (LArg = '--mode') then
        LCli.Values['mode'] := NextVal
      else if (LArg = '-u') or (LArg = '--jev-url') then
        LCli.Values['url'] := NextVal
      else if (LArg = '-a') or (LArg = '--action') then
        LCli.Values['type'] := NextVal
      else if LArg = '--list' then
        ListBans := True
      else if (LArg = '-v') or (LArg = '--verbose') then
        Verbose := True
      else
        raise EArgumentException.CreateFmt('unknown option %s (try --help)', [LArg]);
      Inc(I);
    end;

    if ConfigFile = '' then
    begin
      {$IFNDEF WINDOWS}
      if FileExists('/etc/jevban/jevban.ini') then
        ConfigFile := '/etc/jevban/jevban.ini'
      else
      {$ENDIF}
      if FileExists(ExtractFilePath(ParamStr(0)) + 'jevban.ini') then
        ConfigFile := ExtractFilePath(ParamStr(0)) + 'jevban.ini';
    end
    else if not FileExists(ConfigFile) then
      raise EArgumentException.CreateFmt('config file %s not found', [ConfigFile]);

    LIni := TMemIniFile.Create(ConfigFile);   { '' = empty, all defaults }
    try
      SplitList(LIni.ReadString('jevban', 'logpath', ''), LogPaths);
      FromStart := LIni.ReadBool('jevban', 'from_start', FromStart);
      PollMs := LIni.ReadInteger('jevban', 'poll_ms', PollMs);
      StateFile := LIni.ReadString('jevban', 'statefile', StateFile);
      LogFile := LIni.ReadString('jevban', 'logfile', LogFile);
      IgnoreIpText := LIni.ReadString('jevban', 'ignoreip', IgnoreIpText);
      FindTime := LIni.ReadInt64('jevban', 'findtime', FindTime);
      MaxRetry := LIni.ReadInteger('jevban', 'maxretry', MaxRetry);
      BanTime := LIni.ReadInt64('jevban', 'bantime', BanTime);
      BanIncrement := LIni.ReadBool('jevban', 'bantime.increment', BanIncrement);
      BanMax := LIni.ReadInt64('jevban', 'bantime.maxtime', BanMax);
      ContextLines := LIni.ReadInteger('jevban', 'context_lines', ContextLines);
      MaxClients := LIni.ReadInteger('jevban', 'max_clients', MaxClients);
      Flood4xx := LIni.ReadInteger('jevban', 'flood_4xx', Flood4xx);
      SignatureFile := LIni.ReadString('jevban', 'signatures', SignatureFile);
      BuiltinSignatures := LIni.ReadBool('jevban', 'builtin_signatures', BuiltinSignatures);
      SuspiciousScore := LIni.ReadInteger('jevban', 'suspicious_score', SuspiciousScore);
      Verbose := LIni.ReadBool('jevban', 'verbose', Verbose);

      Mode := LowerCase(LIni.ReadString('decider', 'mode', Mode));
      JevUrl := LIni.ReadString('decider', 'url', JevUrl);
      JevKey := LIni.ReadString('decider', 'api_key', JevKey);
      JevModel := LIni.ReadString('decider', 'model', JevModel);
      TimeoutMs := LIni.ReadInteger('decider', 'timeout_ms', TimeoutMs);
      Threshold := F('threshold', Threshold);
      InstantThreshold := F('instant_threshold', InstantThreshold);
      CacheSecs := LIni.ReadInteger('decider', 'cache_secs', CacheSecs);
      MaxPerMinute := LIni.ReadInteger('decider', 'max_per_minute', MaxPerMinute);
      OnError := LowerCase(LIni.ReadString('decider', 'on_error', OnError));
      RulesMinScore := LIni.ReadInteger('decider', 'rules_min_score', RulesMinScore);
      RulesInstantScore := LIni.ReadInteger('decider', 'rules_instant_score', RulesInstantScore);
      OverrideScore := LIni.ReadInteger('decider', 'override_score', OverrideScore);
      Site := LIni.ReadString('decider', 'site', Site);
      Question := LIni.ReadString('decider', 'question', Question);

      ActionType := LowerCase(LIni.ReadString('action', 'type', ActionType));
      ActionName := LIni.ReadString('action', 'name', ActionName);
      Port := LIni.ReadString('action', 'port', Port);
      Jail := LIni.ReadString('action', 'jail', Jail);
      ActionStart := LIni.ReadString('action', 'actionstart', '');
      ActionStop := LIni.ReadString('action', 'actionstop', '');
      ActionBan := LIni.ReadString('action', 'actionban', '');
      ActionUnban := LIni.ReadString('action', 'actionunban', '');
      DryRun := LIni.ReadBool('action', 'dry_run', DryRun);
      CleanupOnExit := LIni.ReadBool('action', 'cleanup_on_exit', CleanupOnExit);
    finally
      LIni.Free;
    end;

    { the command line wins }
    if LCliLogs.Count > 0 then
      LogPaths.Assign(LCliLogs);
    LVal := LCli.Values['from_start'];
    if LVal <> '' then
      FromStart := True;
    if LCli.Values['dry_run'] <> '' then
      DryRun := True;
    if LCli.Values['mode'] <> '' then
      Mode := LowerCase(LCli.Values['mode']);
    if LCli.Values['url'] <> '' then
      JevUrl := LCli.Values['url'];
    if LCli.Values['type'] <> '' then
      ActionType := LowerCase(LCli.Values['type']);
    if JevKey = '' then
      JevKey := GetEnvironmentVariable('JEV_API_KEY');
    if TestMode then
    begin
      FromStart := True;
      DryRun := True;
    end;
  finally
    LCliLogs.Free;
    LCli.Free;
  end;
end;

procedure TBanConfig.Validate;

  function Safe(const S: string; const AExtra: TSysCharSet): Boolean;
  var
    C: Char;
  begin
    Result := S <> '';
    for C in S do
      if not (C in ['A'..'Z', 'a'..'z', '0'..'9', '_', '-'] + AExtra) then
        Exit(False);
  end;

begin
  IgnoreIp := ParseCidrList(IgnoreIpText);
  if (LogPaths.Count = 0) and not ListBans then
    raise EArgumentException.Create('no log to follow: set logpath in [jevban] or pass --log');
  if (Mode <> 'jev') and (Mode <> 'rules') then
    raise EArgumentException.CreateFmt('mode must be jev or rules, not "%s"', [Mode]);
  if (OnError <> 'rules') and (OnError <> 'skip') then
    raise EArgumentException.CreateFmt('on_error must be rules or skip, not "%s"', [OnError]);
  if (Threshold <= 0) or (Threshold > 1) or (InstantThreshold <= 0) or (InstantThreshold > 1) then
    raise EArgumentException.Create('threshold and instant_threshold are probabilities in (0, 1]');
  if (MaxRetry < 1) or (FindTime < 1) or (BanTime < 1) then
    raise EArgumentException.Create('maxretry, findtime and bantime must be positive');
  if ContextLines < 1 then
    ContextLines := 1;
  if PollMs < 20 then
    PollMs := 20;
  { these go into shell commands; they come from root's config, but a typo
    should fail here, not as a half-applied firewall rule }
  if not Safe(ActionName, []) then
    raise EArgumentException.Create('action name: letters, digits, _ and - only');
  if not Safe(Jail, []) then
    raise EArgumentException.Create('action jail: letters, digits, _ and - only');
  if not Safe(Port, [',', ':', ' ']) then
    raise EArgumentException.Create('action port: e.g. 80,443');
end;

end.
