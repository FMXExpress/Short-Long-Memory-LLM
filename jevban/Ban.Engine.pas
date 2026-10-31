unit Ban.Engine;

(* One access-log line at a time:

     parse -> ignoreip? -> already banned? -> signature score (+ 4xx flood)
       -> below suspicious_score: just remember it as context
       -> otherwise ask the decider about this client:
            jev:   POST the client's recent requests to /v1/systemone;
                   P(malicious) >= threshold is a strike,
                   >= instant_threshold bans at once
            rules: score >= rules_min_score is a strike,
                   >= rules_instant_score bans at once
       -> maxretry strikes within findtime: ban for bantime (doubling for
          repeat offenders when bantime.increment is on, up to maxtime)

   A verdict is reused for cache_secs, so a scanner firing 200 probes costs
   one model call, not 200; past max_per_minute calls (a flood of distinct
   clients) the rules decide. Bans survive a restart in the state file. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, fpjson, jsonparser,
  Ban.Net, Ban.Apache, Ban.Signatures, Ban.Jev, Ban.Firewall, Ban.Config;

type
  TClient = class
  public
    IpText: string;
    Ip: TIpAddr;
    Recent: TStringList;     { request summaries, oldest first }
    Strikes: array of Int64;
    LastSeen: Int64;
    FloodStart: Int64;
    FloodCount: Integer;
    VerdictAt: Int64;        { 0 = no verdict yet }
    VerdictMal: Double;
    VerdictIntent: string;
    constructor Create;
    destructor Destroy; override;
  end;

  TBanRecord = class
  public
    Ip: TIpAddr;
    IpText: string;
    Since: Int64;
    Expires: Int64;
    Count: Integer;          { bans so far: bantime.increment }
    Active: Boolean;
    Reason: string;
  end;

  TBanStats = record
    Lines, Parsed, Ignored, Suspicious, DeciderCalls, DeciderErrors, Strikes, Bans, Unbans: Int64;
  end;

  TBanEngine = class
  private
    FCfg: TBanConfig;
    FSigs: TSignatureSet;
    FJev: TJevClient;
    FAction: TBanAction;
    FClients: TStringList;   { ip -> TClient }
    FBans: TStringList;      { ip -> TBanRecord }
    FMinute: Int64;
    FMinuteCalls: Integer;
    FLastPurge: Int64;
    FLastErrLog: Int64;
    FErrsSinceLog: Integer;
    FDirty: Boolean;
    FLog: TFileStream;
    function GetClient(const E: TAccessEntry; ANow: Int64): TClient;
    procedure Judge(C: TClient; const E: TAccessEntry; const ASummary: string; AScore: Integer;
      ATags: TStrings; ANow: Int64);
    procedure AddStrike(C: TClient; ANow: Int64; AInstant: Boolean; const AReason: string);
    procedure BanClient(C: TClient; const AReason: string);
    procedure Purge(ANow: Int64);
    procedure LoadState;
    procedure SaveState;
  public
    Stats: TBanStats;
    constructor Create(ACfg: TBanConfig);
    destructor Destroy; override;
    procedure Log(const AMsg: string);
    { state, actionstart, re-ban what is still banned }
    procedure Start;
    { state, actionstop when cleanup_on_exit }
    procedure Stop;
    procedure Feed(const ALine: string);
    { unbans the expired, forgets idle clients, saves the state }
    procedure Tick;
    procedure PrintBans;
    function Summary: string;
  end;

implementation

{ TClient }

constructor TClient.Create;
begin
  inherited Create;
  Recent := TStringList.Create;
end;

destructor TClient.Destroy;
begin
  Recent.Free;
  inherited Destroy;
end;

{ TBanEngine }

constructor TBanEngine.Create(ACfg: TBanConfig);
begin
  inherited Create;
  FCfg := ACfg;
  FSigs := TSignatureSet.Create(FCfg.BuiltinSignatures);
  if FCfg.SignatureFile <> '' then
    FSigs.LoadFile(FCfg.SignatureFile);
  FJev := TJevClient.Create;
  FJev.Url := FCfg.JevUrl;
  FJev.ApiKey := FCfg.JevKey;
  FJev.Model := FCfg.JevModel;
  FJev.Site := FCfg.Site;
  FJev.Question := FCfg.Question;
  FJev.TimeoutMs := FCfg.TimeoutMs;
  FAction := TBanAction.Create(@Log);
  FAction.Kind := FCfg.ActionType;
  FAction.Name := FCfg.ActionName;
  FAction.Port := FCfg.Port;
  FAction.Jail := FCfg.Jail;
  FAction.StartCmd := FCfg.ActionStart;
  FAction.StopCmd := FCfg.ActionStop;
  FAction.BanCmd := FCfg.ActionBan;
  FAction.UnbanCmd := FCfg.ActionUnban;
  FAction.DryRun := FCfg.DryRun;
  FAction.ApplyPreset;
  FClients := TStringList.Create;
  FClients.Sorted := True;
  FClients.OwnsObjects := True;
  FBans := TStringList.Create;
  FBans.Sorted := True;
  FBans.OwnsObjects := True;
  if (FCfg.LogFile <> '') and not FCfg.TestMode then
  begin
    ForceDirectories(ExtractFileDir(ExpandFileName(FCfg.LogFile)));
    if FileExists(FCfg.LogFile) then
      FLog := TFileStream.Create(FCfg.LogFile, fmOpenWrite or fmShareDenyNone)
    else
      FLog := TFileStream.Create(FCfg.LogFile, fmCreate);
    FLog.Seek(0, soEnd);
  end;
end;

destructor TBanEngine.Destroy;
begin
  FLog.Free;
  FBans.Free;
  FClients.Free;
  FAction.Free;
  FJev.Free;
  FSigs.Free;
  inherited Destroy;
end;

procedure TBanEngine.Log(const AMsg: string);
var
  LLine: string;
begin
  LLine := FormatDateTime('yyyy-mm-dd hh:nn:ss', Now) + ' jevban: ' + AMsg;
  WriteLn(LLine);
  Flush(Output);
  if FLog <> nil then
  begin
    LLine := LLine + LineEnding;
    FLog.WriteBuffer(LLine[1], Length(LLine));
  end;
end;

function TBanEngine.GetClient(const E: TAccessEntry; ANow: Int64): TClient;
var
  I, LOldest: Integer;
begin
  I := FClients.IndexOf(E.IpText);
  if I >= 0 then
    Exit(TClient(FClients.Objects[I]));
  if FClients.Count >= FCfg.MaxClients then
  begin
    Purge(ANow);
    if FClients.Count >= FCfg.MaxClients then
    begin
      { still full: drop the client seen longest ago }
      LOldest := 0;
      for I := 1 to FClients.Count - 1 do
        if TClient(FClients.Objects[I]).LastSeen < TClient(FClients.Objects[LOldest]).LastSeen then
          LOldest := I;
      FClients.Delete(LOldest);
    end;
  end;
  Result := TClient.Create;
  Result.IpText := E.IpText;
  Result.Ip := E.Ip;
  FClients.AddObject(E.IpText, Result);
end;

procedure TBanEngine.Feed(const ALine: string);
var
  E: TAccessEntry;
  C: TClient;
  LNow: Int64;
  LTags: TStringList;
  LScore, I: Integer;
  LSummary: string;
begin
  Inc(Stats.Lines);
  if not ParseAccessLine(ALine, E) then
  begin
    if FCfg.Verbose then
      Log('unparsed: ' + Copy(ALine, 1, 200));
    Exit;
  end;
  Inc(Stats.Parsed);
  if InCidrList(FCfg.IgnoreIp, E.Ip) then
  begin
    Inc(Stats.Ignored);
    Exit;
  end;
  { replaying a file (--test) runs on the log's clock, live on the wall clock }
  if FCfg.TestMode and (E.Epoch > 0) then
    LNow := E.Epoch
  else
    LNow := UnixNow;
  I := FBans.IndexOf(E.IpText);
  if (I >= 0) and TBanRecord(FBans.Objects[I]).Active then
    Exit;   { already banned: a connection that was open before the ban }
  C := GetClient(E, LNow);
  C.LastSeen := LNow;
  LSummary := EntrySummary(E);
  LTags := TStringList.Create;
  try
    LTags.Duplicates := dupIgnore;
    LScore := FSigs.Score(E, LTags);
    if (FCfg.Flood4xx > 0) and (E.Status >= 400) and (E.Status < 500) then
    begin
      if LNow - C.FloodStart > FCfg.FindTime then
      begin
        C.FloodStart := LNow;
        C.FloodCount := 0;
      end;
      Inc(C.FloodCount);
      if C.FloodCount = FCfg.Flood4xx then
      begin
        Inc(LScore, 6);
        LTags.Add('4xx-flood');
      end;
    end;
    if LScore >= FCfg.SuspiciousScore then
    begin
      Inc(Stats.Suspicious);
      Judge(C, E, LSummary, LScore, LTags, LNow);
    end;
  finally
    LTags.Free;
  end;
  C.Recent.Add(LSummary);
  while C.Recent.Count > FCfg.ContextLines do
    C.Recent.Delete(0);
end;

procedure TBanEngine.Judge(C: TClient; const E: TAccessEntry; const ASummary: string; AScore: Integer;
  ATags: TStrings; ANow: Int64);
var
  V: TJevVerdict;
  LHow, LVerdict, LTags: string;
  LMal: Double;
  LIntent: string;
  LStrike, LInstant, LUseRules: Boolean;
begin
  LTags := ATags.CommaText;
  LMal := -1;
  LIntent := '';
  LUseRules := FCfg.Mode = 'rules';
  LHow := 'rules';
  if not LUseRules then
  begin
    if (C.VerdictAt > 0) and (ANow - C.VerdictAt < FCfg.CacheSecs) then
    begin
      LMal := C.VerdictMal;
      LIntent := C.VerdictIntent;
      LHow := 'jev (cached)';
    end
    else
    begin
      if ANow div 60 <> FMinute then
      begin
        FMinute := ANow div 60;
        FMinuteCalls := 0;
      end;
      if FMinuteCalls >= FCfg.MaxPerMinute then
      begin
        LUseRules := True;
        LHow := 'rules (decider rate limit)';
      end
      else
      begin
        Inc(FMinuteCalls);
        Inc(Stats.DeciderCalls);
        V := FJev.Decide(C.IpText, E.Request, LTags, C.Recent);
        if V.Ok then
        begin
          LMal := V.Malicious;
          LIntent := V.Intent;
          C.VerdictAt := ANow;
          C.VerdictMal := LMal;
          C.VerdictIntent := LIntent;
          LHow := Format('jev %dms', [V.LatencyMs]);
        end
        else
        begin
          Inc(Stats.DeciderErrors);
          Inc(FErrsSinceLog);
          if UnixNow - FLastErrLog >= 60 then
          begin
            Log(Format('decider error (%d since last report): %s', [FErrsSinceLog, V.Error]));
            FLastErrLog := UnixNow;
            FErrsSinceLog := 0;
          end;
          if FCfg.OnError = 'rules' then
          begin
            LUseRules := True;
            LHow := 'rules (decider unavailable)';
          end
          else
            LHow := 'skipped (decider unavailable)';
        end;
      end;
    end;
  end;

  LStrike := False;
  LInstant := False;
  if LUseRules then
  begin
    LStrike := AScore >= FCfg.RulesMinScore;
    LInstant := AScore >= FCfg.RulesInstantScore;
  end
  else if LMal >= 0 then
  begin
    LStrike := LMal >= FCfg.Threshold;
    LInstant := LMal >= FCfg.InstantThreshold;
    if (FCfg.OverrideScore > 0) and (AScore >= FCfg.OverrideScore) and not LStrike then
    begin
      LStrike := True;
      LHow := LHow + ', score override';
    end;
  end;

  if LInstant then
    LVerdict := 'BAN'
  else if LStrike then
    LVerdict := Format('strike %d/%d', [Length(C.Strikes) + 1, FCfg.MaxRetry])
  else
    LVerdict := 'pass';
  if FCfg.Verbose or FCfg.TestMode or LStrike then
  begin
    if LMal >= 0 then
      Log(Format('%s %s score=%d [%s] p=%.2f intent=%s via %s: %s', [C.IpText, LVerdict, AScore,
        LTags, LMal, LIntent, LHow, Copy(ASummary, 1, 300)]))
    else
      Log(Format('%s %s score=%d [%s] via %s: %s', [C.IpText, LVerdict, AScore, LTags, LHow,
        Copy(ASummary, 1, 300)]));
  end;
  if LStrike then
    if LMal >= 0 then
      AddStrike(C, ANow, LInstant, Format('%s, p=%.2f %s', [LTags, LMal, LIntent]))
    else
      AddStrike(C, ANow, LInstant, Format('%s, score %d', [LTags, AScore]));
end;

procedure TBanEngine.AddStrike(C: TClient; ANow: Int64; AInstant: Boolean; const AReason: string);
var
  I, J: Integer;
begin
  Inc(Stats.Strikes);
  J := 0;
  for I := 0 to High(C.Strikes) do
    if ANow - C.Strikes[I] < FCfg.FindTime then
    begin
      C.Strikes[J] := C.Strikes[I];
      Inc(J);
    end;
  SetLength(C.Strikes, J + 1);
  C.Strikes[J] := ANow;
  if AInstant or (Length(C.Strikes) >= FCfg.MaxRetry) then
    BanClient(C, AReason);
end;

procedure TBanEngine.BanClient(C: TClient; const AReason: string);
var
  I: Integer;
  B: TBanRecord;
  LSecs, LNow: Int64;
begin
  I := FBans.IndexOf(C.IpText);
  if I >= 0 then
    B := TBanRecord(FBans.Objects[I])
  else
  begin
    B := TBanRecord.Create;
    B.Ip := C.Ip;
    B.IpText := C.IpText;
    FBans.AddObject(C.IpText, B);
  end;
  if B.Active then
    Exit;
  LSecs := FCfg.BanTime;
  if FCfg.BanIncrement and (B.Count > 0) then
    LSecs := Min(FCfg.BanMax, Round(FCfg.BanTime * Power(2, Min(B.Count, 30))));
  LNow := UnixNow;
  Inc(B.Count);
  B.Active := True;
  B.Since := LNow;
  B.Expires := LNow + LSecs;
  B.Reason := AReason;
  SetLength(C.Strikes, 0);
  Inc(Stats.Bans);
  { "Ban <ip>" is what a fail2ban filter on our logfile would match }
  Log(Format('Ban %s for %ds (ban #%d): %s', [B.IpText, LSecs, B.Count, AReason]));
  if not FCfg.TestMode then
  begin
    FAction.Ban(B.Ip, LSecs);
    FDirty := True;
    SaveState;
  end;
end;

procedure TBanEngine.Purge(ANow: Int64);
var
  I: Integer;
  C: TClient;
  B: TBanRecord;
  LIdle: Int64;
begin
  LIdle := Max(FCfg.FindTime, FCfg.CacheSecs);
  for I := FClients.Count - 1 downto 0 do
  begin
    C := TClient(FClients.Objects[I]);
    if ANow - C.LastSeen > LIdle then
      FClients.Delete(I);
  end;
  { an expired ban is remembered for bantime.increment until maxtime passes }
  for I := FBans.Count - 1 downto 0 do
  begin
    B := TBanRecord(FBans.Objects[I]);
    if not B.Active and (ANow - B.Expires > FCfg.BanMax) then
    begin
      FBans.Delete(I);
      FDirty := True;
    end;
  end;
  FLastPurge := ANow;
end;

procedure TBanEngine.Tick;
var
  I: Integer;
  B: TBanRecord;
  LNow: Int64;
begin
  if FCfg.TestMode then
    Exit;
  LNow := UnixNow;
  for I := 0 to FBans.Count - 1 do
  begin
    B := TBanRecord(FBans.Objects[I]);
    if B.Active and (B.Expires <= LNow) then
    begin
      B.Active := False;
      Inc(Stats.Unbans);
      Log(Format('Unban %s', [B.IpText]));
      FAction.Unban(B.Ip);
      FDirty := True;
    end;
  end;
  if LNow - FLastPurge >= 60 then
    Purge(LNow);
  if FDirty then
    SaveState;
end;

procedure TBanEngine.LoadState;
var
  LData: TJSONData;
  LArr: TJSONArray;
  LObj: TJSONObject;
  LText: TStringList;
  B: TBanRecord;
  LIp: TIpAddr;
  I: Integer;
begin
  if not FileExists(FCfg.StateFile) then
    Exit;
  LText := TStringList.Create;
  try
    LText.LoadFromFile(FCfg.StateFile);
    try
      LData := GetJSON(LText.Text);
    except
      on E: Exception do
      begin
        Log(Format('state file %s unreadable (%s); starting with no bans', [FCfg.StateFile, E.Message]));
        Exit;
      end;
    end;
  finally
    LText.Free;
  end;
  try
    if not (LData is TJSONObject) or not TJSONObject(LData).Find('bans', LArr) then
      Exit;
    for I := 0 to LArr.Count - 1 do
    begin
      if not (LArr[I] is TJSONObject) then
        Continue;
      LObj := TJSONObject(LArr[I]);
      if not ParseIp(LObj.Get('ip', ''), LIp) or (FBans.IndexOf(IpToStr(LIp)) >= 0) then
        Continue;
      B := TBanRecord.Create;
      B.Ip := LIp;
      B.IpText := IpToStr(LIp);
      B.Since := LObj.Get('since', Int64(0));
      B.Expires := LObj.Get('expires', Int64(0));
      B.Count := LObj.Get('count', 1);
      B.Active := LObj.Get('active', False);
      B.Reason := LObj.Get('reason', '');
      FBans.AddObject(B.IpText, B);
    end;
  finally
    LData.Free;
  end;
end;

procedure TBanEngine.SaveState;
var
  LRoot, LObj: TJSONObject;
  LArr: TJSONArray;
  B: TBanRecord;
  I: Integer;
  LText: TStringList;
  LTmp: string;
begin
  if FCfg.TestMode then
    Exit;
  LRoot := TJSONObject.Create;
  LText := TStringList.Create;
  try
    LRoot.Add('version', 1);
    LArr := TJSONArray.Create;
    for I := 0 to FBans.Count - 1 do
    begin
      B := TBanRecord(FBans.Objects[I]);
      LObj := TJSONObject.Create;
      LObj.Add('ip', B.IpText);
      LObj.Add('since', B.Since);
      LObj.Add('expires', B.Expires);
      LObj.Add('count', B.Count);
      LObj.Add('active', B.Active);
      LObj.Add('reason', B.Reason);
      LArr.Add(LObj);
    end;
    LRoot.Add('bans', LArr);
    LText.Text := LRoot.FormatJSON;
    try
      ForceDirectories(ExtractFileDir(ExpandFileName(FCfg.StateFile)));
      LTmp := FCfg.StateFile + '.tmp';
      LText.SaveToFile(LTmp);
      if not RenameFile(LTmp, FCfg.StateFile) then
      begin
        DeleteFile(FCfg.StateFile);
        RenameFile(LTmp, FCfg.StateFile);
      end;
      FDirty := False;
    except
      on E: Exception do
        Log(Format('could not save %s: %s', [FCfg.StateFile, E.Message]));
    end;
  finally
    LText.Free;
    LRoot.Free;
  end;
end;

procedure TBanEngine.Start;
var
  I, LRestored: Integer;
  B: TBanRecord;
  LNow: Int64;
begin
  LoadState;
  FAction.Start;
  LNow := UnixNow;
  LRestored := 0;
  for I := 0 to FBans.Count - 1 do
  begin
    B := TBanRecord(FBans.Objects[I]);
    if not B.Active then
      Continue;
    if B.Expires > LNow then
    begin
      FAction.Ban(B.Ip, B.Expires - LNow);
      Inc(LRestored);
    end
    else
    begin
      B.Active := False;
      FDirty := True;
      { a chain-based action was just flushed by actionstart; others keep
        their rules across our restart and need the unban }
      if FAction.StartCmd = '' then
        FAction.Unban(B.Ip);
      Log(Format('Unban %s (expired while stopped)', [B.IpText]));
    end;
  end;
  if LRestored > 0 then
    Log(Format('restored %d ban(s) from %s', [LRestored, FCfg.StateFile]));
  FLastPurge := LNow;
  if FDirty then
    SaveState;
end;

procedure TBanEngine.Stop;
begin
  SaveState;
  if FCfg.CleanupOnExit then
    FAction.Stop;
end;

procedure TBanEngine.PrintBans;
var
  I, LActive: Integer;
  B: TBanRecord;
  LNow: Int64;
begin
  LoadState;
  LNow := UnixNow;
  LActive := 0;
  for I := 0 to FBans.Count - 1 do
  begin
    B := TBanRecord(FBans.Objects[I]);
    if not B.Active or (B.Expires <= LNow) then
      Continue;
    Inc(LActive);
    WriteLn(Format('%-39s %7ds left  ban #%d  %s', [B.IpText, B.Expires - LNow, B.Count, B.Reason]));
  end;
  WriteLn(Format('%d active ban(s), %d remembered for bantime.increment, state %s',
    [LActive, FBans.Count - LActive, FCfg.StateFile]));
end;

function TBanEngine.Summary: string;
begin
  Result := Format('%d lines, %d parsed, %d ignored, %d suspicious, %d decider calls (%d failed), ' +
    '%d strikes, %d bans, %d unbans', [Stats.Lines, Stats.Parsed, Stats.Ignored, Stats.Suspicious,
    Stats.DeciderCalls, Stats.DeciderErrors, Stats.Strikes, Stats.Bans, Stats.Unbans]);
end;

end.
