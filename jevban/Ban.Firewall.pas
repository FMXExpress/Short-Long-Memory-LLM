unit Ban.Firewall;

(* Firewall actions in the fail2ban mould: actionstart once at startup,
   actionban / actionunban per address, actionstop at shutdown. Each is a
   shell command line (/bin/sh -c, cmd.exe /c on Windows) with tags:
     <ip>        the address, canonical (only ever a parsed address)
     <v>         4 or 6;  <family> inet or inet6;  <fwfamily> ipv4 or ipv6
     <iptables>  iptables or ip6tables; a start/stop command using it runs
                 once per family
     <name> <port> <jail> <bantime>
   Presets: iptables (default), nftables, ufw, firewalld, fail2ban (hands the
   address to a fail2ban jail), netsh (Windows), none (log only). Any of the
   four commands set in the config replaces the preset's. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Process, Ban.Net;

type
  TLogProc = procedure(const AMsg: string) of object;

  TBanAction = class
  private
    FLog: TLogProc;
    function Expand(const ACmd: string; const AIp: TIpAddr; AHasIp: Boolean;
      const AIptables: string; ABanSecs: Int64): string;
    function Run(const ACmd: string): Boolean;
    procedure RunLifecycle(const ACmd, AWhat: string);
    procedure Say(const AMsg: string);
  public
    Kind: string;
    Name: string;
    Port: string;
    Jail: string;
    StartCmd, StopCmd, BanCmd, UnbanCmd: string;
    DryRun: Boolean;
    constructor Create(ALog: TLogProc);
    { fills the four commands from Kind; raises on an unknown preset }
    procedure ApplyPreset;
    procedure Start;
    procedure Stop;
    function Ban(const AIp: TIpAddr; ABanSecs: Int64): Boolean;
    function Unban(const AIp: TIpAddr): Boolean;
  end;

implementation

constructor TBanAction.Create(ALog: TLogProc);
begin
  inherited Create;
  FLog := ALog;
  Kind := 'iptables';
  Name := 'jevban';
  Port := '80,443';
  Jail := 'jevban';
end;

procedure TBanAction.Say(const AMsg: string);
begin
  if Assigned(FLog) then
    FLog(AMsg);
end;

procedure TBanAction.ApplyPreset;
var
  LStart, LStop, LBan, LUnban: string;
begin
  LStart := '';
  LStop := '';
  LBan := '';
  LUnban := '';
  if Kind = 'iptables' then
  begin
    { a chain of our own, hooked into INPUT for the web ports; flushed at
      start so restored bans are not duplicated }
    LStart := '<iptables> -N <name> 2>/dev/null; <iptables> -F <name> && ' +
      '{ <iptables> -C INPUT -p tcp -m multiport --dports <port> -j <name> 2>/dev/null || ' +
      '<iptables> -I INPUT -p tcp -m multiport --dports <port> -j <name>; }';
    LStop := '<iptables> -D INPUT -p tcp -m multiport --dports <port> -j <name>; ' +
      '<iptables> -F <name>; <iptables> -X <name>';
    LBan := '<iptables> -I <name> 1 -s <ip> -j DROP';
    LUnban := '<iptables> -D <name> -s <ip> -j DROP';
  end
  else if Kind = 'nftables' then
  begin
    LStart := 'nft add table inet <name> && ' +
      'nft ''add set inet <name> v4 { type ipv4_addr; }'' && ' +
      'nft ''add set inet <name> v6 { type ipv6_addr; }'' && ' +
      'nft ''add chain inet <name> input { type filter hook input priority -1; }'' && ' +
      'nft flush chain inet <name> input && nft flush set inet <name> v4 && nft flush set inet <name> v6 && ' +
      'nft ''add rule inet <name> input ip saddr @v4 tcp dport { <port> } drop'' && ' +
      'nft ''add rule inet <name> input ip6 saddr @v6 tcp dport { <port> } drop''';
    LStop := 'nft delete table inet <name>';
    LBan := 'nft add element inet <name> v<v> { <ip> }';
    LUnban := 'nft delete element inet <name> v<v> { <ip> }';
  end
  else if Kind = 'ufw' then
  begin
    LBan := 'ufw prepend deny from <ip> to any port <port> proto tcp';
    LUnban := 'ufw delete deny from <ip> to any port <port> proto tcp';
  end
  else if Kind = 'firewalld' then
  begin
    { runtime rules only: jevban restores its bans itself after a restart }
    LBan := 'firewall-cmd --add-rich-rule="rule family=<fwfamily> source address=<ip> drop"';
    LUnban := 'firewall-cmd --remove-rich-rule="rule family=<fwfamily> source address=<ip> drop"';
  end
  else if Kind = 'fail2ban' then
  begin
    LBan := 'fail2ban-client set <jail> banip <ip>';
    LUnban := 'fail2ban-client set <jail> unbanip <ip>';
  end
  else if Kind = 'netsh' then
  begin
    LBan := 'netsh advfirewall firewall add rule name="<name> <ip>" dir=in action=block remoteip=<ip>';
    LUnban := 'netsh advfirewall firewall delete rule name="<name> <ip>"';
  end
  else if (Kind <> 'none') and (Kind <> 'command') then
    raise Exception.CreateFmt('unknown action type "%s" (iptables, nftables, ufw, firewalld, ' +
      'fail2ban, netsh, command, none)', [Kind]);
  { the config's own commands win over the preset }
  if StartCmd = '' then
    StartCmd := LStart;
  if StopCmd = '' then
    StopCmd := LStop;
  if BanCmd = '' then
    BanCmd := LBan;
  if UnbanCmd = '' then
    UnbanCmd := LUnban;
  if (Kind = 'command') and (BanCmd = '') then
    raise Exception.Create('action type "command" needs actionban');
end;

function TBanAction.Expand(const ACmd: string; const AIp: TIpAddr; AHasIp: Boolean;
  const AIptables: string; ABanSecs: Int64): string;
begin
  Result := ACmd;
  if AHasIp then
  begin
    Result := StringReplace(Result, '<ip>', IpToStr(AIp), [rfReplaceAll]);
    if AIp.IsV4 then
    begin
      Result := StringReplace(Result, '<v>', '4', [rfReplaceAll]);
      Result := StringReplace(Result, '<family>', 'inet', [rfReplaceAll]);
      Result := StringReplace(Result, '<fwfamily>', 'ipv4', [rfReplaceAll]);
      Result := StringReplace(Result, '<iptables>', 'iptables', [rfReplaceAll]);
    end
    else
    begin
      Result := StringReplace(Result, '<v>', '6', [rfReplaceAll]);
      Result := StringReplace(Result, '<family>', 'inet6', [rfReplaceAll]);
      Result := StringReplace(Result, '<fwfamily>', 'ipv6', [rfReplaceAll]);
      Result := StringReplace(Result, '<iptables>', 'ip6tables', [rfReplaceAll]);
    end;
  end
  else
    Result := StringReplace(Result, '<iptables>', AIptables, [rfReplaceAll]);
  Result := StringReplace(Result, '<name>', Name, [rfReplaceAll]);
  Result := StringReplace(Result, '<port>', Port, [rfReplaceAll]);
  Result := StringReplace(Result, '<jail>', Jail, [rfReplaceAll]);
  Result := StringReplace(Result, '<bantime>', IntToStr(ABanSecs), [rfReplaceAll]);
end;

function TBanAction.Run(const ACmd: string): Boolean;
var
  LProc: TProcess;
  LOut: TStringList;
begin
  if ACmd = '' then
    Exit(True);
  if DryRun then
  begin
    Say('[dry-run] ' + ACmd);
    Exit(True);
  end;
  LProc := TProcess.Create(nil);
  LOut := TStringList.Create;
  try
    {$IFDEF WINDOWS}
    LProc.Executable := GetEnvironmentVariable('COMSPEC');
    if LProc.Executable = '' then
      LProc.Executable := 'cmd.exe';
    LProc.Parameters.Add('/c');
    {$ELSE}
    LProc.Executable := '/bin/sh';
    LProc.Parameters.Add('-c');
    {$ENDIF}
    LProc.Parameters.Add(ACmd);
    LProc.Options := [poWaitOnExit, poUsePipes, poStderrToOutPut];
    LProc.ShowWindow := swoHide;
    try
      LProc.Execute;
    except
      on E: Exception do
      begin
        Say(Format('action failed to start: %s (%s)', [ACmd, E.Message]));
        Exit(False);
      end;
    end;
    LOut.LoadFromStream(LProc.Output);
    Result := LProc.ExitStatus = 0;
    if not Result then
      Say(Format('action exit %d: %s%s', [LProc.ExitStatus, ACmd,
        StringReplace(' | ' + Trim(LOut.Text), LineEnding, ' | ', [rfReplaceAll])]));
  finally
    LOut.Free;
    LProc.Free;
  end;
end;

procedure TBanAction.RunLifecycle(const ACmd, AWhat: string);
var
  LNone: TIpAddr;
begin
  if ACmd = '' then
    Exit;
  LNone := Default(TIpAddr);
  if Pos('<iptables>', ACmd) > 0 then
  begin
    if not Run(Expand(ACmd, LNone, False, 'iptables', 0)) then
      Say(AWhat + ' failed for IPv4');
    { many hosts have no ip6tables; not fatal }
    if not Run(Expand(ACmd, LNone, False, 'ip6tables', 0)) then
      Say(AWhat + ' failed for IPv6 (IPv6 bans will not work)');
  end
  else if not Run(Expand(ACmd, LNone, False, '', 0)) then
    Say(AWhat + ' failed');
end;

procedure TBanAction.Start;
begin
  RunLifecycle(StartCmd, 'actionstart');
end;

procedure TBanAction.Stop;
begin
  RunLifecycle(StopCmd, 'actionstop');
end;

function TBanAction.Ban(const AIp: TIpAddr; ABanSecs: Int64): Boolean;
begin
  Result := Run(Expand(BanCmd, AIp, True, '', ABanSecs));
end;

function TBanAction.Unban(const AIp: TIpAddr): Boolean;
begin
  Result := Run(Expand(UnbanCmd, AIp, True, '', 0));
end;

end.
