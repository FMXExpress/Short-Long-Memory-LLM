program jevban;

(* jevban: fail2ban for web attackers, with a Jev-style decision model as
   the judge. It follows Apache access logs, flags requests that look like
   security probes (/.env, /wp-login.php, ../, ${jndi:, sqlmap, ...), asks
   the decision API (OpenJev or the hosted Jev API, POST /v1/systemone)
   whether that client is attacking, and hands the ones that are to the
   firewall (iptables, nftables, ufw, firewalld, a fail2ban jail, netsh). *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}BaseUnix,{$ENDIF}
  SysUtils, Classes,
  Ban.Config, Ban.Tail, Ban.Engine;

var
  Cfg: TBanConfig;
  Engine: TBanEngine;
  Tailer: TLogTailer;
  Lines: TStringList;
  Line: string;
  Stopping: Boolean = False;

{$IFDEF UNIX}
procedure OnSignal(ASig: cint); cdecl;
begin
  Stopping := True;
end;

procedure InstallSignals;
begin
  FpSignal(SIGTERM, @OnSignal);
  FpSignal(SIGINT, @OnSignal);
  FpSignal(SIGPIPE, signalhandler(SIG_IGN));
  FpSignal(SIGHUP, signalhandler(SIG_IGN));
end;
{$ELSE}
procedure InstallSignals;
begin
end;
{$ENDIF}

procedure RunTest;
var
  I: Integer;
begin
  for I := 0 to Cfg.LogPaths.Count - 1 do
    if not FileExists(Cfg.LogPaths[I]) then
      raise EArgumentException.CreateFmt('%s not found', [Cfg.LogPaths[I]]);
  Engine.Log(Format('test run over %s, mode %s%s', [Tailer.FileNames, Cfg.Mode,
    BoolToStr(Cfg.Mode = 'jev', ' at ' + Cfg.JevUrl, '')]));
  while Tailer.Poll(Lines) do
  begin
    for Line in Lines do
      Engine.Feed(Line);
    Lines.Clear;
  end;
  Engine.Log(Engine.Summary);
end;

procedure RunDaemon;
begin
  Engine.Start;
  Engine.Log(Format('following %s; mode %s%s; action %s%s', [Tailer.FileNames, Cfg.Mode,
    BoolToStr(Cfg.Mode = 'jev', ' at ' + Cfg.JevUrl, ''), Cfg.ActionType,
    BoolToStr(Cfg.DryRun, ' (dry run)', '')]));
  while not Stopping do
  begin
    Lines.Clear;
    if Tailer.Poll(Lines) then
      for Line in Lines do
      begin
        Engine.Feed(Line);
        if Stopping then
          Break;
      end;
    Engine.Tick;
    if Lines.Count = 0 then
      Sleep(Cfg.PollMs);
  end;
  Engine.Log('stopping: ' + Engine.Summary);
  Engine.Stop;
end;

begin
  Cfg := TBanConfig.Create;
  Engine := nil;
  Tailer := nil;
  Lines := TStringList.Create;
  try
    try
      if not Cfg.Load then
        Exit;
      Cfg.Validate;
      Engine := TBanEngine.Create(Cfg);
      if Cfg.ListBans then
      begin
        Engine.PrintBans;
        Exit;
      end;
      InstallSignals;
      Tailer := TLogTailer.Create(Cfg.LogPaths, Cfg.FromStart);
      if Cfg.TestMode then
        RunTest
      else
        RunDaemon;
    except
      on E: EArgumentException do
      begin
        WriteLn(StdErr, 'jevban: ', E.Message);
        ExitCode := 2;
      end;
      on E: Exception do
      begin
        WriteLn(StdErr, 'jevban: ', E.ClassName, ': ', E.Message);
        ExitCode := 1;
      end;
    end;
  finally
    Tailer.Free;
    Engine.Free;
    Lines.Free;
    Cfg.Free;
  end;
end.
