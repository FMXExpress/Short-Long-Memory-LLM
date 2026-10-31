unit Mem.Https;

(* HTTPS downloads for provisioning, built on FPC's fphttpclient but with our
   own TLS socket handler, because FPC 3.2.2's stock one falls short in three
   ways that matter when the download is a native library we will load:

   - it does not verify certificates (no CA store, no peer or hostname check);
     this handler verifies the chain against the system CA store and checks
     the hostname with SSL_set1_host;
   - it cannot go through an HTTPS proxy (it starts TLS with the proxy itself
     instead of sending CONNECT); this handler tunnels when HTTPS_PROXY is set,
     except for hosts listed in NO_PROXY;
   - on Linux it never tries libssl.so.3, and on Windows it never tries the
     OpenSSL 3 DLL names, so a machine with only OpenSSL 3 had no TLS at all;
     EnableOpenSSL3 adds those names before the library is first loaded.

   On Windows, put libssl-3-x64.dll and libcrypto-3-x64.dll next to
   memagent.exe, with cacert.pem for the trusted roots (the installer does). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes;

{ GET AUrl into ADest (via ADest.part, so a broken download never looks
  finished), following up to 10 redirects. Raises on any failure. }
procedure HttpsGetToFile(const AUrl, ADest: string; AVerbose: Boolean = True);

implementation

uses
  ctypes, sockets, ssockets, sslbase, dynlibs, URIParser, fphttpclient, openssl, fpopenssl;

type
  TSslSet1Host = function(ssl: PSSL; hostname: PAnsiChar): cint; cdecl;
  TSslCtxDefaultPaths = function(ctx: PSSL_CTX): cint; cdecl;
  TSslCtxLoadVerify = function(ctx: PSSL_CTX; CAfile, CApath: PAnsiChar): cint; cdecl;
  TErrErrorStringN = procedure(e: culong; buf: PAnsiChar; len: csize_t); cdecl;

  { TLS over an already-connected TCP socket, optionally after a CONNECT
    handshake with an HTTP proxy. }
  TVerifiedTlsHandler = class(TSocketHandler)
  private
    FHost: string;
    FPort: Word;
    FTunnel: Boolean;
    FCtx: TSSLContext;
    FSsl: TSSL;
    procedure Fail(const AMsg: string);
    procedure OpenTunnel;
    procedure StartTls;
  public
    constructor Create(const AHost: string; APort: Word; ATunnel: Boolean); reintroduce;
    destructor Destroy; override;
    function Connect: Boolean; override;
    function Close: Boolean; override;
    function Shutdown(BiDirectional: Boolean): Boolean; override;
    function Send(const Buffer; Count: Integer): Integer; override;
    function Recv(const Buffer; Count: Integer): Integer; override;
    function BytesAvailable: Integer; override;
  end;

  { fphttpclient wiring: hands out a verifying handler for the current target }
  TDownloader = class
  private
    FClient: TFPHTTPClient;
    FTarget: TURI;
    FProxy: TURI;
    FHaveProxy, FUseProxy: Boolean;
    FNoProxy: string;
    procedure PickRoute(const AHost: string);
    procedure GetHandler(Sender: TObject; const UseSSL: Boolean; out AHandler: TSocketHandler);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Get(const AUrl: string; AStream: TStream; AVerbose: Boolean);
  end;

var
  GSslSet1Host: TSslSet1Host = nil;
  GCtxDefaultPaths: TSslCtxDefaultPaths = nil;
  GCtxLoadVerify: TSslCtxLoadVerify = nil;
  GErrString: TErrErrorStringN = nil;
  GSslReady: Boolean = False;

procedure EnableOpenSSL3;
begin
  {$IFDEF WINDOWS}
  { tried after ssleay32.dll and libssl32.dll, which modern installs lack }
  {$IFDEF WIN64}
  DLLSSLName3 := 'libssl-3-x64.dll';
  DLLUtilName2 := 'libcrypto-3-x64.dll';
  {$ELSE}
  DLLSSLName3 := 'libssl-3.dll';
  DLLUtilName2 := 'libcrypto-3.dll';
  {$ENDIF}
  {$ELSE}
  { '' (the -dev symlink) first, then OpenSSL 3, then the 1.x names }
  DLLVersions[2] := '.3';
  DLLVersions[3] := '.1.1';
  {$ENDIF}
end;

procedure EnsureSsl;
begin
  if GSslReady then
    Exit;
  EnableOpenSSL3;
  if not InitSSLInterface then
    raise EInOutError.Create('OpenSSL is not available (need libssl 1.1 or 3; on Windows ' +
      'libssl-3-x64.dll + libcrypto-3-x64.dll next to the executable)');
  Pointer(GSslSet1Host) := GetProcedureAddress(SSLLibHandle, 'SSL_set1_host');
  Pointer(GCtxDefaultPaths) := GetProcedureAddress(SSLLibHandle, 'SSL_CTX_set_default_verify_paths');
  Pointer(GCtxLoadVerify) := GetProcedureAddress(SSLLibHandle, 'SSL_CTX_load_verify_locations');
  Pointer(GErrString) := GetProcedureAddress(SSLUtilHandle, 'ERR_error_string_n');
  if (GSslSet1Host = nil) or (GCtxDefaultPaths = nil) then
    raise EInOutError.Create('OpenSSL is too old for hostname verification (need 1.1.0 or newer)');
  GSslReady := True;
end;

function LastSslError: string;
var
  LBuf: array[0..255] of AnsiChar;
  LErr: culong;
begin
  LErr := ErrGetError;
  if (LErr = 0) or (GErrString = nil) then
    Exit('');
  GErrString(LErr, @LBuf[0], SizeOf(LBuf));
  Result := string(PAnsiChar(@LBuf[0]));
end;

{ TVerifiedTlsHandler }

constructor TVerifiedTlsHandler.Create(const AHost: string; APort: Word; ATunnel: Boolean);
begin
  inherited Create;
  FHost := AHost;
  FPort := APort;
  FTunnel := ATunnel;
end;

destructor TVerifiedTlsHandler.Destroy;
begin
  FreeAndNil(FSsl);
  FreeAndNil(FCtx);
  inherited Destroy;
end;

procedure TVerifiedTlsHandler.Fail(const AMsg: string);
begin
  raise ESocketError.CreateFmt('TLS to %s:%d: %s', [FHost, FPort, AMsg]);
end;

procedure TVerifiedTlsHandler.OpenTunnel;
var
  LReq, LResp: AnsiString;
  LCh: AnsiChar;
begin
  LReq := Format('CONNECT %s:%d HTTP/1.1'#13#10'Host: %s:%d'#13#10#13#10, [FHost, FPort, FHost, FPort]);
  if fpSend(Socket.Handle, @LReq[1], Length(LReq), 0) <> Length(LReq) then
    Fail('could not send CONNECT to the proxy');
  LResp := '';
  repeat
    if fpRecv(Socket.Handle, @LCh, 1, 0) <> 1 then
      Fail('proxy closed the connection during CONNECT');
    LResp := LResp + LCh;
    if Length(LResp) > 16384 then
      Fail('oversized proxy response');
  until (Length(LResp) >= 4) and (Copy(LResp, Length(LResp) - 3, 4) = #13#10#13#10);
  if (Pos(' 200', Copy(LResp, 1, Pos(#13#10, LResp))) = 0) then
    Fail('proxy refused CONNECT: ' + Copy(LResp, 1, Pos(#13#10, LResp) - 1));
end;

procedure TVerifiedTlsHandler.StartTls;
var
  LHost, LBundle: AnsiString;
  LRes: cint;
begin
  EnsureSsl;
  LHost := AnsiString(FHost);
  FCtx := TSSLContext.Create(stAny);
  { the system CA store; also honours SSL_CERT_FILE / SSL_CERT_DIR }
  if GCtxDefaultPaths(FCtx.CTX) <> 1 then
    Fail('could not load the system CA certificates');
  (* Windows has no OpenSSL CA store: a release ships cacert.pem (Mozilla's
     roots, from curl.se) next to the executable *)
  LBundle := AnsiString(ExtractFilePath(ParamStr(0)) + 'cacert.pem');
  if FileExists(LBundle) and (GCtxLoadVerify <> nil) and
     (GCtxLoadVerify(FCtx.CTX, PAnsiChar(LBundle), nil) <> 1) then
    Fail('could not load ' + LBundle);
  FCtx.SetVerify(SSL_VERIFY_PEER, nil);
  FSsl := TSSL.Create(FCtx);
  if FSsl.SetFd(Socket.Handle) <> 1 then
    Fail('SSL_set_fd failed');
  FSsl.Ctrl(SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, PAnsiChar(LHost));
  if GSslSet1Host(FSsl.SSL, PAnsiChar(LHost)) <> 1 then
    Fail('SSL_set1_host failed');
  LRes := FSsl.Connect;
  if LRes <> 1 then
    Fail(Format('handshake failed (verify result %d) %s', [FSsl.VerifyResult, LastSslError]));
  if FSsl.VerifyResult <> X509_V_OK then
    Fail(Format('certificate not trusted (verify result %d)', [FSsl.VerifyResult]));
end;

function TVerifiedTlsHandler.Connect: Boolean;
begin
  if FTunnel then
    OpenTunnel;
  StartTls;
  Result := True;
end;

function TVerifiedTlsHandler.Close: Boolean;
begin
  Result := True;
  if FSsl <> nil then
    FSsl.Shutdown;
end;

function TVerifiedTlsHandler.Shutdown(BiDirectional: Boolean): Boolean;
begin
  Result := Close;
end;

function TVerifiedTlsHandler.Send(const Buffer; Count: Integer): Integer;
begin
  Result := FSsl.Write(@Buffer, Count);
end;

function TVerifiedTlsHandler.Recv(const Buffer; Count: Integer): Integer;
begin
  Result := FSsl.Read(@Buffer, Count);
  if Result < 0 then
    Result := 0;   { fphttpclient treats 0 as end of stream }
end;

function TVerifiedTlsHandler.BytesAvailable: Integer;
begin
  Result := FSsl.Pending;
end;

{ TDownloader }

constructor TDownloader.Create;
var
  LProxy: string;
begin
  inherited Create;
  FClient := TFPHTTPClient.Create(nil);
  FClient.AllowRedirect := False;   { followed in Get, so each hop gets a handler for its host }
  FClient.AddHeader('User-Agent', 'memagent/1.0');
  FClient.OnGetSocketHandler := @GetHandler;
  LProxy := GetEnvironmentVariable('HTTPS_PROXY');
  if LProxy = '' then
    LProxy := GetEnvironmentVariable('https_proxy');
  FHaveProxy := LProxy <> '';
  if FHaveProxy then
    FProxy := ParseURI(LProxy, 'http', 3128);
  FNoProxy := GetEnvironmentVariable('NO_PROXY');
  if FNoProxy = '' then
    FNoProxy := GetEnvironmentVariable('no_proxy');
end;

function NoProxyMatches(const AHost, AList: string): Boolean;
var
  LEntries: TStringList;
  LEntry, LHost: string;
begin
  { NO_PROXY entries: '*', an exact host, or a domain suffix ('.corp' or
    'corp' both match 'x.corp'); CIDR ranges are not interpreted }
  Result := False;
  LHost := LowerCase(AHost);
  LEntries := TStringList.Create;
  try
    LEntries.CommaText := StringReplace(AList, ' ', '', [rfReplaceAll]);
    for LEntry in LEntries do
    begin
      if LEntry = '' then
        Continue;
      if (LEntry = '*') or (LowerCase(LEntry) = LHost) then
        Exit(True);
      if (LEntry[1] = '.') and (Copy(LHost, Length(LHost) - Length(LEntry) + 1, MaxInt) = LowerCase(LEntry)) then
        Exit(True);
      if (LEntry[1] <> '.') and (Copy(LHost, Length(LHost) - Length(LEntry), MaxInt) = '.' + LowerCase(LEntry)) then
        Exit(True);
    end;
  finally
    LEntries.Free;
  end;
end;

procedure TDownloader.PickRoute(const AHost: string);
begin
  FUseProxy := FHaveProxy and not NoProxyMatches(AHost, FNoProxy);
  if FUseProxy then
  begin
    FClient.Proxy.Host := FProxy.Host;
    FClient.Proxy.Port := FProxy.Port;
  end
  else
    FClient.Proxy.Host := '';   { fphttpclient's ProxyActive is Host <> '' }
end;

destructor TDownloader.Destroy;
begin
  FClient.Free;
  inherited Destroy;
end;

procedure TDownloader.GetHandler(Sender: TObject; const UseSSL: Boolean; out AHandler: TSocketHandler);
var
  LPort: Word;
begin
  if not UseSSL then
    raise ESocketError.Create('refusing plain HTTP for a download');
  LPort := FTarget.Port;
  if LPort = 0 then
    LPort := 443;
  AHandler := TVerifiedTlsHandler.Create(FTarget.Host, LPort, FUseProxy);
end;

procedure TDownloader.Get(const AUrl: string; AStream: TStream; AVerbose: Boolean);
var
  LUrl, LLocation, LNext: string;
  LHop: Integer;
begin
  LUrl := AUrl;
  for LHop := 0 to 10 do
  begin
    FTarget := ParseURI(LUrl);
    PickRoute(FTarget.Host);
    AStream.Size := 0;
    FClient.HTTPMethod('GET', LUrl, AStream, [200, 301, 302, 303, 307, 308]);
    if FClient.ResponseStatusCode = 200 then
      Exit;
    LLocation := FClient.GetHeader(FClient.ResponseHeaders, 'Location');
    if LLocation = '' then
      raise EHTTPClient.CreateFmt('redirect without Location from %s', [LUrl]);
    if AVerbose then
      WriteLn(StdErr, '[download]   -> ', Copy(LLocation, 1, 100));
    if not ResolveRelativeURI(LUrl, LLocation, LNext) then
      raise EHTTPClient.CreateFmt('bad redirect %s', [LLocation]);
    LUrl := LNext;
  end;
  raise EHTTPClient.CreateFmt('too many redirects for %s', [AUrl]);
end;

procedure HttpsGetToFile(const AUrl, ADest: string; AVerbose: Boolean);
const
  ATTEMPTS = 3;
var
  LDl: TDownloader;
  LPart: string;
  LStream: TFileStream;
  LTry: Integer;
begin
  if AVerbose then
    WriteLn(StdErr, '[download] ', AUrl);
  ForceDirectories(ExtractFileDir(ADest));
  LPart := ADest + '.part';
  { a network error (DNS, connection reset) is often gone a few seconds later:
    retry it, after 2 s then 4 s; an HTTP error status is not retried }
  for LTry := 1 to ATTEMPTS do
  begin
    LDl := TDownloader.Create;
    try
      try
        LStream := TFileStream.Create(LPart, fmCreate);
        try
          LDl.Get(AUrl, LStream, AVerbose);
        finally
          LStream.Free;
        end;
        Break;
      except
        on E: ESocketError do
        begin
          DeleteFile(LPart);
          if LTry = ATTEMPTS then
            raise;
          WriteLn(StdErr, '[download] ', E.Message, '; retrying');
          Sleep(2000 * LTry);
        end;
        on Exception do
        begin
          DeleteFile(LPart);
          raise;
        end;
      end;
    finally
      LDl.Free;
    end;
  end;
  if FileExists(ADest) then
    DeleteFile(ADest);
  if not RenameFile(LPart, ADest) then
    raise EInOutError.CreateFmt('could not move %s into place', [LPart]);
end;

end.
