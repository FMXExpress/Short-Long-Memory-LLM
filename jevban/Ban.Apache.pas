unit Ban.Apache;

(* Apache access log lines: common, combined and vhost_combined
     [%v:%p ]%h %l %u %t "%r" %>s %b[ "%{Referer}i" "%{User-Agent}i"]
   (nginx's default "combined" is the same shape). The bracketed time field
   anchors the parse: the client address is the third token before it, so an
   optional vhost prefix needs no configuration. Apache escapes quotes and
   non-printable bytes in "%r" as \" and \xhh; Request keeps that escaped
   text (safe to log and to show a model), Decoded is what the attacker
   actually sent, URL-decoded, for the signatures. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, DateUtils, Ban.Net;

type
  TAccessEntry = record
    VHost: string;
    Ip: TIpAddr;
    IpText: string;     { canonical (IpToStr) }
    TimeText: string;
    Epoch: Int64;       { the log time, Unix seconds UTC; 0 if unparsable }
    Request: string;    { "%r" as logged, Apache-escaped }
    Method, Target, Protocol: string;
    Decoded: string;    { Target with \xhh and %hh (twice) undone }
    Status: Integer;
    Bytes: Int64;
    Referer, Agent: string;
  end;

function ParseAccessLine(const ALine: string; out E: TAccessEntry): Boolean;
{ one line for a model or a log: time, request, status, size, agent }
function EntrySummary(const E: TAccessEntry; AMaxLen: Integer = 400): string;
function ApacheUnescape(const S: string): string;
function UrlDecode(const S: string): string;
function UnixNow: Int64;

implementation

function UnixNow: Int64;
begin
  Result := DateTimeToUnix(LocalTimeToUniversal(Now));
end;

function ApacheUnescape(const S: string): string;
var
  I, LHi, LLo: Integer;

  function Hex(C: Char): Integer;
  begin
    case C of
      '0'..'9': Result := Ord(C) - 48;
      'a'..'f': Result := Ord(C) - 87;
      'A'..'F': Result := Ord(C) - 55;
    else
      Result := -1;
    end;
  end;

begin
  Result := '';
  I := 1;
  while I <= Length(S) do
  begin
    if (S[I] = '\') and (I < Length(S)) then
    begin
      if (S[I + 1] = 'x') and (I + 3 <= Length(S)) then
      begin
        LHi := Hex(S[I + 2]);
        LLo := Hex(S[I + 3]);
        if (LHi >= 0) and (LLo >= 0) then
        begin
          Result := Result + Chr(LHi * 16 + LLo);
          Inc(I, 4);
          Continue;
        end;
      end;
      case S[I + 1] of
        '"', '\': Result := Result + S[I + 1];
        'n': Result := Result + #10;
        'r': Result := Result + #13;
        't': Result := Result + #9;
      else
        Result := Result + S[I] + S[I + 1];
      end;
      Inc(I, 2);
      Continue;
    end;
    Result := Result + S[I];
    Inc(I);
  end;
end;

function UrlDecode(const S: string): string;
var
  I, LVal: Integer;
begin
  Result := '';
  I := 1;
  while I <= Length(S) do
  begin
    if (S[I] = '%') and (I + 2 <= Length(S)) and
       TryStrToInt('$' + Copy(S, I + 1, 2), LVal) then
    begin
      Result := Result + Chr(LVal);
      Inc(I, 3);
      Continue;
    end;
    if S[I] = '+' then
      Result := Result + ' '
    else
      Result := Result + S[I];
    Inc(I);
  end;
end;

{ next token: "quoted" (escapes kept), [bracketed], or up to a space }
function NextToken(const L: string; var P: Integer; out T: string): Boolean;
var
  LStart: Integer;
begin
  while (P <= Length(L)) and (L[P] = ' ') do
    Inc(P);
  if P > Length(L) then
    Exit(False);
  Result := True;
  if L[P] = '"' then
  begin
    Inc(P);
    LStart := P;
    while (P <= Length(L)) and (L[P] <> '"') do
    begin
      if (L[P] = '\') and (P < Length(L)) then
        Inc(P);
      Inc(P);
    end;
    T := '"' + Copy(L, LStart, P - LStart);   { the quote marks the kind }
    Inc(P);
  end
  else if L[P] = '[' then
  begin
    LStart := P;
    while (P <= Length(L)) and (L[P] <> ']') do
      Inc(P);
    T := Copy(L, LStart, P - LStart + 1);
    Inc(P);
  end
  else
  begin
    LStart := P;
    while (P <= Length(L)) and (L[P] <> ' ') do
      Inc(P);
    T := Copy(L, LStart, P - LStart);
  end;
end;

const
  CMonths: array[1..12] of string = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec');

{ [10/Oct/2000:13:55:36 -0700] }
function ParseApacheTime(const S: string): Int64;
var
  LDay, LMon, LYear, LHour, LMin, LSec, LOff, I: Integer;
  LText: string;
begin
  Result := 0;
  LText := Copy(S, 2, Length(S) - 2);
  if Length(LText) < 26 then
    Exit;
  LMon := 0;
  for I := 1 to 12 do
    if CMonths[I] = Copy(LText, 4, 3) then
      LMon := I;
  if (LMon = 0) or
     not TryStrToInt(Copy(LText, 1, 2), LDay) or
     not TryStrToInt(Copy(LText, 8, 4), LYear) or
     not TryStrToInt(Copy(LText, 13, 2), LHour) or
     not TryStrToInt(Copy(LText, 16, 2), LMin) or
     not TryStrToInt(Copy(LText, 19, 2), LSec) or
     not TryStrToInt(Copy(LText, 22, 5), LOff) then
    Exit;
  if (LDay < 1) or (LDay > 31) or (LHour > 23) or (LMin > 59) or (LSec > 60) then
    Exit;
  try
    Result := DateTimeToUnix(EncodeDateTime(LYear, LMon, LDay, LHour, LMin, LSec mod 60, 0));
  except
    Exit(0);
  end;
  { local = UTC + offset; -0530 parses as -530: div and mod keep the sign }
  LOff := (LOff div 100) * 3600 + (LOff mod 100) * 60;
  Result := Result - LOff;
end;

function Unquote(const T: string): string;
begin
  if (T <> '') and (T[1] = '"') then
    Result := Copy(T, 2, MaxInt)
  else
    Result := T;
end;

function ParseAccessLine(const ALine: string; out E: TAccessEntry): Boolean;
var
  LTokens: TStringList;
  LTok, LReq: string;
  P, LTime, I, LSp: Integer;
begin
  Result := False;
  E := Default(TAccessEntry);
  LTokens := TStringList.Create;
  try
    P := 1;
    LTime := -1;
    while (LTokens.Count < 16) and NextToken(ALine, P, LTok) do
    begin
      if (LTime < 0) and (LTok <> '') and (LTok[1] = '[') then
        LTime := LTokens.Count;
      LTokens.Add(LTok);
    end;
    { host ident user [time] "request" status bytes }
    if (LTime < 3) or (LTokens.Count < LTime + 4) then
      Exit;
    if not ParseIp(LTokens[LTime - 3], E.Ip) then
      Exit;
    E.IpText := IpToStr(E.Ip);
    if LTime >= 4 then
      E.VHost := LTokens[LTime - 4];
    E.TimeText := LTokens[LTime];
    E.Epoch := ParseApacheTime(E.TimeText);
    LReq := LTokens[LTime + 1];
    if (LReq = '') or (LReq[1] <> '"') then
      Exit;
    E.Request := Unquote(LReq);
    E.Status := StrToIntDef(LTokens[LTime + 2], 0);
    E.Bytes := StrToInt64Def(LTokens[LTime + 3], 0);
    if LTokens.Count > LTime + 4 then
      E.Referer := Unquote(LTokens[LTime + 4]);
    if LTokens.Count > LTime + 5 then
      E.Agent := Unquote(LTokens[LTime + 5]);
  finally
    LTokens.Free;
  end;
  { METHOD target PROTOCOL; a malformed request keeps it all in Target }
  LReq := E.Request;
  LSp := Pos(' ', LReq);
  if LSp > 0 then
  begin
    E.Method := Copy(LReq, 1, LSp - 1);
    LReq := Copy(LReq, LSp + 1, MaxInt);
    I := Length(LReq);
    while (I > 0) and (LReq[I] <> ' ') do
      Dec(I);
    if (I > 0) and (Copy(LReq, I + 1, 5) = 'HTTP/') then
    begin
      E.Protocol := Copy(LReq, I + 1, MaxInt);
      LReq := Copy(LReq, 1, I - 1);
    end;
    E.Target := LReq;
  end
  else
    E.Target := LReq;
  E.Decoded := UrlDecode(UrlDecode(ApacheUnescape(E.Target)));
  Result := True;
end;

function EntrySummary(const E: TAccessEntry; AMaxLen: Integer): string;
begin
  Result := Format('%s "%s" %d %d ua="%s"', [Copy(E.TimeText, 2, 20), E.Request, E.Status,
    E.Bytes, E.Agent]);
  if E.Referer <> '-' then
    Result := Result + Format(' ref="%s"', [E.Referer]);
  if Length(Result) > AMaxLen then
    Result := Copy(Result, 1, AMaxLen) + '...';
end;

end.
