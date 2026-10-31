unit Ban.Net;

(* IPv4/IPv6 addresses and CIDR ranges. Everything is held in the 128-bit
   form (IPv4 as ::ffff:a.b.c.d) so one matcher covers both families.
   An address only reaches a firewall command after ParseIp accepted it and
   IpToStr re-printed it, so a log line can never smuggle shell text into an
   action. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes;

type
  TIpAddr = record
    Bytes: array[0..15] of Byte;
    IsV4: Boolean;
  end;

  TCidr = record
    Addr: TIpAddr;
    Bits: Integer;   { prefix length over the 128-bit form }
  end;

  TCidrList = array of TCidr;

function ParseIp(const S: string; out A: TIpAddr): Boolean;
{ canonical text: dotted quad, or RFC 5952 compressed IPv6 }
function IpToStr(const A: TIpAddr): string;
function ParseCidr(const S: string; out C: TCidr): Boolean;
function CidrMatch(const C: TCidr; const A: TIpAddr): Boolean;
{ space or comma separated; raises EConvertError on a bad entry }
function ParseCidrList(const S: string): TCidrList;
function InCidrList(const L: TCidrList; const A: TIpAddr): Boolean;

implementation

function ParseV4(const S: string; out B: array of Byte): Boolean;
var
  LPart, LIdx, LVal, I: Integer;
  LCh: Char;
begin
  Result := False;
  LPart := 0;
  LVal := 0;
  LIdx := 0;   { digits in the current part }
  for I := 1 to Length(S) + 1 do
  begin
    if I <= Length(S) then
      LCh := S[I]
    else
      LCh := '.';
    if LCh in ['0'..'9'] then
    begin
      Inc(LIdx);
      if LIdx > 3 then
        Exit;
      LVal := LVal * 10 + Ord(LCh) - Ord('0');
    end
    else if LCh = '.' then
    begin
      if (LIdx = 0) or (LVal > 255) or (LPart > 3) then
        Exit;
      B[LPart] := LVal;
      Inc(LPart);
      LVal := 0;
      LIdx := 0;
    end
    else
      Exit;
  end;
  Result := LPart = 4;
end;

function HexVal(C: Char): Integer;
begin
  case C of
    '0'..'9': Result := Ord(C) - Ord('0');
    'a'..'f': Result := Ord(C) - Ord('a') + 10;
    'A'..'F': Result := Ord(C) - Ord('A') + 10;
  else
    Result := -1;
  end;
end;

{ colon-separated hex groups; the last may be a dotted quad (two groups) }
function ParseGroups(const S: string; ALast: Boolean; var W: array of Word; out N: Integer): Boolean;
var
  LParts: TStringList;
  LPart: string;
  LV4: array[0..3] of Byte;
  I, J, LVal, LDigit: Integer;
begin
  Result := False;
  N := 0;
  if S = '' then
    Exit(True);
  LParts := TStringList.Create;
  try
    LParts.StrictDelimiter := True;
    LParts.Delimiter := ':';
    LParts.DelimitedText := S;
    for I := 0 to LParts.Count - 1 do
    begin
      LPart := LParts[I];
      if (Pos('.', LPart) > 0) and ALast and (I = LParts.Count - 1) then
      begin
        if (N > 6) or not ParseV4(LPart, LV4) then
          Exit;
        W[N] := LV4[0] shl 8 or LV4[1];
        W[N + 1] := LV4[2] shl 8 or LV4[3];
        Inc(N, 2);
        Continue;
      end;
      if (LPart = '') or (Length(LPart) > 4) or (N > 7) then
        Exit;
      LVal := 0;
      for J := 1 to Length(LPart) do
      begin
        LDigit := HexVal(LPart[J]);
        if LDigit < 0 then
          Exit;
        LVal := LVal shl 4 or LDigit;
      end;
      W[N] := LVal;
      Inc(N);
    end;
    Result := True;
  finally
    LParts.Free;
  end;
end;

function ParseV6(const S: string; out A: TIpAddr): Boolean;
var
  LHead, LTail: array[0..7] of Word;
  LNHead, LNTail, LGap, I: Integer;
  LWords: array[0..7] of Word;
begin
  Result := False;
  FillChar(LWords, SizeOf(LWords), 0);
  LGap := Pos('::', S);
  if LGap > 0 then
  begin
    if Pos('::', Copy(S, LGap + 2, MaxInt)) > 0 then
      Exit;
    if not ParseGroups(Copy(S, 1, LGap - 1), False, LHead, LNHead) or
       not ParseGroups(Copy(S, LGap + 2, MaxInt), True, LTail, LNTail) or
       (LNHead + LNTail > 7) then
      Exit;
    for I := 0 to LNHead - 1 do
      LWords[I] := LHead[I];
    for I := 0 to LNTail - 1 do
      LWords[8 - LNTail + I] := LTail[I];
  end
  else
  begin
    if not ParseGroups(S, True, LHead, LNHead) or (LNHead <> 8) then
      Exit;
    for I := 0 to 7 do
      LWords[I] := LHead[I];
  end;
  for I := 0 to 7 do
  begin
    A.Bytes[I * 2] := LWords[I] shr 8;
    A.Bytes[I * 2 + 1] := LWords[I] and $FF;
  end;
  { ::ffff:a.b.c.d is the IPv4 client of a dual-stack socket }
  A.IsV4 := True;
  for I := 0 to 9 do
    if A.Bytes[I] <> 0 then
      A.IsV4 := False;
  A.IsV4 := A.IsV4 and (A.Bytes[10] = $FF) and (A.Bytes[11] = $FF);
  Result := True;
end;

function ParseIp(const S: string; out A: TIpAddr): Boolean;
var
  LV4: array[0..3] of Byte;
  LText: string;
begin
  FillChar(A, SizeOf(A), 0);
  LText := S;
  if (Length(LText) > 2) and (LText[1] = '[') and (LText[Length(LText)] = ']') then
    LText := Copy(LText, 2, Length(LText) - 2);
  if (LText = '') or (Length(LText) > 45) then
    Exit(False);
  if Pos(':', LText) > 0 then
    Exit(ParseV6(LText, A));
  Result := ParseV4(LText, LV4);
  if Result then
  begin
    A.Bytes[10] := $FF;
    A.Bytes[11] := $FF;
    Move(LV4[0], A.Bytes[12], 4);
    A.IsV4 := True;
  end;
end;

function IpToStr(const A: TIpAddr): string;
var
  LWords: array[0..7] of Word;
  I, LBestAt, LBestLen, LRunAt, LRunLen: Integer;
begin
  if A.IsV4 then
    Exit(Format('%d.%d.%d.%d', [A.Bytes[12], A.Bytes[13], A.Bytes[14], A.Bytes[15]]));
  for I := 0 to 7 do
    LWords[I] := A.Bytes[I * 2] shl 8 or A.Bytes[I * 2 + 1];
  { the longest run of two or more zero groups becomes '::' }
  LBestAt := -1;
  LBestLen := 1;
  LRunAt := -1;
  LRunLen := 0;
  for I := 0 to 7 do
    if LWords[I] = 0 then
    begin
      if LRunAt < 0 then
        LRunAt := I;
      Inc(LRunLen);
      if LRunLen > LBestLen then
      begin
        LBestAt := LRunAt;
        LBestLen := LRunLen;
      end;
    end
    else
    begin
      LRunAt := -1;
      LRunLen := 0;
    end;
  Result := '';
  I := 0;
  while I <= 7 do
  begin
    if I = LBestAt then
    begin
      Result := Result + '::';
      Inc(I, LBestLen);
      Continue;
    end;
    if (Result <> '') and (Result[Length(Result)] <> ':') then
      Result := Result + ':';
    Result := Result + LowerCase(IntToHex(LWords[I], 1));
    Inc(I);
  end;
end;

function ParseCidr(const S: string; out C: TCidr): Boolean;
var
  LSlash, LBits: Integer;
begin
  LSlash := Pos('/', S);
  if LSlash = 0 then
  begin
    Result := ParseIp(S, C.Addr);
    C.Bits := 128;
    Exit;
  end;
  Result := ParseIp(Copy(S, 1, LSlash - 1), C.Addr) and
    TryStrToInt(Copy(S, LSlash + 1, MaxInt), LBits);
  if not Result then
    Exit;
  if C.Addr.IsV4 then
  begin
    Result := (LBits >= 0) and (LBits <= 32);
    C.Bits := 96 + LBits;
  end
  else
  begin
    Result := (LBits >= 0) and (LBits <= 128);
    C.Bits := LBits;
  end;
end;

function CidrMatch(const C: TCidr; const A: TIpAddr): Boolean;
var
  LFull, LRest: Integer;
  LMask: Byte;
begin
  LFull := C.Bits div 8;
  LRest := C.Bits mod 8;
  if (LFull > 0) and not CompareMem(@C.Addr.Bytes[0], @A.Bytes[0], LFull) then
    Exit(False);
  if LRest = 0 then
    Exit(True);
  LMask := Byte($FF shl (8 - LRest));
  Result := (C.Addr.Bytes[LFull] and LMask) = (A.Bytes[LFull] and LMask);
end;

function ParseCidrList(const S: string): TCidrList;
var
  LItems: TStringList;
  LItem: string;
  LCidr: TCidr;
begin
  Result := nil;
  LItems := TStringList.Create;
  try
    LItems.Delimiter := ' ';
    LItems.DelimitedText := StringReplace(S, ',', ' ', [rfReplaceAll]);
    for LItem in LItems do
    begin
      if Trim(LItem) = '' then
        Continue;
      if not ParseCidr(Trim(LItem), LCidr) then
        raise EConvertError.CreateFmt('not an IP address or CIDR range: %s', [LItem]);
      SetLength(Result, Length(Result) + 1);
      Result[High(Result)] := LCidr;
    end;
  finally
    LItems.Free;
  end;
end;

function InCidrList(const L: TCidrList; const A: TIpAddr): Boolean;
var
  LCidr: TCidr;
begin
  for LCidr in L do
    if CidrMatch(LCidr, A) then
      Exit(True);
  Result := False;
end;

end.
