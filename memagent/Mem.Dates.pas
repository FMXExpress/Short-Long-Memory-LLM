unit Mem.Dates;

(* Relative dates made absolute when a message is stored.

   "I ran a charity race last Saturday." is stored as "I ran a charity race last
   Saturday (the Saturday before 25 May 2023).", against the date the message was said.
   Weeks later "last Saturday" means another day, and a small model asked "when?" repeats
   the words it sees; with the date beside them it can read the answer off. LoCoMo's
   temporal questions were memagent's weakest (F1 0.11, bench/README.md).

   A message that starts with a bracketed date, as a recorded conversation loaded with
   /log does ("[1:56 pm on 8 May, 2023] Caroline: ..."), is resolved against that date;
   any other against the time it is stored. The original words stay; the date is added
   in parentheses after them, once (a second pass adds nothing). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, DateUtils, StrUtils, Math, RegExpr;

{ the date in a leading "[... 8 May, 2023 ...]", if the message starts with one }
function LeadDate(const AText: string; out ADate: TDateTime): Boolean;

{ AText with an absolute date after each relative one, against ARef }
function ResolveDates(const AText: string; ARef: TDateTime): string;

{ ResolveDates against the message's own leading date, or else ANow }
function ResolveMessageDates(const AText: string; ANow: TDateTime): string;

implementation

const
  MONTHS: array[1..12] of string = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December');
  { DayOfTheWeek: 1 = Monday }
  WEEKDAYS: array[1..7] of string = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
    'Sunday');
  NUMBERS: array[1..10] of string = ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten');

function NewRx(const AExpr: string): TRegExpr;
begin
  Result := TRegExpr.Create;
  Result.ModifierI := True;
  Result.Expression := AExpr;
end;

function DayText(D: TDateTime): string;
begin
  Result := Format('%d %s %d', [DayOf(D), MONTHS[MonthOf(D)], YearOf(D)]);
end;

function MonthText(D: TDateTime): string;
begin
  Result := MONTHS[MonthOf(D)] + ' ' + IntToStr(YearOf(D));
end;

function MonthIndex(const S: string): Integer;
var
  I: Integer;
begin
  for I := 1 to 12 do
    if SameText(Copy(MONTHS[I], 1, 3), Copy(S, 1, 3)) then
      Exit(I);
  Result := 0;
end;

function WeekdayIndex(const S: string): Integer;
var
  I: Integer;
begin
  for I := 1 to 7 do
    if SameText(WEEKDAYS[I], S) then
      Exit(I);
  Result := 0;
end;

function Count(const S: string): Integer;
var
  I: Integer;
  T: string;
begin
  T := LowerCase(Trim(S));
  if (T = 'a') or (T = 'an') then
    Exit(1);
  if (T = 'a couple of') or (T = 'couple of') then
    Exit(2);
  if (T = 'a few') or (T = 'few') then
    Exit(3);
  for I := 1 to 10 do
    if T = NUMBERS[I] then
      Exit(I);
  Result := StrToIntDef(T, 0);
end;

function LeadDate(const AText: string; out ADate: TDateTime): Boolean;
var
  R: TRegExpr;
  M: Integer;
begin
  Result := False;
  ADate := 0;
  R := NewRx('^\s*\[[^\]]*?(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})[^\]]*\]');
  try
    if R.Exec(AText) then
    begin
      M := MonthIndex(R.Match[2]);
      if M > 0 then
        Result := TryEncodeDate(StrToInt(R.Match[3]), M, StrToInt(R.Match[1]), ADate);
    end;
  finally
    R.Free;
  end;
end;

{ the absolute form of one matched phrase, or '' to leave it as it is }
function Absolute(const APhrase: string; ARef: TDateTime): string;
var
  R: TRegExpr;
  P, Unit_, Which: string;
  N, W, Back: Integer;
begin
  Result := '';
  P := LowerCase(APhrase);
  if (P = 'the day before yesterday') or (P = 'day before yesterday') then
    Exit(DayText(IncDay(ARef, -2)));
  if (P = 'yesterday') or (P = 'last night') then
    Exit(DayText(IncDay(ARef, -1)));
  if (P = 'today') or (P = 'tonight') or (P = 'this morning') or (P = 'this afternoon') or
    (P = 'this evening') then
    Exit(DayText(ARef));
  if P = 'tomorrow' then
    Exit(DayText(IncDay(ARef, 1)));
  R := NewRx('^(last|next|this) (\w+)$');
  try
    if R.Exec(P) then
    begin
      Which := R.Match[1];
      Unit_ := R.Match[2];
      if Unit_ = 'week' then
        Exit(IfThen(Which = 'last', 'the week before ', IfThen(Which = 'next', 'the week after ', 'the week of ')) +
          DayText(ARef));
      if Unit_ = 'weekend' then
        Exit(IfThen(Which = 'last', 'the weekend before ', 'the weekend after ') + DayText(ARef));
      if Unit_ = 'month' then
        Exit(MonthText(IncMonth(ARef, IfThen(Which = 'last', -1, IfThen(Which = 'next', 1, 0)))));
      if Unit_ = 'year' then
        Exit(IntToStr(YearOf(ARef) + IfThen(Which = 'last', -1, IfThen(Which = 'next', 1, 0))));
      W := WeekdayIndex(Unit_);
      if W > 0 then
      begin
        if Which = 'last' then
          Exit('the ' + WEEKDAYS[W] + ' before ' + DayText(ARef));
        Exit('the ' + WEEKDAYS[W] + ' after ' + DayText(ARef));
      end;
      Exit('');
    end;
  finally
    R.Free;
  end;
  R := NewRx('^(.+?) (day|week|month|year)s? ago$');
  try
    if R.Exec(P) then
    begin
      N := Count(R.Match[1]);
      if N <= 0 then
        Exit('');
      Unit_ := R.Match[2];
      if Unit_ = 'day' then
        Exit(IfThen(N = 3, 'a few days before ' + DayText(ARef), DayText(IncDay(ARef, -N))));
      if Unit_ = 'week' then
        Exit('the week of ' + DayText(IncDay(ARef, -7 * N)));
      if Unit_ = 'month' then
        Exit(MonthText(IncMonth(ARef, -N)));
      Back := N;
      Exit(IntToStr(YearOf(ARef) - Back));
    end;
  finally
    R.Free;
  end;
end;

function ResolveDates(const AText: string; ARef: TDateTime): string;
var
  R: TRegExpr;
  Last, After: Integer;
  Abs_: string;
begin
  if ARef = 0 then
    Exit(AText);
  R := NewRx('\b((the )?day before yesterday|yesterday|last night|tonight|today|this (morning|afternoon|evening)|' +
    'tomorrow|(last|next|this) (weekend|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|' +
    '(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|a few|few|a couple of|couple of) ' +
    '(days?|weeks?|months?|years?) ago)\b');
  Result := '';
  Last := 1;
  try
    if R.Exec(AText) then
      repeat
        After := R.MatchPos[0] + R.MatchLen[0];
        Result := Result + Copy(AText, Last, After - Last);
        Last := After;
        { already resolved: "last week (the week before ...)" }
        if Copy(AText, After, 2) <> ' (' then
        begin
          Abs_ := Absolute(R.Match[0], ARef);
          if Abs_ <> '' then
            Result := Result + ' (' + Abs_ + ')';
        end;
      until not R.ExecNext;
    Result := Result + Copy(AText, Last, MaxInt);
  finally
    R.Free;
  end;
end;

function ResolveMessageDates(const AText: string; ANow: TDateTime): string;
var
  D: TDateTime;
begin
  if not LeadDate(AText, D) then
    D := ANow;
  Result := ResolveDates(AText, D);
end;

end.
