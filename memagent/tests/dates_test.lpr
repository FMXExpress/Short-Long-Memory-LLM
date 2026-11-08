program dates_test;

(* Mem.Dates on fixed cases: dates_test exits 1 on the first mismatch. *)

{$mode objfpc}{$H+}

uses
  SysUtils, Mem.Dates;

var
  Fails: Integer = 0;

procedure Check(const AIn, AWant: string);
var
  Got: string;
begin
  Got := ResolveMessageDates(AIn, EncodeDate(2026, 11, 8));
  if Got <> AWant then
  begin
    WriteLn('FAIL ', AIn, LineEnding, '  got  ', Got, LineEnding, '  want ', AWant);
    Inc(Fails);
  end;
end;

begin
  Check('[1:56 pm on 8 May, 2023] Caroline: I went to a support group yesterday.',
    '[1:56 pm on 8 May, 2023] Caroline: I went to a support group yesterday (7 May 2023).');
  Check('[9:00 am on 25 May, 2023] Melanie: I ran a charity race last Saturday!',
    '[9:00 am on 25 May, 2023] Melanie: I ran a charity race last Saturday (the Saturday before 25 May 2023)!');
  Check('[9:00 am on 25 May, 2023] Melanie: We go camping next month and I painted it last year.',
    '[9:00 am on 25 May, 2023] Melanie: We go camping next month (June 2023) and I painted it last year (2022).');
  Check('[9:00 am on 9 June, 2023] C: I gave a talk last week, 4 years ago I moved.',
    '[9:00 am on 9 June, 2023] C: I gave a talk last week (the week before 9 June 2023), 4 years ago (2019) I moved.');
  Check('[9:00 am on 9 June, 2023] C: that was two days ago, and a few days ago too.',
    '[9:00 am on 9 June, 2023] C: that was two days ago (7 June 2023), and a few days ago (a few days before 9 June 2023) too.');
  Check('I moved here last year.', 'I moved here last year (2025).');
  Check('Last weekend was fun.', 'Last weekend (the weekend before 8 November 2026) was fun.');
  Check('I moved here last year (2025).', 'I moved here last year (2025).');
  Check('My sister lives in Lyon.', 'My sister lives in Lyon.');
  Check('Today is the day before yesterday?', 'Today (8 November 2026) is the day before yesterday (6 November 2026)?');
  Check('[no date here] I left last month.', '[no date here] I left last month (October 2026).');
  if Fails > 0 then
    Halt(1);
  WriteLn('dates_test: all passed');
end.
