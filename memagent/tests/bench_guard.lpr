program bench_guard;

(* Offline evaluation of Mem.Guard on answers a model actually gave.

     bench_guard <cache_dir> <facts.json> <bench_memory result.json>

   For each condition recorded in the result file, the guard replaces
   unlicensed answers with the abstention; accuracy (known questions) and
   confabulation (unknown questions) are reported before and after, plus how
   many correct answers the guard wrongly blocked. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Classes, fpjson, jsonparser, Mem.Store, Mem.Retrieval, Mem.Guard;

function LoadJson(const APath: string): TJSONObject;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := GetJSON(L.Text) as TJSONObject;
  finally
    L.Free;
  end;
end;

function Norm(const S: string): string;
var
  I: Integer;
begin
  Result := ' ' + LowerCase(S) + ' ';
  for I := 1 to Length(Result) do
    if not (Result[I] in ['a'..'z', '0'..'9', #128..#255]) then
      Result[I] := ' ';
  while Pos('  ', Result) > 0 do
    Result := StringReplace(Result, '  ', ' ', [rfReplaceAll]);
end;

function Has(const AAnswer, AValue: string): Boolean;
begin
  Result := Pos(Norm(AValue), Norm(AAnswer)) > 0;
end;

var
  Ret: TRetrieval;
  Data, Res: TJSONObject;
  Facts: TFactArray;
  Rows, URows: TJSONArray;
  Conds: array[0..2] of string = ('lora', 'lora+rag', 'rag');
  UConds: array[0..2] of string = ('lora', 'lora+rag-near', 'rag-near');
  I, C, K, N, Before, After, Blocked, FalseBlocked, AnyVal: Integer;
  Row: TJSONObject;
  Ans, Why: string;
begin
  Ret := TRetrieval.Create(ParamStr(1), 'off', '', 'auto', 'bge-reranker-base');
  Data := LoadJson(ParamStr(2));
  Res := LoadJson(ParamStr(3));
  SetLength(Facts, Data.Arrays['facts'].Count);
  for I := 0 to High(Facts) do
  begin
    Facts[I].Id := I + 1;
    Facts[I].Subject := Data.Arrays['facts'].Objects[I].Get('subject', '');
    Facts[I].Value := Data.Arrays['facts'].Objects[I].Get('value', '');
    Facts[I].Text := Data.Arrays['facts'].Objects[I].Get('text', '');
  end;
  Rows := Res.Arrays['answers'];
  URows := Res.Arrays['unknown_answers'];
  WriteLn(Format('%s (%s): %d known, %d unknown questions', [Res.Get('label', ''), ParamStr(3), Rows.Count, URows.Count]));
  for C := 0 to High(Conds) do
  begin
    Before := 0; After := 0; Blocked := 0; FalseBlocked := 0;
    for I := 0 to Rows.Count - 1 do
    begin
      Row := Rows.Objects[I];
      Ans := Row.Get(Conds[C], '');
      if Has(Ans, Row.Get('value', '')) then Inc(Before);
      if not Licensed(Ret, Row.Get('q', ''), Ans, Facts, Why) then
      begin
        Inc(Blocked);
        if Has(Ans, Row.Get('value', '')) then
        begin
          Inc(FalseBlocked);
          if GetEnvironmentVariable('GUARD_DEBUG') = '1' then
            WriteLn(Format('    false block: %s [subject "%s"] -> %s | blocked by: %s',
              [Row.Get('q', ''), QuestionSubject(Row.Get('q', '')), Copy(Ans, 1, 50), Why]));
        end;
        Ans := ABSTAIN_REPLY;
      end;
      if Has(Ans, Row.Get('value', '')) then Inc(After);
    end;
    WriteLn(Format('  known   %-14s accuracy %.3f -> %.3f   (blocked %d, of them correct %d)',
      [Conds[C], Before / Rows.Count, After / Rows.Count, Blocked, FalseBlocked]));
  end;
  for C := 0 to High(UConds) do
  begin
    Before := 0; After := 0; N := 0;
    for I := 0 to URows.Count - 1 do
    begin
      Row := URows.Objects[I];
      if Row.Find(UConds[C]) = nil then Continue;
      Inc(N);
      Ans := Row.Get(UConds[C], '');
      AnyVal := 0;
      for K := 0 to High(Facts) do
        if Has(Ans, Facts[K].Value) then AnyVal := 1;
      Inc(Before, AnyVal);
      if not Licensed(Ret, Row.Get('q', ''), Ans, Facts, Why) then
        Ans := ABSTAIN_REPLY;
      AnyVal := 0;
      for K := 0 to High(Facts) do
        if Has(Ans, Facts[K].Value) then AnyVal := 1;
      Inc(After, AnyVal);
    end;
    if N > 0 then
      WriteLn(Format('  unknown %-14s confabulation %.3f -> %.3f   (n=%d)', [UConds[C], Before / N, After / N, N]));
  end;
  Res.Free;
  Data.Free;
  Ret.Free;
end.
