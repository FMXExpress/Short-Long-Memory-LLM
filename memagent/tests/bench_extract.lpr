program bench_extract;

(* Extraction coverage: how many facts, said the way people say them, the
   rule extractor (Mem.Extract, the tiny profile's only extractor) stores at
   all, with the right detail.

     bench_extract bench/facts_natural.json [dev|heldout] [-v]
     bench_extract bench/chatter.json lines [-v]     false positives: lines that state no fact

   A fact counts when some extracted fact's value contains the expected
   value (case-insensitive). -v lists the misses and what was extracted. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  SysUtils, Classes, StrUtils, fpjson, jsonparser, Mem.Extract;

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

var
  Data: TJSONObject;
  Items, Got: TJSONArray;
  I, J, Hits: Integer;
  Split, Want, Seen: string;
  Ok, Verbose: Boolean;
begin
  Split := 'dev';
  if ParamCount >= 2 then
    Split := ParamStr(2);
  Verbose := (ParamCount >= 3) and (ParamStr(3) = '-v');
  Data := LoadJson(ParamStr(1));
  try
    if Split = 'lines' then
    begin
      Hits := 0;
      Items := Data.Arrays['lines'];
      for I := 0 to Items.Count - 1 do
      begin
        Got := ExtractFactsRules(Items.Strings[I]);
        try
          if Got.Count > 0 then
          begin
            Inc(Hits);
            if Verbose then
              WriteLn('  false positive: ', Items.Strings[I], '  -> ', Got.Objects[0].Get('text', ''));
          end;
        finally
          Got.Free;
        end;
      end;
      WriteLn(Format('false positives (chatter): %d/%d = %.3f', [Hits, Items.Count, Hits / Items.Count]));
      Exit;
    end;
    Items := Data.Arrays[Split];
    Hits := 0;
    for I := 0 to Items.Count - 1 do
    begin
      Want := LowerCase(Items.Objects[I].Get('value', ''));
      Got := ExtractFactsRules(Items.Objects[I].Get('say', ''));
      try
        Ok := False;
        Seen := '';
        for J := 0 to Got.Count - 1 do
        begin
          if Pos(Want, LowerCase(Got.Objects[J].Get('value', ''))) > 0 then
            Ok := True;
          Seen := Seen + ' | ' + Got.Objects[J].Get('text', '') + ' {' + Got.Objects[J].Get('subject', '') + ' = ' +
            Got.Objects[J].Get('value', '') + '}';
        end;
        if Ok then
          Inc(Hits)
        else if Verbose then
          WriteLn('  miss: ', Items.Objects[I].Get('say', ''), '  -> ', IfThen(Seen = '', '(nothing)', Seen));
      finally
        Got.Free;
      end;
    end;
    WriteLn(Format('extraction coverage (%s): %d/%d = %.3f', [Split, Hits, Items.Count, Hits / Items.Count]));
  finally
    Data.Free;
  end;
end.
