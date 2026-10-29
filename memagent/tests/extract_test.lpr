program extract_test;

(* Mem.Extract must match trainer/memlib.py: every case in extract_cases.json
   (dumped by gen_extract_cases.py) gives the same facts and has_statement. *)

{$mode objfpc}{$H+}

uses
  {$ifdef unix}cwstring,{$endif} SysUtils, Classes, fpjson, jsonparser, Mem.Utf8, Mem.Extract;

var
  GFail, GCount: Integer;

function SameFacts(AWant, AGot: TJSONArray; out AWhy: string): Boolean;
var
  I: Integer;
  W, G: TJSONObject;
  K: string;
begin
  AWhy := '';
  if AWant.Count <> AGot.Count then
  begin
    AWhy := Format('%d facts, want %d', [AGot.Count, AWant.Count]);
    Exit(False);
  end;
  for I := 0 to AWant.Count - 1 do
  begin
    W := AWant.Objects[I];
    G := AGot.Objects[I];
    for K in ['text', 'subject', 'value', 'kind'] do
      if W.Get(K, '') <> G.Get(K, '') then
        AWhy := AWhy + Format(' %s=%s want %s', [K, G.Get(K, ''), W.Get(K, '')]);
    if Abs(W.Get('confidence', 0.0) - G.Get('confidence', 0.0)) > 1e-9 then
      AWhy := AWhy + Format(' confidence=%g want %g', [G.Get('confidence', 0.0), W.Get('confidence', 0.0)]);
    if W.Arrays['questions'].AsJSON <> G.Arrays['questions'].AsJSON then
      AWhy := AWhy + ' questions ' + G.Arrays['questions'].AsJSON;
  end;
  Result := AWhy = '';
end;

procedure Check(AOk: Boolean; const AWhat, AWhy: string);
begin
  Inc(GCount);
  if not AOk then
  begin
    Inc(GFail);
    WriteLn('FAIL ', AWhat, ': ', AWhy);
  end;
end;

var
  LCases: TJSONObject;
  LStream: TFileStream;
  LCase: TJSONObject;
  LGot: TJSONArray;
  I: Integer;
  LWhy, LFile: string;
begin
  LFile := ExtractFilePath(ParamStr(0)) + '../memagent/tests/extract_cases.json';
  if ParamCount > 0 then
    LFile := ParamStr(1);
  LStream := TFileStream.Create(LFile, fmOpenRead);
  try
    LCases := GetJSON(LStream) as TJSONObject;
  finally
    LStream.Free;
  end;
  for I := 0 to LCases.Arrays['rules'].Count - 1 do
  begin
    LCase := LCases.Arrays['rules'].Objects[I];
    LGot := ExtractFactsRules(LCase.Get('text', ''));
    try
      Check(SameFacts(LCase.Arrays['facts'], LGot, LWhy), 'rules ' + LCase.Get('text', ''), LWhy);
    finally
      LGot.Free;
    end;
    Check(HasStatement(LCase.Get('text', '')) = LCase.Get('has_statement', False),
      'has_statement ' + LCase.Get('text', ''), BoolToStr(not LCase.Get('has_statement', False), True));
  end;
  for I := 0 to LCases.Arrays['llm'].Count - 1 do
  begin
    LCase := LCases.Arrays['llm'].Objects[I];
    LGot := FactsFromLlmOutput(LCase.Get('out', ''), LCase.Get('source', ''));
    try
      Check(SameFacts(LCase.Arrays['facts'], LGot, LWhy), 'llm ' + LCase.Get('out', ''), LWhy);
    finally
      LGot.Free;
    end;
  end;
  LCases.Free;
  WriteLn(Format('extract_test: %d/%d checks passed', [GCount - GFail, GCount]));
  if GFail > 0 then
    Halt(1);
end.
