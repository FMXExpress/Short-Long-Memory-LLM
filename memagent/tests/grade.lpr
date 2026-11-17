program grade;

(* A benchmark's answers graded by the decision model (Mem.Decide.TJudge.Correct).

     grade [--config memagent.ini] IN.json OUT.json

   IN: a JSON array of {"q": question, "accept": [expected, ...], "answer": text}.
   OUT: the same array, each item with "p_correct" (calibrated) and "ms". The judge's
   GGUF is [judge] model, or OpenJev-Flash-9B in <cache>/jev (downloaded once). *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cwstring,{$ENDIF}
  SysUtils, Classes, fpjson, jsonparser, Mem.Config, Mem.Decide;

var
  Cfg: TMemConfig;
  Judge: TJudge;
  LIn: TJSONArray;
  L: TStringList;
  I, J, LArg: Integer;
  LItem: TJSONObject;
  LAccept: TJSONArray;
  LExp: array of string;
  LPath, LInFile, LOutFile: string;
begin
  LInFile := '';
  LOutFile := '';
  LArg := 1;
  while LArg <= ParamCount do
  begin
    if (ParamStr(LArg) = '--config') or (ParamStr(LArg) = '--data') or (ParamStr(LArg) = '--profile') then
      Inc(LArg)
    else if LInFile = '' then
      LInFile := ParamStr(LArg)
    else
      LOutFile := ParamStr(LArg);
    Inc(LArg);
  end;
  if LOutFile = '' then
  begin
    WriteLn(StdErr, 'usage: grade [--config memagent.ini] IN.json OUT.json');
    Halt(2);
  end;
  Cfg := TMemConfig.Create;
  L := TStringList.Create;
  try
    Cfg.Load;
    Cfg.InferGpu := -1;
    if GetEnvironmentVariable('GRADE_GPU') <> '' then
      Cfg.InferGpu := StrToIntDef(GetEnvironmentVariable('GRADE_GPU'), -1);
    { [judge] model, or OpenJev when the agent's judge is off }
    if (Cfg.JudgeModel = '') or SameText(Cfg.JudgeModel, 'off') then
      Cfg.JudgeModel := 'openjev-flash-9b';
    LPath := JudgeModelPath(Cfg, True);
    if LPath = '' then
    begin
      WriteLn(StdErr, 'grade: no judge model');
      Halt(1);
    end;
    L.LoadFromFile(LInFile);
    LIn := GetJSON(L.Text) as TJSONArray;
    Judge := TJudge.Create(Cfg, LPath);
    try
      WriteLn(StdErr, 'grade: ', Judge.Describe, ', ', LIn.Count, ' answers');
      Flush(StdErr);
      for I := 0 to LIn.Count - 1 do
      begin
        LItem := LIn.Objects[I];
        LAccept := LItem.Arrays['accept'];
        SetLength(LExp, LAccept.Count);
        for J := 0 to LAccept.Count - 1 do
          LExp[J] := LAccept.Strings[J];
        LItem.Add('p_correct', Judge.Correct(LItem.Get('q', ''), LItem.Get('answer', ''), LExp));
        LItem.Add('ms', Judge.LastMs);
        if (I + 1) mod 10 = 0 then
        begin
          WriteLn(StdErr, 'grade: ', I + 1, '/', LIn.Count, ' (', Judge.LastMs, ' ms the last)');
          Flush(StdErr);
        end;
      end;
      L.Text := LIn.FormatJSON;
      L.SaveToFile(LOutFile);
    finally
      Judge.Free;
      LIn.Free;
    end;
  finally
    L.Free;
    Cfg.Free;
  end;
end.
