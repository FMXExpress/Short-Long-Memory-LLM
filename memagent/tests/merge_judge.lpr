program merge_judge;

(* The fact-merge decision on labelled pairs: the rules (TAgent.SameSubject) and the judge.

     merge_judge --config <ini> [--reuse <earlier out.jsonl>] <merge_cases.json> <out.jsonl>

   The ini supplies the retrieval models and [judge] (bench/jev/make_merge_cases.py makes
   the cases). For each (stored fact, new fact) pair it records

     rules    what SameSubject decides for that one candidate: subject cosine >=
              [retrieval] dedup_min_cos, not DistinctSubjects, cross-encoder >=
              dedup_min_rerank
     p        the judge's P(the new fact replaces the stored one), Mem.Decide.Replaces

   The agent merges when rules and p >= [judge] merge_min; bench/jev/score_merge.py
   scores every combination and threshold. --reuse takes p from an earlier run on the same
   cases, by id, instead of asking the judge: a change to the rules is re-scored in seconds. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Classes, DateUtils, fpjson, jsonparser, Mem.Config, Mem.Retrieval, Mem.Guard, Mem.Decide;

function LoadJson(const APath: string): TJSONData;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := GetJSON(L.Text);
  finally
    L.Free;
  end;
end;

var
  Cfg: TMemConfig;
  Ret: TRetrieval;
  Judge: TJudge;
  Cases: TJSONArray;
  C, O, N: TJSONObject;
  Out_: TStringList;
  I: Integer;
  Cos, Rr, P: Double;
  Distinct, Rules: Boolean;
  Sc: TScoreArray;
  Path, ReusePath: string;
  Start: TDateTime;
  Reused: TStringList;
  K: Integer;
  RJ: TJSONObject;
begin
  ReusePath := '';
  for K := 1 to ParamCount - 1 do
    if ParamStr(K) = '--reuse' then
      ReusePath := ParamStr(K + 1);
  Reused := TStringList.Create;
  Judge := nil;
  Cfg := TMemConfig.Create;
  Cfg.Load;
  Ret := TRetrieval.Create(Cfg.CacheDir, Cfg.Embedder, Cfg.EmbedModel, Cfg.Reranker, Cfg.RerankModel);
  if not Ret.CanDedup or not Ret.HasReranker then
    raise Exception.Create('the merge rules need the onnx embedder and the reranker (memagent --provision)');
  if ReusePath <> '' then
  begin
    Reused.LoadFromFile(ReusePath);
    for K := Reused.Count - 1 downto 0 do
      if Trim(Reused[K]) = '' then
        Reused.Delete(K);
    WriteLn('judge: p reused from ', ReusePath);
  end
  else
  begin
    Path := JudgeModelPath(Cfg, True);
    if Path = '' then
      raise Exception.Create('no judge model: set [judge] model');
    Judge := TJudge.Create(Cfg, Path);
    WriteLn('judge: ', Judge.Describe);
  end;
  Cases := LoadJson(ParamStr(ParamCount - 1)) as TJSONArray;
  Out_ := TStringList.Create;
  for I := 0 to Cases.Count - 1 do
  begin
    C := TJSONObject(Cases[I]);
    O := C.Objects['old'];
    N := C.Objects['new'];
    Cos := Cosine(Ret.Embed(N.Get('subject', '')), Ret.Embed(O.Get('subject', '')));
    Distinct := DistinctSubjects(N.Get('subject', ''), O.Get('subject', ''));
    Sc := Ret.RerankScores(N.Get('subject', ''), [O.Get('subject', '')]);
    Rr := Sc[0];
    Rules := (Cos >= Cfg.DedupMinCos) and not Distinct and (Rr >= Cfg.DedupMinRerank);
    Start := Now;
    if Judge = nil then
    begin
      RJ := GetJSON(Reused[I]) as TJSONObject;
      if RJ.Get('id', -1) <> C.Get('id', 0) then
        raise Exception.Create('--reuse: the earlier run is not over the same cases');
      P := RJ.Get('p', 0.0);
      RJ.Free;
    end
    else
      P := Judge.Replaces(O.Get('text', ''), N.Get('text', ''));
    Out_.Add(Format('{"id": %d, "replaces": %d, "category": "%s", "cos": %.3f, "distinct": %s, ' +
      '"rerank": %.4f, "rules": %s, "p": %.4f, "ms": %d}',
      [C.Get('id', 0), C.Get('replaces', 0), StringToJSONString(C.Get('category', '')), Cos,
       BoolToStr(Distinct, 'true', 'false'), Rr, BoolToStr(Rules, 'true', 'false'), P,
       MilliSecondsBetween(Now, Start)]));
    Out_.SaveToFile(ParamStr(ParamCount));
    WriteLn(Format('%d/%d label=%d rules=%s p=%.3f  %s -> %s', [I + 1, Cases.Count, C.Get('replaces', 0),
      BoolToStr(Rules, 'merge', 'keep'), P, O.Get('text', ''), N.Get('text', '')]));
  end;
  Out_.Free;
  Reused.Free;
  Judge.Free;
  Ret.Free;
  Cfg.Free;
end.
