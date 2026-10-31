unit Ban.Jev;

(* Client for a Jev-style decision API (POST /v1/systemone): the hosted Jev
   API, or OpenJev (huggingface.co/openjev/openjev) behind its helper shim,
   which keeps the same request and response shapes. The model gets no
   training for this task; the decision is described in plain words at
   request time:

     state     = the site description, the client's recent requests and the
                 request that tripped a signature
     questions = "malicious": noul (calibrated probability of yes)
                 "intent":    choice over what the client is doing

   and the answers come back as numbers, not free text:
     {"answers": {"malicious": {"type": "noul", "noul": 0.97},
                  "intent": {"type": "choice", "choice": "vulnerability_scanner",
                             "probabilities": {...}, "confidence": 0.9}}}

   Log lines are written by the attacker, so they go in as quoted data in the
   state, never into the question; the readout is a single option letter, so
   the worst a crafted line can do is argue for a lower probability. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, fpjson, jsonparser, DateUtils;

type
  TJevVerdict = record
    Ok: Boolean;
    Error: string;
    Malicious: Double;        { calibrated P(yes) }
    Intent: string;
    IntentConfidence: Double;
    LatencyMs: Int64;
  end;

  TJevClient = class
  public
    Url: string;
    ApiKey: string;
    Model: string;
    Site: string;            { what this server runs, in plain words }
    Question: string;
    TimeoutMs: Integer;
    constructor Create;
    function BuildRequest(const AIp, AFlagged, ATags: string; ARecent: TStrings): string;
    function Decide(const AIp, AFlagged, ATags: string; ARecent: TStrings): TJevVerdict;
  end;

function ParseJevResponse(const ABody: string; out V: TJevVerdict): Boolean;

const
  CDefaultQuestion = 'Is this client probing for or exploiting security vulnerabilities on this ' +
    'server (vulnerability scanning, exploit attempts, hunting for secrets, admin panels or ' +
    'backups, brute-forcing logins), rather than browsing normally or crawling as a legitimate ' +
    'search engine?';
  CDefaultSite = 'A public web server. Judge requests against what a normal visitor or a ' +
    'well-behaved search engine would ask for.';

implementation

uses
  Mem.Https;

const
  CIntents: array[0..5, 0..1] of string = (
    ('legitimate_visitor', 'a person or app using the site normally, including the odd broken link'),
    ('search_crawler', 'a well-behaved search engine or monitoring bot fetching public pages'),
    ('vulnerability_scanner', 'probing many paths for known vulnerable software, admin panels, ' +
      'leaked secrets, backups or config files'),
    ('exploit_attempt', 'sending an attack payload: injection, path traversal, remote code ' +
      'execution, web shell access'),
    ('credential_attack', 'brute-forcing or stuffing passwords against a login'),
    ('abusive_scraper', 'aggressively harvesting content or bandwidth'));

constructor TJevClient.Create;
begin
  inherited Create;
  Url := 'http://127.0.0.1:3000/v1/systemone';
  Model := 'openjev';
  Site := CDefaultSite;
  Question := CDefaultQuestion;
  TimeoutMs := 8000;
end;

function TJevClient.BuildRequest(const AIp, AFlagged, ATags: string; ARecent: TStrings): string;
var
  LRoot, LState, LQuestions, LQ, LCrit: TJSONObject;
  LRecent: TJSONArray;
  I: Integer;
begin
  LRoot := TJSONObject.Create;
  try
    LRoot.Add('model', Model);
    LState := TJSONObject.Create;
    LState.Add('server', Site);
    LState.Add('client_ip', AIp);
    LState.Add('flagged_request', AFlagged);
    LState.Add('signature_hits', ATags);
    LRecent := TJSONArray.Create;
    if ARecent <> nil then
      for I := 0 to ARecent.Count - 1 do
        LRecent.Add(ARecent[I]);
    LState.Add('recent_requests_from_this_client_oldest_first', LRecent);
    LRoot.Add('state', LState);

    LQuestions := TJSONObject.Create;
    LQ := TJSONObject.Create;
    LQ.Add('type', 'noul');
    LQ.Add('instructions', Question);
    LCrit := TJSONObject.Create;
    LCrit.Add('true', 'The client is attacking or probing the server and should be blocked.');
    LCrit.Add('false', 'The client is a legitimate visitor or crawler, even if a request failed.');
    LQ.Add('criteria', LCrit);
    LQuestions.Add('malicious', LQ);

    LQ := TJSONObject.Create;
    LQ.Add('type', 'choice');
    LQ.Add('instructions', 'What is this client doing?');
    LCrit := TJSONObject.Create;
    for I := 0 to High(CIntents) do
      LCrit.Add(CIntents[I, 0], CIntents[I, 1]);
    LQ.Add('criteria', LCrit);
    LQuestions.Add('intent', LQ);
    LRoot.Add('questions', LQuestions);
    Result := LRoot.AsJSON;
  finally
    LRoot.Free;
  end;
end;

function ParseJevResponse(const ABody: string; out V: TJevVerdict): Boolean;
var
  LData: TJSONData;
  LAnswers, LMal, LIntent: TJSONObject;
begin
  V := Default(TJevVerdict);
  V.Malicious := -1;
  Result := False;
  try
    LData := GetJSON(ABody);
  except
    on E: Exception do
    begin
      V.Error := 'response is not JSON: ' + Copy(ABody, 1, 120);
      Exit;
    end;
  end;
  try
    if not (LData is TJSONObject) or
       (TJSONObject(LData).Find('answers', LAnswers) = False) then
    begin
      V.Error := 'response has no "answers": ' + Copy(ABody, 1, 160);
      Exit;
    end;
    if LAnswers.Find('malicious', LMal) then
      V.Malicious := LMal.Get('noul', -1.0);
    if LAnswers.Find('intent', LIntent) then
    begin
      V.Intent := LIntent.Get('choice', '');
      V.IntentConfidence := LIntent.Get('confidence', 0.0);
    end;
    if (V.Malicious < 0) or (V.Malicious > 1) then
    begin
      V.Error := 'answers.malicious.noul missing or out of range';
      Exit;
    end;
    V.Ok := True;
    Result := True;
  finally
    LData.Free;
  end;
end;

function TJevClient.Decide(const AIp, AFlagged, ATags: string; ARecent: TStrings): TJevVerdict;
var
  LBody, LResp: string;
  LStatus: Integer;
  LStart: TDateTime;
begin
  LBody := BuildRequest(AIp, AFlagged, ATags, ARecent);
  LStart := Now;
  try
    if ApiKey <> '' then
      LResp := HttpPostJson(Url, LBody, ['Authorization', 'Bearer ' + ApiKey, 'User-Agent', 'jevban/1.0'],
        TimeoutMs, LStatus)
    else
      LResp := HttpPostJson(Url, LBody, ['User-Agent', 'jevban/1.0'], TimeoutMs, LStatus);
  except
    on E: Exception do
    begin
      Result := Default(TJevVerdict);
      Result.Error := Format('%s: %s', [E.ClassName, E.Message]);
      Result.LatencyMs := MilliSecondsBetween(Now, LStart);
      Exit;
    end;
  end;
  if LStatus <> 200 then
  begin
    Result := Default(TJevVerdict);
    Result.Error := Format('HTTP %d: %s', [LStatus, Copy(LResp, 1, 160)]);
  end
  else
    ParseJevResponse(LResp, Result);
  Result.LatencyMs := MilliSecondsBetween(Now, LStart);
end;

end.
