program priors_test;

(* Mem.Priors and the priors tables of Mem.Store:
   - which sentences are standing instructions, and the belief they become
   - restating confirms, negating contradicts, unrelated stays apart
   - each message counts once; contradiction deprecates, confirmation restores *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  SysUtils, Mem.Priors, Mem.Store;

var
  Fails, Checks: Integer;

procedure Check(ACond: Boolean; const AWhat: string);
begin
  Inc(Checks);
  if not ACond then
  begin
    Inc(Fails);
    WriteLn('FAIL  ', AWhat);
  end;
end;

procedure Detects(const AText, ABelief: string; APolarity: Integer);
var
  C: TPriorCandidates;
begin
  C := DetectPriors(AText);
  Check((Length(C) = 1) and (C[0].Belief = ABelief) and (C[0].Polarity = APolarity),
    Format('"%s" -> "%s" (%d), got %d: %s', [AText, ABelief, APolarity, Length(C),
    BoolToStr(Length(C) > 0, C[0].Belief, '-')]));
end;

procedure Ignores(const AText: string);
begin
  Check(Length(DetectPriors(AText)) = 0, Format('"%s" is not a standing instruction', [AText]));
end;

function Match(const A, B: string): TPriorMatch;
var
  CA, CB: TPriorCandidate;
begin
  CA := PriorFromText(A);
  CB := PriorFromText(B);
  Result := MatchPrior(CA.Key, CA.Polarity, CB.Key, CB.Polarity);
end;

function Find(AStore: TMemStore; AId: Int64): TPriorRec;
var
  P: TPriorRec;
begin
  for P in AStore.Priors(False) do
    if P.Id = AId then
      Exit(P);
  raise Exception.CreateFmt('no prior #%d', [AId]);
end;

var
  Store: TMemStore;
  Db: string;
  Id: Int64;
  P: TPriorRec;
begin
  Fails := 0;
  Checks := 0;

  { detection }
  Detects('From now on, keep your answers to one sentence.', 'Keep your answers to one sentence.', 1);
  Detects('Please never use emojis.', 'Never use emojis.', -1);
  Detects('Don''t use emojis.', 'Don''t use emojis.', -1);
  Detects('Actually, you can use emojis.', 'Use emojis.', 1);
  Detects('I prefer short answers.', 'The user prefers short answers.', 1);
  Detects('I''d like you to explain with code examples.', 'Explain with code examples.', 1);
  Detects('Stop apologizing.', 'Avoid apologizing.', -1);
  Detects('Going forward, answer in metric units.', 'Answer in metric units.', 1);
  Detects('My dog''s name is Biscuit. From now on, call me Eli.', 'Call me Eli.', 1);
  Ignores('Don''t worry about it.');
  Ignores('Never mind.');
  Ignores('I prefer tea.');
  Ignores('Do you always use emojis?');
  Ignores('Use Python for this.');
  Ignores('My dog''s name is Biscuit.');

  { matching }
  Check(Match('Never use emojis.', 'Don''t use emojis.') = pmSame, 'never = don''t');
  Check(Match('Never use emojis.', 'Always use emojis.') = pmOpposite, 'never vs always');
  Check(Match('Avoid using emojis.', 'Use emojis.') = pmOpposite, 'avoid using vs use');
  Check(Match('Never use emojis.', 'Keep your answers short.') = pmNone, 'unrelated');

  { evidence, deprecation, restoring }
  Db := GetTempFileName('', 'priors') + '.db';
  Store := TMemStore.Create(Db);
  try
    Id := Store.AddPrior('Never use emojis.', PriorFromText('Never use emojis.').Key, 'learned', '', -1, 1, 1);
    Check(Store.AddPriorEvidence(Id, 10, 1, 'said', 0.6, 0.4), 'first evidence counts');
    Check(not Store.AddPriorEvidence(Id, 10, 1, 'said again', 0.6, 0.4), 'the same message counts once');
    P := Find(Store, Id);
    Check((P.Alpha = 2) and (P.Beta = 1) and (P.Evidence = 1), 'alpha 2, beta 1 after one message');
    Check(Length(Store.Priors(True, 0.6)) = 1, '0.67 is in the prompt');
    Store.AddPriorEvidence(Id, 11, -1, 'contradicted', 0.6, 0.4);
    Check(Length(Store.Priors(True, 0.6)) = 0, '0.50 is out of the prompt');
    Check(not Find(Store, Id).Deprecated, '0.50 is not deprecated');
    Store.AddPriorEvidence(Id, 12, -1, 'contradicted again', 0.6, 0.4);
    Check(Find(Store, Id).Deprecated, 'contradicted down to 0.40: deprecated, not deleted');
    Store.AddPriorEvidence(Id, 13, 1, 'restated', 0.6, 0.4);
    Store.AddPriorEvidence(Id, 14, 1, 'restated', 0.6, 0.4);
    Store.AddPriorEvidence(Id, 15, 1, 'restated', 0.6, 0.4);
    Check(not Find(Store, Id).Deprecated, 'confirmed back to 0.62: restored');
    Check(Store.AddPriorEvidence(Id, 0, 1, '/prior', 0.6, 0.4) and Store.AddPriorEvidence(Id, 0, 1, '/prior', 0.6, 0.4),
      'command evidence (no message) always counts');
    Check(Store.SetPriorDeprecated(Id, True, 'dropped') and Find(Store, Id).Deprecated, '/prior drop');
    Check(Length(Store.Priors(True, 0.0)) = 0, 'a dropped prior is out of the prompt at any confidence');
  finally
    Store.Free;
    DeleteFile(Db);
  end;

  if Fails = 0 then
    WriteLn('priors_test: ', Checks, '/', Checks, ' checks passed')
  else
  begin
    WriteLn('priors_test: ', Fails, ' of ', Checks, ' checks FAILED');
    Halt(1);
  end;
end.
