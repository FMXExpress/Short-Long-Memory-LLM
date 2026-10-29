program retrieval_test;

(* Checks the local retrieval models against hand-picked pairs, and pins the
   observations the memagent.ini thresholds are based on. Needs the models in
   <cache_dir> (memagent --provision); without them only the static tier runs.
   Usage: retrieval_test <cache_dir> *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Math, Mem.Retrieval;

var
  Failures: Integer = 0;

procedure Check(ACond: Boolean; const AWhat: string);
begin
  if ACond then
    WriteLn('  ok    ', AWhat)
  else
  begin
    WriteLn('  FAIL  ', AWhat);
    Inc(Failures);
  end;
end;

function Cos(R: TRetrieval; const A, B: string): Single;
begin
  Result := Cosine(R.Embed(A), R.Embed(B));
end;

function Rr(R: TRetrieval; const A, B: string): Single;
begin
  Result := R.RerankScores(A, [B])[0];
end;

{ the agent's same-subject rule (TAgent.SameSubject) }
function SameSubject(R: TRetrieval; const A, B: string): Boolean;
begin
  Result := (Cos(R, A, B) >= 0.7) and (Rr(R, A, B) >= 0.5);
end;

procedure BatchMatchesSingle(R: TRetrieval);
const
  Docs: array[0..3] of string = ('Your dog''s name is Biscuit.', 'ok',
    'The wifi password is tulip42 and the guest network is called Lighthouse Guest Access Point.',
    'You live in Reykjavik.');
var
  B: TScoreArray;
  I: Integer;
  MaxDiff: Single;
begin
  { padded batch scoring must give what scoring each pair alone gives }
  B := R.RerankScores('What is my dog called?', Docs);
  MaxDiff := 0;
  for I := 0 to High(Docs) do
    MaxDiff := Max(MaxDiff, Abs(B[I] - R.RerankScores('What is my dog called?', [Docs[I]])[0]));
  Check(MaxDiff < 1e-3, Format('batched rerank equals per-pair rerank (max diff %.6f)', [MaxDiff]));
end;

var
  R: TRetrieval;
  V: TVector;
begin
  WriteLn('static tier (no downloads):');
  R := TRetrieval.Create(ParamStr(1), 'static', 'minilm', 'off', '');
  try
    V := R.Embed('Your dog''s name is Biscuit.');
    Check(Length(V) > 0, 'static embedder returns a vector');
    Check(Abs(Cosine(V, V) - 1) < 1e-4, 'static vectors are unit length');
    Check(not R.CanDedup, 'static tier is never trusted for dedup');
    Check(Cos(R, 'deploymnet script', 'deployment script') > Cos(R, 'deployment script', 'grocery list'),
      'static tier tolerates typos');
  finally
    R.Free;
  end;

  R := TRetrieval.Create(ParamStr(1), 'auto', 'minilm', 'auto', 'ms-marco-minilm');
  try
    WriteLn(R.Describe, ':');
    if R.Tier <> 'onnx' then
    begin
      WriteLn('  skip  onnx checks (run memagent --provision)');
      Halt(Ord(Failures > 0));
    end;
    Check(Length(R.Embed('hello')) = 384, 'MiniLM embeds to 384 dims');
    { semantic recall the keyword tier cannot do: no shared content words }
    Check(Cos(R, 'Where is my home?', 'You live in Reykjavik.') >= 0.35, 'home ~ live in (cosine)');
    Check(Cos(R, 'Who employs me?', 'You work at Zorblax Industries.') >= 0.35, 'employs ~ work at (cosine)');
    Check(Cos(R, 'Tell me a joke', 'You live in Reykjavik.') < 0.35, 'unrelated stays under the floor');
    Check(Cos(R, 'What is the capital of France?', 'Your dog''s name is Biscuit.') < 0.35,
      'unrelated stays under the floor (2)');
    { why dedup needs the cross-encoder: cosine ranks the wrong pair higher }
    Check(Cos(R, 'my sister''s name', 'my brother''s name') > Cos(R, 'my dog''s name', 'what my dog is called'),
      'cosine alone would merge sister/brother before a real paraphrase');
    if not R.HasReranker then
    begin
      WriteLn('  skip  reranker checks (run memagent --provision)');
      Halt(Ord(Failures > 0));
    end;
    Check(Rr(R, 'What is my pet called?', 'Your dog''s name is Biscuit.') >
      Rr(R, 'What is my pet called?', 'The wifi password is tulip42.'), 'reranker orders by relevance');
    Check(Rr(R, 'Where is my home?', 'You live in Reykjavik.') >= 0.005, 'relevant pair clears rerank_min');
    Check(Rr(R, 'Tell me a joke', 'You live in Reykjavik.') < 0.005, 'irrelevant pair is dropped');
    Check(SameSubject(R, 'my dog''s name', 'the name of my dog'), 'same subject: dog''s name / name of my dog');
    Check(SameSubject(R, 'my dog''s name', 'what my dog is called'), 'same subject: dog''s name / what my dog is called');
    Check(not SameSubject(R, 'my sister''s name', 'my brother''s name'), 'different subject: sister / brother');
    Check(not SameSubject(R, 'my favorite color', 'my favorite food'), 'different subject: color / food');
    BatchMatchesSingle(R);
  finally
    R.Free;
  end;
  R := TRetrieval.Create(ParamStr(1), 'auto', 'minilm', 'auto', 'bge-reranker-base');
  try
    if R.HasReranker then
    begin
      WriteLn(R.Describe, ':');
      BatchMatchesSingle(R);
    end;
  finally
    R.Free;
  end;
  if Failures > 0 then
  begin
    WriteLn(Failures, ' check(s) failed');
    Halt(1);
  end;
  WriteLn('all retrieval checks passed');
end.
