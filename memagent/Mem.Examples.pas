unit Mem.Examples;

(* The training examples a fact becomes. Two kinds ([train] examples):

   - templates: question paraphrases ("What is my dog's name?", "Remind me,
     what is my dog's name?") all answered with the fact text, word for word.
     Every fact shares the same templates, so an adapter can learn "a
     my-X question gets one of my sentences" instead of the facts, and it
     answers questions about things never stated with a stored value.
   - varied: the same fact from several sides, as in Hatakeyama-Sato et al.
     2023 (arXiv 2312.03360) and O'Neill et al. 2026 (arXiv 2607.11020),
     where varied restatements are what make a fact usable and keep earlier
     facts intact as later ones are written:
       short and full answers to the question paraphrases
       the value asked from the other side ("What is Biscuit to me?")
       yes/no checks, including another fact's value ("Is Leo my dog's
         name?" -> "No. ..."), which ties the value to its own subject
       fill in the blank, and two facts asked together

   Built from the fact's text, subject and value with rules, so it works for
   a 135M model that cannot write its own paraphrases. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, StrUtils, Math;

type
  TFact = record
    Id: Int64;
    Text, Subject, Value: string;
    { the last one is held out as the recall probe }
    Questions: array of string;
  end;
  TFacts = array of TFact;

  TQA = record
    Q, A: string;
  end;
  TQAs = array of TQA;

  TRandFunc = function(AN: Integer): Integer of object;

{ the fact text as the answer to the first AN questions (not the probe) }
function TemplateExamples(const AFact: TFact; AN: Integer): TQAs;
{ AN examples for AFacts[AIndex], cycling through the kinds above }
function VariedExamples(const AFacts: TFacts; AIndex, AN: Integer; ARand: TRandFunc): TQAs;

implementation

function QA(const AQ, AA: string): TQA;
begin
  Result.Q := AQ;
  Result.A := AA;
end;

function TemplateExamples(const AFact: TFact; AN: Integer): TQAs;
var
  I: Integer;
begin
  Result := nil;
  for I := 0 to Min(AN, Length(AFact.Questions) - 1) - 1 do
  begin
    SetLength(Result, Length(Result) + 1);
    Result[High(Result)] := QA(AFact.Questions[I], AFact.Text);
  end;
end;

function Sentence(const S: string): string;
begin
  Result := Trim(S);
  if (Result <> '') and not (Result[Length(Result)] in ['.', '!', '?']) then
    Result := Result + '.';
end;

function VariedExamples(const AFacts: TFacts; AIndex, AN: Integer; ARand: TRandFunc): TQAs;
const
  KINDS = 8;
var
  F: TFact;
  LNQ, LKind, LRound, I, P: Integer;
  LOther: TFact;
  LHasOther: Boolean;
  LPrefix, LRest, LQ: string;

  function Question(AOffset: Integer): string;
  begin
    { the paraphrases, without the held-out probe }
    Result := F.Questions[(LRound * 3 + AOffset) mod LNQ];
  end;

  function Other: TFact;
  var
    J: Integer;
  begin
    { another fact with a different value }
    J := ARand(Length(AFacts) - 1);
    if J >= AIndex then
      Inc(J);
    Result := AFacts[J];
  end;

begin
  Result := nil;
  F := AFacts[AIndex];
  LNQ := Max(1, Length(F.Questions) - 1);
  P := Pos(LowerCase(F.Value), LowerCase(F.Text));
  for I := 0 to AN - 1 do
  begin
    LKind := I mod KINDS;
    LRound := I div KINDS;
    LHasOther := Length(AFacts) > 1;
    if LHasOther then
    begin
      LOther := Other;
      LHasOther := not SameText(LOther.Value, F.Value);
    end;
    case LKind of
      0: LQ := '';  { the paraphrase with the full fact }
      1: LQ := '';  { the paraphrase with the value alone }
      2: LQ := IfThen(LRound mod 2 = 0, 'What is ' + F.Value + ' to me?',
        'What do you know about ' + F.Value + '?');
      3: LQ := 'Is ' + F.Value + ' ' + F.Subject + '?';
      4: if LHasOther then
          LQ := 'Is ' + LOther.Value + ' ' + F.Subject + '?'
        else
          LQ := '';
      5: if P > 0 then
        begin
          LPrefix := TrimRight(Copy(F.Text, 1, P - 1));
          LRest := Copy(F.Text, P + Length(F.Value), MaxInt);
          LQ := 'Fill in the blank: ' + LPrefix + ' ___' + LRest;
        end
        else
          LQ := '';
      6: if LHasOther then
          LQ := 'Tell me ' + F.Subject + ' and ' + LOther.Subject + '.'
        else
          LQ := '';
      7: LQ := '';  { a paraphrase answered "It's <value>." }
    end;
    SetLength(Result, Length(Result) + 1);
    case LKind of
      1: Result[High(Result)] := QA(Question(1), Sentence(F.Value));
      2: Result[High(Result)] := QA(LQ, F.Text);
      3: Result[High(Result)] := QA(LQ, 'Yes. ' + F.Text);
      4: if LQ <> '' then
          Result[High(Result)] := QA(LQ, 'No. ' + F.Text)
        else
          Result[High(Result)] := QA(Question(2), F.Text);
      5: if LQ <> '' then
          Result[High(Result)] := QA(LQ, Sentence(F.Value))
        else
          Result[High(Result)] := QA(Question(2), Sentence(F.Value));
      6: if LQ <> '' then
          Result[High(Result)] := QA(LQ, F.Text + ' ' + LOther.Text)
        else
          Result[High(Result)] := QA(Question(1), F.Text);
      7: Result[High(Result)] := QA(Question(2), 'It''s ' + Sentence(F.Value));
    else
      Result[High(Result)] := QA(Question(0), F.Text);
    end;
  end;
end;

end.
