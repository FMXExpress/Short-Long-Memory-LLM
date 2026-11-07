program jev_guard;

(* The answer guard as a decision-model readout, next to the current guard (Mem.Guard).

     jev_guard <cache_dir> <cases.json> <facts.json> <model.gguf> <out.json> [threads] [T] [noul_t] [think]

   For every case (bench/jev/make_cases.py: a question, an answer a model gave, the
   facts two-hop retrieval returns, and whether the answer was right) it records

     guard    Licensed(): the reranker + binding rules memagent runs today, against
              every stored fact (facts.json), as the agent does
     p_yes    a Jev-style decision over the retrieved facts: one forward pass, the
              logits of the option letters at the first output position, softmax
              over the letters at temperature T, then sigmoid(logit(p)/noul_t) as
              OpenJev's helper does (JevBan, Ban.Local). The question and answer go
              in the state as quoted data, never into the question.

   think=1 adds the empty <think></think> block OpenJev's chat template expects; 0 for
   a model without one (Qwen2.5). *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Classes, Math, DateUtils, fpjson, jsonparser, Mem.Store, Mem.Retrieval, Mem.Guard,
  Mem.Llama;

const
  CQuestion = 'The assistant answered the user''s question about the user''s own people. Decide whether ' +
    'the answer is right according to the stored facts in the state. It is right only if every name and ' +
    'value it gives about the person the question asks about is backed by those facts for that same ' +
    'person (a relation such as "my sister" is resolved through the fact that names her), and it answers ' +
    'what was asked; for a yes/no question it must say yes or no correctly.';
  COptYes = 'yes: the facts back the answer and it answers the question';
  COptNo = 'no: the answer gives a name or value the facts do not back for that person, gives another ' +
    'person''s value, or does not answer what was asked';

var
  Model: PLlamaModel;
  Ctx: PLlamaContext;
  Vocab: PLlamaVocab;
  LetterA, LetterB: TLlamaToken;
  NCtx: Integer = 4096;

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

{ json.dumps(s, ensure_ascii=False) }
function PyStr(const S: string): string;
begin
  Result := '"' + StringToJSONString(S, False) + '"';
end;

function Tokenize(const AText: string; ASpecial: Boolean): TLlamaTokens;
var
  N: Integer;
begin
  Result := nil;
  SetLength(Result, Length(AText) + 16);
  N := llama_tokenize(Vocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, ASpecial);
  if N < 0 then
  begin
    SetLength(Result, -N);
    N := llama_tokenize(Vocab, PAnsiChar(AText), Length(AText), @Result[0], Length(Result), False, ASpecial);
  end;
  if N < 0 then
    raise Exception.Create('tokenize failed');
  SetLength(Result, N);
end;

{ P(option A) from the letter logits at the first output position }
function ReadoutA(const APrompt: string; ATemp: Double; out ATokens: Integer): Double;
var
  T: TLlamaTokens;
  LPos, N: Integer;
  L: PSingle;
  A, B, M: Double;
begin
  T := Tokenize(APrompt, True);
  ATokens := Length(T);
  if Length(T) >= NCtx then
    raise Exception.CreateFmt('prompt of %d tokens', [Length(T)]);
  llama_memory_clear(llama_get_memory(Ctx), True);
  LPos := 0;
  while LPos < Length(T) do
  begin
    N := Min(512, Length(T) - LPos);
    if llama_decode(Ctx, llama_batch_get_one(@T[LPos], N)) <> 0 then
      raise Exception.Create('llama_decode failed');
    Inc(LPos, N);
  end;
  L := llama_get_logits_ith(Ctx, -1);
  A := L[LetterA] / ATemp;
  B := L[LetterB] / ATemp;
  M := Max(A, B);
  Result := Exp(A - M) / (Exp(A - M) + Exp(B - M));
end;

var
  Ret: TRetrieval;
  Cases, FactsJ: TJSONArray;
  Facts: TFactArray;
  C: TJSONObject;
  MP: TLlamaModelParams;
  CP: TLlamaContextParams;
  Out_: TStringList;
  I, J, Threads, NTok: Integer;
  T, NoulT, P, Z, Noul: Double;
  Think: Boolean;
  State, Prompt, Why, Line: string;
  Allow: Boolean;
  FList: TJSONArray;
  Start: TDateTime;
  Lt: TLlamaTokens;
begin
  if ParamCount < 5 then
  begin
    WriteLn('jev_guard <cache_dir> <cases.json> <facts.json> <model.gguf> <out.json> [threads] [T] [noul_t] [think]');
    Halt(2);
  end;
  Threads := StrToIntDef(ParamStr(6), 0);
  if Threads <= 0 then
    Threads := 4;
  T := StrToFloatDef(ParamStr(7), 1.07);
  NoulT := StrToFloatDef(ParamStr(8), 1.074766);
  Think := ParamStr(9) <> '0';

  Ret := TRetrieval.Create(ParamStr(1), 'off', '', 'auto', 'bge-reranker-base');
  Cases := LoadJson(ParamStr(2)) as TJSONArray;
  FactsJ := LoadJson(ParamStr(3)) as TJSONArray;
  SetLength(Facts, FactsJ.Count);
  for I := 0 to FactsJ.Count - 1 do
  begin
    Facts[I] := Default(TFactRec);
    Facts[I].Id := I + 1;
    Facts[I].Text := TJSONObject(FactsJ[I]).Get('text', '');
    Facts[I].Subject := TJSONObject(FactsJ[I]).Get('subject', '');
    Facts[I].Value := TJSONObject(FactsJ[I]).Get('value', '');
    Facts[I].Confidence := 1;
  end;

  LlamaLoad(IncludeTrailingPathDelimiter(ParamStr(1)) + 'llama');
  llama_backend_init;
  MP := llama_model_default_params();
  MP.n_gpu_layers := 0;
  Model := llama_model_load_from_file(PAnsiChar(AnsiString(ParamStr(4))), MP);
  if Model = nil then
    raise Exception.Create('could not load ' + ParamStr(4));
  Vocab := llama_model_get_vocab(Model);
  CP := llama_context_default_params();
  CP.n_ctx := NCtx;
  CP.n_batch := 512;
  CP.n_ubatch := 512;
  CP.n_threads := Threads;
  CP.n_threads_batch := Threads;
  Ctx := llama_init_from_model(Model, CP);
  if Ctx = nil then
    raise Exception.Create('no context');
  Lt := Tokenize('A', False);
  LetterA := Lt[0];
  Lt := Tokenize('B', False);
  LetterB := Lt[0];

  Out_ := TStringList.Create;
  for I := 0 to Cases.Count - 1 do
  begin
    C := TJSONObject(Cases[I]);
    Allow := Licensed(Ret, C.Get('q', ''), C.Get('answer', ''), Facts, Why, 0.03);
    FList := C.Arrays['facts'];
    State := '{"stored_facts": [';
    for J := 0 to FList.Count - 1 do
    begin
      if J > 0 then
        State := State + ', ';
      State := State + PyStr(FList.Strings[J]);
    end;
    State := State + '], "user_question": ' + PyStr(C.Get('q', '')) + ', "assistant_answer": ' +
      PyStr(C.Get('answer', '')) + '}';
    Prompt := '<|im_start|>user' + #10 + 'State:' + #10 + State + #10#10 + 'Question: ' + CQuestion + #10 +
      'Options:' + #10 + '[A] ' + COptYes + #10 + '[B] ' + COptNo + #10#10 +
      'Answer with the letter of the best option only.<|im_end|>' + #10 + '<|im_start|>assistant' + #10;
    if Think then
      Prompt := Prompt + '<think>' + #10#10 + '</think>' + #10#10;
    Start := Now;
    P := EnsureRange(ReadoutA(Prompt, T, NTok), 1e-4, 1 - 1e-4);
    Z := Ln(P / (1 - P)) / NoulT;
    Noul := 1 / (1 + Exp(-Z));
    Line := Format('{"id": %d, "right": %s, "type": %s, "model": %s, "arm": %s, "guard_allow": %s, ' +
      '"guard_why": %s, "p_yes": %.4f, "tokens": %d, "ms": %d}',
      [C.Get('id', 0), BoolToStr(C.Get('right', False), 'true', 'false'), PyStr(C.Get('type', '')),
       PyStr(C.Get('model', '')), PyStr(C.Get('arm', '')), BoolToStr(Allow, 'true', 'false'), PyStr(Why),
       Noul, NTok, MilliSecondsBetween(Now, Start)]);
    Out_.Add(Line);
    WriteLn(Format('%d/%d right=%s guard=%s p_yes=%.3f %dms', [I + 1, Cases.Count,
      BoolToStr(C.Get('right', False), 'Y', 'N'), BoolToStr(Allow, 'allow', 'block'), Noul,
      MilliSecondsBetween(Now, Start)]));
    Out_.SaveToFile(ParamStr(5));
  end;
  Out_.Free;
  llama_free(Ctx);
  llama_model_free(Model);
end.
