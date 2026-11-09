program layout_test;

(* Mem.Layout.PlanLayout without a GPU: the layout each machine gets, which card does
   what, and whether training runs on an f32 copy or the model as loaded. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  SysUtils, DateUtils, Mem.Config, Mem.Layout;

const
  GB = Int64(1024) * 1024 * 1024;

var
  Fails, Checks: Integer;

procedure Check(ACond: Boolean; const AWhat: string);
begin
  Inc(Checks);
  if not ACond then
  begin
    Inc(Fails);
    WriteLn('FAIL: ', AWhat);
  end;
end;

function Gpus(const AFreeGB: array of Double): TGpuInfos;
var
  I: Integer;
begin
  SetLength(Result, Length(AFreeGB));
  for I := 0 to High(AFreeGB) do
  begin
    Result[I].Name := 'test card ' + IntToStr(I);
    Result[I].FreeBytes := Round(AFreeGB[I] * GB);
    Result[I].TotalBytes := Round(AFreeGB[I] * GB) + GB;
  end;
end;

function Cfg(const ALayout: string): TMemConfig;
begin
  Result := TMemConfig.Create;
  Result.Layout := ALayout;
  Result.LayoutGpu := 0;
  Result.LayoutSecondGpu := 1;
  Result.InferGpu := 3;
  Result.TrainGpu := 4;
  Result.TrainBase := 'auto';
  Result.TrainWhen := 'auto';
end;

function Plan(const ALayout: string; const AGpus: TGpuInfos; AKnown: Boolean; AModel, AF32: Int64;
  const ABase: string = 'auto'; const AWhen: string = 'auto'): TLayoutPlan;
var
  C: TMemConfig;
begin
  C := Cfg(ALayout);
  try
    C.TrainBase := ABase;
    C.TrainWhen := AWhen;
    Result := PlanLayout(C, AGpus, AKnown, AModel, AF32);
  finally
    C.Free;
  end;
end;

function Raises(const ALayout: string; const AGpus: TGpuInfos): Boolean;
begin
  Result := False;
  try
    Plan(ALayout, AGpus, True, 2 * GB, 6 * GB);
  except
    on ELayout do
      Result := True;
  end;
end;

var
  P: TLayoutPlan;
  Small, SmallF32, Big, BigF32: Int64;
begin
  Fails := 0;
  Checks := 0;
  { the small profile: Qwen2.5-1.5B q8_0, 1.9 GB; its f32 copy about 6.3 GB }
  Small := Round(1.9 * GB);
  SmallF32 := F32Bytes('/nonexistent/small.gguf', 'qwen2.5-1.5b-instruct-q8_0.gguf', Small);
  Check(Abs(SmallF32 - Round(1.9 * GB * 32 / 8.5)) < GB div 100, 'f32 estimate from a q8_0 name');
  { a 27B at Q4_K_M: 17.4 GB, about 115 GB as f32 }
  Big := Round(17.44 * GB);
  BigF32 := F32Bytes('/nonexistent/big.gguf', 'Qwen3.8-27B-Q4_K_M.gguf', Big);
  Check((BigF32 > 110 * GB) and (BigF32 < 120 * GB), 'f32 estimate for a 27B Q4_K_M: ' + GiB(BigF32));

  P := Plan('auto', Gpus([]), True, Small, SmallF32);
  Check((P.Layout = 'cpu') and (P.InferGpu = -1) and (P.TrainGpu = -1) and (P.TrainBase = 'f32') and
    (P.TrainWhen = 'always'), 'auto, no GPU: cpu, f32 copy');

  P := Plan('auto', Gpus([22]), True, Small, SmallF32);
  Check((P.Layout = 'one-gpu') and (P.InferGpu = 0) and (P.TrainGpu = 0) and (P.TrainBase = 'f32'),
    'auto, one 24 GB card, 1.5B: one-gpu with an f32 copy beside the chat model (' + P.Why + ')');

  P := Plan('auto', Gpus([22]), True, Big, BigF32);
  Check((P.Layout = 'one-gpu') and (P.TrainBase = 'model'), 'auto, one 24 GB card, 27B: the model as loaded (' +
    P.Why + ')');

  P := Plan('auto', Gpus([7]), True, Small, SmallF32);
  Check((P.Layout = 'one-gpu') and (P.TrainBase = 'model'),
    'auto, an 8 GB card, 1.5B: chat (1.9) + f32 (6.3) + margin do not fit in 7 free, so the model as loaded');

  P := Plan('one-gpu-idle', Gpus([10]), True, Small, SmallF32);
  Check((P.TrainWhen = 'idle') and (P.TrainBase = 'f32') and (P.InferGpu = 0) and (P.TrainGpu = 0),
    'one-gpu-idle, 12 GB card, 1.5B: the chat model is unloaded, so the f32 copy fits (' + P.Why + ')');
  P := Plan('one-gpu', Gpus([10]), True, Small, SmallF32);
  Check(P.TrainBase = 'model', 'one-gpu, 12 GB card, 1.5B: with the chat model loaded the f32 copy does not fit');

  P := Plan('one-gpu-idle', Gpus([22]), True, Big, BigF32);
  Check((P.TrainWhen = 'idle') and (P.TrainBase = 'model'), 'one-gpu-idle, 27B: the model as loaded');

  P := Plan('auto', Gpus([22, 22]), True, Small, SmallF32);
  Check((P.Layout = 'two-gpus') and (P.InferGpu = 0) and (P.TrainGpu = 1) and (P.TrainBase = 'f32'),
    'auto, two cards: chat on 0, training on 1');

  P := Plan('chat-gpu-train-cpu', Gpus([22]), True, Small, SmallF32);
  Check((P.InferGpu = 0) and (P.TrainGpu = -1) and (P.TrainBase = 'f32'), 'chat-gpu-train-cpu');

  P := Plan('chat-cpu-train-gpu', Gpus([10]), True, Small, SmallF32);
  Check((P.InferGpu = -1) and (P.TrainGpu = 0) and (P.TrainBase = 'f32'),
    'chat-cpu-train-gpu, 12 GB card, 1.5B: the card holds only training, f32 fits');

  P := Plan('chat-cpu-train-gpu', Gpus([22]), True, Big, BigF32);
  Check((P.TrainGpu = 0) and (P.TrainBase = 'model'), 'chat-cpu-train-gpu, 27B: the model as loaded');

  P := Plan('cpu', Gpus([22]), True, Small, SmallF32);
  Check((P.InferGpu = -1) and (P.TrainGpu = -1), 'cpu layout ignores the card');

  P := Plan('custom', Gpus([22]), True, Small, SmallF32);
  Check((P.InferGpu = 3) and (P.TrainGpu = 4), 'custom keeps [gpu] infer and train');

  P := Plan('auto', nil, False, Small, SmallF32);
  Check((P.Layout = 'custom') and (P.InferGpu = 3), 'auto, GPUs not listable: custom');

  P := Plan('one-gpu', Gpus([22]), True, Small, SmallF32, 'model');
  Check(P.TrainBase = 'model', 'train_base = model is kept');
  P := Plan('cpu', Gpus([]), True, Small, SmallF32, 'auto', 'idle');
  Check(P.TrainWhen = 'idle', 'train_when = idle on any layout');
  P := Plan('one-gpu', Gpus([22]), True, 0, 0);
  Check(P.TrainBase = 'f32', 'model size unknown: the f32 copy (as before layouts)');

  Check(Raises('two-gpus', Gpus([22])), 'two-gpus with one card is refused');
  Check(Raises('one-gpu', Gpus([])), 'one-gpu with no card is refused');
  Check(Raises('chat-gpu-train-cpu', Gpus([])), 'chat-gpu-train-cpu with no card is refused');
  Check(Raises('three-gpus', Gpus([22])), 'an unknown layout is refused');
  Check(not Raises('two-gpus', Gpus([22, 22])), 'two-gpus with two cards');

  Check(InNightWindow('', EncodeDateTime(2026, 11, 9, 12, 0, 0, 0)), 'no window: any time');
  Check(InNightWindow('01:00-06:00', EncodeDateTime(2026, 11, 9, 3, 30, 0, 0)), '01:00-06:00 at 03:30');
  Check(not InNightWindow('01:00-06:00', EncodeDateTime(2026, 11, 9, 6, 0, 0, 0)), '01:00-06:00 ends at 06:00');
  Check(InNightWindow('22:00-06:00', EncodeDateTime(2026, 11, 9, 23, 0, 0, 0)), '22:00-06:00 at 23:00');
  Check(InNightWindow('22:00-06:00', EncodeDateTime(2026, 11, 9, 2, 0, 0, 0)), '22:00-06:00 at 02:00');
  Check(not InNightWindow('22:00-06:00', EncodeDateTime(2026, 11, 9, 12, 0, 0, 0)), '22:00-06:00 at 12:00');

  if Fails = 0 then
    WriteLn('layout_test: ', Checks, '/', Checks, ' checks passed')
  else
  begin
    WriteLn('layout_test: ', Fails, ' of ', Checks, ' checks failed');
    Halt(1);
  end;
end.
