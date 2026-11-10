unit Mem.Layout;

(* Where chat and training run: [gpu] layout, resolved once at startup.

     layout              chat on   training on  training runs
     cpu                 CPU       CPU          between turns
     chat-gpu-train-cpu  GPU       CPU          between turns (small models: the card answers fast,
                                                the CPU trains in the background)
     chat-cpu-train-gpu  CPU       GPU          between turns
     two-gpus            GPU gpu   GPU second   between turns, never sharing a card
     one-gpu             GPU       same GPU     between turns, waiting while a turn is answered
     one-gpu-idle        GPU       same GPU     only when idle (idle_min, night): the chat model is
                                                unloaded so training has the whole card
     auto                no GPU -> cpu, one -> one-gpu, two or more -> two-gpus
     custom              [gpu] infer and train as written (files from before layouts)

   [consolidate] train_base = auto picks what the llama trainer runs the frozen base
   on: an f32 copy (the fastest to train, 4x a 4-bit model's size) when it fits on
   the training device, else the GGUF as loaded (on the chat engine's card, the
   engine's own copy). On the CPU it is always the f32 copy: there the backward pass
   through f16 weights was 5.8x slower (bench/README.md).

   PlanLayout is the decision, a pure function of the settings, the GPUs found and
   the model's size, so it is tested without a GPU (tests/layout_test.lpr). *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Mem.Config;

type
  TGpuInfo = record
    Name: string;
    FreeBytes, TotalBytes: Int64;
  end;
  TGpuInfos = array of TGpuInfo;

  TLayoutPlan = record
    Layout: string;          { the layout in effect (auto resolved) }
    InferGpu, TrainGpu: Integer;
    TrainBase: string;       { f32 | model }
    TrainWhen: string;       { always | idle }
    Why: string;             { how auto and train_base decided, for the startup line }
  end;

  ELayout = class(Exception);

const
  LAYOUTS: array[0..7] of string = ('auto', 'cpu', 'chat-gpu-train-cpu', 'chat-cpu-train-gpu', 'two-gpus',
    'one-gpu', 'one-gpu-idle', 'custom');

{ bytes an f32 copy of AGguf needs: the copy itself if it exists, else from the
  quantization its name says (ANameHint: the download's file name) }
function F32Bytes(const AGguf, ANameHint: string; AModelBytes: Int64): Int64;
{ room for training beyond the weights: activations, K/V, the adapter and AdamW }
function TrainMargin(AModelBytes: Int64): Int64;
{ the decision; AKnown = False when the GPUs could not be listed (then indices are
  taken as written and nothing is checked against the cards) }
function PlanLayout(ACfg: TMemConfig; const AGpus: TGpuInfos; AKnown: Boolean; AModelBytes, AF32Bytes: Int64): TLayoutPlan;
{ lists the GPUs, plans, and sets ACfg.InferGpu, TrainGpu, TrainBase and TrainWhen;
  returns the startup line. Raises ELayout for a layout the machine cannot run. }
function ResolveLayout(ACfg: TMemConfig): string;
{ hh:mm-hh:mm, possibly across midnight; '' is always }
function InNightWindow(const AWindow: string; ANow: TDateTime): Boolean;
function GiB(ABytes: Int64): string;

implementation

uses
  StrUtils, DateUtils, Mem.Llama, Mem.LlamaChat;

function GiB(ABytes: Int64): string;
begin
  Result := FormatFloat('0.0', ABytes / (1024.0 * 1024 * 1024)) + ' GB';
end;

function BitsPerWeight(const AName: string): Double;
var
  N: string;
begin
  { the usual llama.cpp file types; mixed quants are averages over the whole file }
  N := LowerCase(AName);
  if Pos('f32', N) > 0 then Exit(32);
  if (Pos('bf16', N) > 0) or (Pos('f16', N) > 0) then Exit(16);
  if Pos('q8_0', N) > 0 then Exit(8.5);
  if Pos('q6_k', N) > 0 then Exit(6.6);
  if Pos('q5_k', N) > 0 then Exit(5.7);
  if (Pos('q5_0', N) > 0) or (Pos('q5_1', N) > 0) then Exit(5.5);
  if Pos('q4_k', N) > 0 then Exit(4.85);
  if (Pos('q4_0', N) > 0) or (Pos('q4_1', N) > 0) or (Pos('iq4', N) > 0) or (Pos('mxfp4', N) > 0) then Exit(4.5);
  if Pos('q3_k', N) > 0 then Exit(3.9);
  if Pos('iq3', N) > 0 then Exit(3.5);
  if (Pos('q2_k', N) > 0) or (Pos('iq2', N) > 0) then Exit(3.0);
  Result := 16;
end;

function FileBytes(const APath: string): Int64;
var
  R: TSearchRec;
begin
  Result := 0;
  if FindFirst(APath, faAnyFile, R) = 0 then
  begin
    Result := R.Size;
    FindClose(R);
  end;
end;

function F32Bytes(const AGguf, ANameHint: string; AModelBytes: Int64): Int64;
var
  LCopy: string;
begin
  LCopy := ChangeFileExt(AGguf, '') + '-f32.gguf';
  Result := FileBytes(LCopy);
  if Result > 0 then
    Exit;
  if AModelBytes <= 0 then
    Exit(0);
  Result := Round(AModelBytes * 32.0 / BitsPerWeight(ANameHint + ' ' + ExtractFileName(AGguf)));
end;

function TrainMargin(AModelBytes: Int64): Int64;
begin
  Result := Int64(1024) * 1024 * 1024 + AModelBytes div 8;
end;

function PlanLayout(ACfg: TMemConfig; const AGpus: TGpuInfos; AKnown: Boolean; AModelBytes, AF32Bytes: Int64): TLayoutPlan;
var
  L, LWhy: string;
  N, G, S: Integer;
  LAvail: Int64;
  LShared: Boolean;

  procedure Need(AIndex: Integer);
  begin
    if AKnown and ((AIndex < 0) or (AIndex >= N)) then
      raise ELayout.CreateFmt('[gpu] layout = %s needs GPU %d, and llama.cpp sees %d GPU%s%s', [L, AIndex, N,
        IfThen(N = 1, '', 's'), IfThen(N = 0, ' (a CPU-only build, or no card: use layout = cpu)', '')]);
  end;

  function FreeOn(AIndex: Integer): Int64;
  begin
    if AKnown and (AIndex >= 0) and (AIndex < N) then
      Result := AGpus[AIndex].FreeBytes
    else
      Result := -1;
  end;

begin
  Result := Default(TLayoutPlan);
  L := ACfg.Layout;
  if AnsiIndexText(L, LAYOUTS) < 0 then
    raise ELayout.CreateFmt('[gpu] layout = %s: use one of auto, cpu, chat-gpu-train-cpu, chat-cpu-train-gpu, ' +
      'two-gpus, one-gpu, one-gpu-idle, custom', [L]);
  N := Length(AGpus);
  G := ACfg.LayoutGpu;
  S := ACfg.LayoutSecondGpu;
  LWhy := '';
  if L = 'auto' then
  begin
    if not AKnown then
    begin
      L := 'custom';
      LWhy := 'auto: the GPUs could not be listed, so [gpu] infer and train as written';
    end
    else
    begin
      if N = 0 then
        L := 'cpu'
      else if N = 1 then
        L := 'one-gpu'
      else
        L := 'two-gpus';
      LWhy := Format('auto: %d GPU%s', [N, IfThen(N = 1, '', 's')]);
      if (L = 'two-gpus') and (S = G) then
        S := IfThen(G = 0, 1, 0);
    end;
  end;
  Result.Layout := L;
  if L = 'cpu' then
  begin
    Result.InferGpu := -1;
    Result.TrainGpu := -1;
  end
  else if L = 'chat-gpu-train-cpu' then
  begin
    Need(G);
    Result.InferGpu := G;
    Result.TrainGpu := -1;
  end
  else if L = 'chat-cpu-train-gpu' then
  begin
    Need(G);
    Result.InferGpu := -1;
    Result.TrainGpu := G;
  end
  else if L = 'two-gpus' then
  begin
    if S = G then
      raise ELayout.CreateFmt('[gpu] layout = two-gpus: gpu and second are both %d; give second another card', [G]);
    Need(G);
    Need(S);
    Result.InferGpu := G;
    Result.TrainGpu := S;
  end
  else if (L = 'one-gpu') or (L = 'one-gpu-idle') then
  begin
    Need(G);
    Result.InferGpu := G;
    Result.TrainGpu := G;
  end
  else
  begin
    Result.InferGpu := ACfg.InferGpu;
    Result.TrainGpu := ACfg.TrainGpu;
  end;

  { when training may run }
  if (ACfg.TrainWhen = 'idle') or ((ACfg.TrainWhen = 'auto') and (L = 'one-gpu-idle')) then
    Result.TrainWhen := 'idle'
  else
    Result.TrainWhen := 'always';

  { what the trainer runs the frozen base on }
  if ACfg.TrainBase <> 'auto' then
    Result.TrainBase := ACfg.TrainBase
  else if Result.TrainGpu < 0 then
    Result.TrainBase := 'f32'
  else if (AModelBytes <= 0) or (AF32Bytes <= 0) then
  begin
    Result.TrainBase := 'f32';
    LWhy := LWhy + IfThen(LWhy = '', '', '; ') + 'model size unknown, so the f32 copy';
  end
  else
  begin
    LAvail := FreeOn(Result.TrainGpu);
    (* the chat model shares the card while training runs, unless it is unloaded
       for training (idle) or lives elsewhere *)
    LShared := (Result.TrainGpu = Result.InferGpu) and (Result.TrainWhen <> 'idle');
    if (LAvail >= 0) and LShared then
      LAvail := LAvail - AModelBytes;
    if LAvail < 0 then
    begin
      Result.TrainBase := IfThen(LShared, 'model', 'f32');
      LWhy := LWhy + IfThen(LWhy = '', '', '; ') + 'free memory unknown';
    end
    else if AF32Bytes + TrainMargin(AModelBytes) <= LAvail then
    begin
      Result.TrainBase := 'f32';
      LWhy := LWhy + IfThen(LWhy = '', '', '; ') + Format('the f32 copy (%s) fits', [GiB(AF32Bytes)]);
    end
    else
    begin
      Result.TrainBase := 'model';
      LWhy := LWhy + IfThen(LWhy = '', '', '; ') + Format('an f32 copy needs %s, %s free, so the model as loaded',
        [GiB(AF32Bytes + TrainMargin(AModelBytes)), GiB(LAvail)]);
    end;
  end;
  Result.Why := LWhy;
end;

function InNightWindow(const AWindow: string; ANow: TDateTime): Boolean;
var
  P: Integer;
  A, B, T: Double;

  function Clock(const S: string): Double;
  var
    Q: Integer;
  begin
    Q := Pos(':', S);
    if Q = 0 then
      Result := StrToIntDef(Trim(S), 0) * 60
    else
      Result := StrToIntDef(Trim(Copy(S, 1, Q - 1)), 0) * 60 + StrToIntDef(Trim(Copy(S, Q + 1, 2)), 0);
  end;

begin
  if Trim(AWindow) = '' then
    Exit(True);
  P := Pos('-', AWindow);
  if P = 0 then
    Exit(True);
  A := Clock(Copy(AWindow, 1, P - 1));
  B := Clock(Copy(AWindow, P + 1, MaxInt));
  T := HourOf(ANow) * 60 + MinuteOf(ANow);
  if A <= B then
    Result := (T >= A) and (T < B)
  else
    Result := (T >= A) or (T < B);   { across midnight: 22:00-06:00 }
end;

function ResolveLayout(ACfg: TMemConfig): string;
var
  LGpus: TLlamaGpus;
  LInfo: TGpuInfos;
  LKnown: Boolean;
  LModel, LF32: Int64;
  LPlan: TLayoutPlan;
  I: Integer;
  LCards, LWhere: string;
begin
  LKnown := False;
  LInfo := nil;
  try
    ForceDirectories(ACfg.LogDir);   { llama.log goes there }
    LlamaSetup(ACfg);
    LKnown := LlamaGpus(LGpus);
    SetLength(LInfo, Length(LGpus));
    for I := 0 to High(LGpus) do
    begin
      LInfo[I].Name := LGpus[I].Name;
      LInfo[I].FreeBytes := LGpus[I].FreeBytes;
      LInfo[I].TotalBytes := LGpus[I].TotalBytes;
    end;
  except
    on Exception do
      LKnown := False;   { no llama.cpp here (the python engine): nothing to list }
  end;
  LModel := FileBytes(ACfg.GgufModel);
  LF32 := F32Bytes(ACfg.GgufModel, ACfg.GgufUrl, LModel);
  LPlan := PlanLayout(ACfg, LInfo, LKnown, LModel, LF32);
  ACfg.InferGpu := LPlan.InferGpu;
  ACfg.TrainGpu := LPlan.TrainGpu;
  ACfg.TrainBase := LPlan.TrainBase;
  ACfg.TrainWhen := LPlan.TrainWhen;
  LCards := '';
  for I := 0 to High(LInfo) do
    LCards := LCards + Format('%sGPU %d %s (%s of %s free)', [IfThen(I = 0, '', ', '), I, LInfo[I].Name,
      GiB(LInfo[I].FreeBytes), GiB(LInfo[I].TotalBytes)]);
  if ACfg.ProfileTrainer = 'none' then
    LWhere := 'no training (a chat-only profile)'
  else if LPlan.TrainWhen = 'idle' then
    LWhere := Format('training on %s when idle (%s min without a message%s; the chat model is unloaded for it)',
      [DeviceName(LPlan.TrainGpu), FormatFloat('0.##', ACfg.IdleMin),
       IfThen(ACfg.NightWindow = '', '', ', within ' + ACfg.NightWindow)])
  else if LPlan.TrainGpu = LPlan.InferGpu then
    LWhere := 'training on the same ' + DeviceName(LPlan.TrainGpu) + ' between turns'
  else
    LWhere := 'training on ' + DeviceName(LPlan.TrainGpu);
  if ACfg.ProfileTrainer = 'none' then
    Result := Format('layout %s: chat on %s, %s', [LPlan.Layout, DeviceName(LPlan.InferGpu), LWhere])
  else
    Result := Format('layout %s: chat on %s, %s, from %s', [LPlan.Layout, DeviceName(LPlan.InferGpu), LWhere,
      IfThen(LPlan.TrainBase = 'model', 'the model as loaded', 'an f32 copy')]);
  if LPlan.Why <> '' then
    Result := Result + ' (' + LPlan.Why + ')';
  if LCards <> '' then
    Result := Result + LineEnding + 'gpus: ' + LCards;
  (* the CUDA backward pass through f16 or quantized weights is not there yet
     (bench/README.md, "One GPU for chat and training") *)
  if (ACfg.ProfileTrainer <> 'none') and (LPlan.TrainBase = 'model') and (LPlan.TrainGpu >= 0) and (BitsPerWeight(ACfg.GgufUrl + ' ' +
    ExtractFileName(ACfg.GgufModel)) < 32) then
    Result := Result + LineEnding + 'note: llama.cpp''s CUDA backend has no out_prod for f16 or quantized ' +
      'weights yet, so part of each training step runs on the CPU: expect slow training';
end;

end.
