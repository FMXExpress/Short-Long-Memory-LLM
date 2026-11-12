unit Mem.Gguf;

(* The shape of a model, read from its GGUF header without loading it: the vocabulary,
   the layers, their width and the feed-forward width. The layout planner needs it
   before anything is loaded: what training holds per token of a window is set by
   these, not by the file's size (TrainBytesPerToken).

   GGUF v2/v3: "GGUF", version, tensor count, key-value count, then the key-values
   (key, type, value; arrays carry their element type and count). Only the header is
   read; the tensor data is never touched. *)

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math, bufstream;

type
  TGgufShape = record
    Arch: string;
    Vocab: Int64;      { entries in tokenizer.ggml.tokens }
    Layers: Int64;     { <arch>.block_count }
    Embd: Int64;       { <arch>.embedding_length }
    FF: Int64;         { <arch>.feed_forward_length (the widest layer when it varies) }
  end;

{ False (and AShape zeroed) when APath is not a readable GGUF file }
function ReadGgufShape(const APath: string; out AShape: TGgufShape): Boolean;

(* Bytes the llama trainer holds per token of a training window, beyond the weights:
   ggml's backward pass keeps every forward activation and its gradient (no
   checkpointing), and the logits over the whole vocabulary with their softmax and
   gradient. Fitted to two models measured at windows of 512 and 256 tokens through
   their own weights (bench/README.md, "Training on a small machine"):
   33.5 x vocab + 2.76 x layers x (10 x width + 3 x ff) bytes, which gives Qwen3-0.6B
   6.29 MiB a token and SmolLM2-135M 2.39 MiB, as measured. Not yet checked on a
   third model. 0 when the shape is unknown. *)
function TrainBytesPerToken(const AShape: TGgufShape): Int64;

implementation

const
  T_UINT8 = 0; T_INT8 = 1; T_UINT16 = 2; T_INT16 = 3; T_UINT32 = 4; T_INT32 = 5;
  T_FLOAT32 = 6; T_BOOL = 7; T_STRING = 8; T_ARRAY = 9; T_UINT64 = 10; T_INT64 = 11; T_FLOAT64 = 12;

function ReadGgufShape(const APath: string; out AShape: TGgufShape): Boolean;
var
  F: TFileStream;
  S: TReadBufStream;
  LMagic: array[0..3] of AnsiChar;
  LVersion, LType, LElem: UInt32;
  LTensors, LKvs, I, J, LCount: UInt64;
  LKey: string;
  LInts: array of Int64;

  function U32: UInt32;
  begin
    Result := 0;
    S.ReadBuffer(Result, 4);
  end;

  function U64: UInt64;
  begin
    Result := 0;
    S.ReadBuffer(Result, 8);
  end;

  function Str: string;
  var
    N: UInt64;
  begin
    N := U64;
    if N > 1 shl 20 then
      raise EReadError.Create('GGUF string too long');
    SetLength(Result, N);
    if N > 0 then
      S.ReadBuffer(Result[1], N);
  end;

  function Size(AType: UInt32): Integer;
  begin
    case AType of
      T_UINT8, T_INT8, T_BOOL: Result := 1;
      T_UINT16, T_INT16: Result := 2;
      T_UINT32, T_INT32, T_FLOAT32: Result := 4;
      T_UINT64, T_INT64, T_FLOAT64: Result := 8;
    else
      raise EReadError.CreateFmt('GGUF value type %d', [AType]);
    end;
  end;

  { an integer of AType, or 0 for a float or bool }
  function Int(AType: UInt32): Int64;
  var
    B: array[0..7] of Byte;
  begin
    FillChar(B, SizeOf(B), 0);
    S.ReadBuffer(B, Size(AType));
    case AType of
      T_UINT8: Result := PByte(@B)^;
      T_INT8: Result := PShortInt(@B)^;
      T_UINT16: Result := PWord(@B)^;
      T_INT16: Result := PSmallInt(@B)^;
      T_UINT32: Result := PLongWord(@B)^;
      T_INT32: Result := PLongInt(@B)^;
      T_UINT64, T_INT64: Result := PInt64(@B)^;
    else
      Result := 0;
    end;
  end;

  procedure Skip(AType: UInt32);
  begin
    if AType = T_STRING then
      Str
    else
      S.Seek(Size(AType), soCurrent);
  end;

  function Wanted(const AKey: string): Boolean;
  begin
    Result := (AKey = 'general.architecture') or (AKey = AShape.Arch + '.block_count') or
      (AKey = AShape.Arch + '.embedding_length') or (AKey = AShape.Arch + '.feed_forward_length');
  end;

  procedure Take(const AKey: string; AValue: Int64);
  begin
    if AKey = AShape.Arch + '.block_count' then
      AShape.Layers := Max(AShape.Layers, AValue)
    else if AKey = AShape.Arch + '.embedding_length' then
      AShape.Embd := Max(AShape.Embd, AValue)
    else if AKey = AShape.Arch + '.feed_forward_length' then
      AShape.FF := Max(AShape.FF, AValue);
  end;

begin
  AShape := Default(TGgufShape);
  Result := False;
  if not FileExists(APath) then
    Exit;
  try
    F := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
    try
      S := TReadBufStream.Create(F, 1 shl 16);
      try
        S.ReadBuffer(LMagic, 4);
        if LMagic <> 'GGUF' then
          Exit;
        LVersion := U32;
        if (LVersion < 2) or (LVersion > 3) then
          Exit;
        LTensors := U64;
        LKvs := U64;
        if (LTensors > 1 shl 24) or (LKvs > 1 shl 16) then
          Exit;
        LInts := nil;
        for I := 1 to LKvs do
        begin
          LKey := Str;
          LType := U32;
          if LType = T_STRING then
          begin
            if LKey = 'general.architecture' then
              AShape.Arch := Str
            else
              Str;
          end
          else if LType = T_ARRAY then
          begin
            LElem := U32;
            LCount := U64;
            if LKey = 'tokenizer.ggml.tokens' then
              AShape.Vocab := LCount;
            for J := 1 to LCount do
              if (LElem <> T_STRING) and (LElem <> T_ARRAY) and Wanted(LKey) then
                Take(LKey, Int(LElem))
              else if LElem = T_ARRAY then
                raise EReadError.Create('GGUF nested array')
              else
                Skip(LElem);
          end
          else if Wanted(LKey) then
            Take(LKey, Int(LType))
          else
            Skip(LType);
        end;
        Result := (AShape.Vocab > 0) and (AShape.Layers > 0) and (AShape.Embd > 0);
      finally
        S.Free;
      end;
    finally
      F.Free;
    end;
  except
    on Exception do
    begin
      AShape := Default(TGgufShape);
      Result := False;
    end;
  end;
end;

function TrainBytesPerToken(const AShape: TGgufShape): Int64;
var
  LFF: Int64;
begin
  if (AShape.Vocab <= 0) or (AShape.Layers <= 0) or (AShape.Embd <= 0) then
    Exit(0);
  LFF := AShape.FF;
  if LFF <= 0 then
    LFF := 4 * AShape.Embd;
  Result := Round(33.5 * AShape.Vocab + 2.76 * AShape.Layers * (10 * AShape.Embd + 3 * LFF));
end;

end.
