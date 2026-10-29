program https_get;

(* Test driver for Mem.Https: https_get <url> <dest>. Exit 0 on success,
   1 with the error on stderr otherwise. *)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cwstring,{$ENDIF}
  Mem.Utf8, SysUtils, Mem.Https;

begin
  try
    HttpsGetToFile(ParamStr(1), ParamStr(2), False);
  except
    on E: Exception do
    begin
      WriteLn(StdErr, E.ClassName, ': ', E.Message);
      Halt(1);
    end;
  end;
end.
