unit Mem.Utf8;

(* Makes UTF-8 the codepage of every AnsiString, of the console and of file
   names, whatever the process locale. List it first in each program's uses
   clause (after cthreads/cwstring).

   Why: FPC 3.2.2's string is AnsiString(CP_ACP). With a C/POSIX locale, or
   no widestring manager, every UTF8String -> string conversion (fpjson
   returns UTF8String) is lossy, so "Zoë" from the model came back as the
   Latin-1 byte 0xEB and was stored as invalid UTF-8 in memory.db. *)

{$mode objfpc}{$H+}

interface

implementation

{$IFDEF MSWINDOWS}
uses
  Windows;
{$ENDIF}

initialization
  {$IFDEF MSWINDOWS}
  { the console shows and reads UTF-8 ("Zoë"), not the OEM codepage }
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  {$ENDIF}
  SetMultiByteConversionCodePage(CP_UTF8);
  SetMultiByteFileSystemCodePage(CP_UTF8);
  SetMultiByteRTLFileSystemCodePage(CP_UTF8);
  SetTextCodePage(Input, CP_UTF8);
  SetTextCodePage(Output, CP_UTF8);
  SetTextCodePage(StdErr, CP_UTF8);
end.
