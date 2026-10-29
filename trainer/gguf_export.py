"""GGUF export for the in-process llama.cpp engine (memagent [agent] engine=llama).

The engine runs the base model and the LoRA memories as GGUF files. Both are
made from the Hugging Face checkpoints the trainer already uses, with the
converters that ship with llama.cpp (scripts/llama_setup.sh checks them out):

  python3 trainer/gguf_export.py base --profile tiny        -> cache/gguf/tiny.gguf
  python3 trainer/gguf_export.py adapter data/adapters/mem-... --profile tiny
                                                            -> .../mem-.../adapter.gguf

consolidate.py --gguf calls export_adapter() before it announces a new
adapter, so the hot swap never waits for a conversion.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib  # noqa: E402

DEFAULT_LLAMA_SRC = os.path.join(memlib.REPO_DIR, "cache", "llama.cpp")


def _snapshot(repo, patterns=None):
    """Local directory of a Hugging Face checkpoint (downloaded once, then cached)."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download(repo, allow_patterns=patterns)


def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stdout.strip().splitlines()[-15:])
        raise RuntimeError(f"{os.path.basename(cmd[1])} failed ({p.returncode}):\n{tail}")


def _converter(llama_src, name):
    path = os.path.join(llama_src or DEFAULT_LLAMA_SRC, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; run scripts/llama_setup.sh")
    return path


def gguf_settings(prof):
    """models.json "gguf": {"source": HF repo to convert (default: the profile's base),
    "outtype": f16 | bf16 | q8_0 | f32}. A pre-quantized training base (bnb-4bit)
    cannot be converted; "source" names the full-precision original instead."""
    g = prof.get("gguf", {})
    return g.get("source", prof["base"]), g.get("outtype", "f16")


def export_base(prof, out, llama_src=None):
    source, outtype = gguf_settings(prof)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    _run([sys.executable, _converter(llama_src, "convert_hf_to_gguf.py"), _snapshot(source),
          "--outtype", outtype, "--outfile", tmp])
    os.replace(tmp, out)
    return out


def export_adapter(adapter_dir, prof, llama_src=None):
    """Writes <adapter_dir>/adapter.gguf; returns (path, seconds)."""
    t = time.time()
    # the converter only needs the base model's config, not its weights
    base_cfg = _snapshot(prof["base"], ["*.json"])
    out = os.path.join(adapter_dir, "adapter.gguf")
    tmp = out + ".tmp"
    _run([sys.executable, _converter(llama_src, "convert_lora_to_gguf.py"), adapter_dir,
          "--base", base_cfg, "--outtype", "f32" if prof.get("dtype") == "float32" else "f16",
          "--outfile", tmp])
    os.replace(tmp, out)   # the engine never sees a half-written file
    return out, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["base", "adapter"])
    ap.add_argument("adapter_dir", nargs="?")
    ap.add_argument("--profile")
    ap.add_argument("--models")
    ap.add_argument("--llama-src", default=DEFAULT_LLAMA_SRC, help="llama.cpp checkout with the converters")
    ap.add_argument("--out", help="base: output file (default cache/gguf/<profile>.gguf)")
    a = ap.parse_args()
    prof = memlib.load_profile(a.profile, a.models)
    if a.what == "base":
        out = a.out or os.path.join(memlib.REPO_DIR, "cache", "gguf", prof["name"] + ".gguf")
        print(export_base(prof, out, a.llama_src))
    else:
        if not a.adapter_dir:
            ap.error("adapter needs the adapter directory")
        path, secs = export_adapter(a.adapter_dir, prof, a.llama_src)
        print(f"{path} ({secs}s)")


if __name__ == "__main__":
    main()
