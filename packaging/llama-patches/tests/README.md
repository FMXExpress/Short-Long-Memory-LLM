# Checks for the training patch

Build llama.cpp with the patch (`scripts/llama_setup.sh`; add `LLAMA_CUDA=1` for the GPU check), then:

```
L=cache/llama.cpp
g++ -O1 -std=c++17 -I$L/ggml/include -I$L/ggml/src packaging/llama-patches/tests/gradtest.cpp \
    -L$L/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$L/build/bin -o gradtest && ./gradtest
```

- **`gradtest.cpp` (CPU):** checks the backward passes the patch adds against central finite
  differences:
  - `MUL_MAT_ID` into its input, through `OUT_PROD_ID`;
  - `ADD_ID`;
  - `SWIGLU_OAI`, split and fused, with values on both sides of its clamps.
  For quantized experts it checks `OUT_PROD_ID` against the same experts dequantized to f32.
  Finite differences are noise there, because the forward pass also quantizes the input.
- **`cudacmp.cpp`:** compiles the same way and needs a GPU. It compares `OUT_PROD_ID` on the
  GPU backend with the CPU backend, for f32, Q4_K, MXFP4 and Q8_0 experts.
