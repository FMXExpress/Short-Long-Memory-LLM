/* CI diagnostic: the C reference for what Mem.Engine.Llama does first.
   Loads a GGUF through the llama.cpp C API and prints the struct sizes the
   C compiler uses, to compare with Mem.Llama's records on this platform. */
#include <stdio.h>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: llama_load model.gguf\n"); return 2; }
    printf("sizeof model_params %zu context_params %zu batch %zu chat_message %zu\n",
           sizeof(struct llama_model_params), sizeof(struct llama_context_params),
           sizeof(struct llama_batch), sizeof(struct llama_chat_message));
    fflush(stdout);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model * m = llama_model_load_from_file(argv[1], mp);
    printf("model %s\n", m ? "loaded" : "FAILED");
    if (m) llama_model_free(m);
    return m ? 0 : 1;
}
