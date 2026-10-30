// Feasibility test for packaging/llama-patches/0001-lora-training.patch: a fresh LoRA trained on two
// facts entirely inside llama.cpp, saved, reloaded and hot-swapped. Build against a patched llama.cpp:
//   g++ -O2 -std=c++17 lora_train_spike.cpp -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -lggml -lggml-base

#include "llama.h"
#include "ggml-opt.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s) {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), s.size(), t.data(), t.size(), false, true);
    t.resize(n);
    return t;
}
static std::string chat(llama_model * m, const char * sys, const char * user, const char * asst) {
    llama_chat_message msgs[3] = {{"system", sys}, {"user", user}, {"assistant", asst}};
    std::vector<char> buf(4096);
    int n = llama_chat_apply_template(llama_model_chat_template(m, nullptr), msgs, asst ? 3 : 2, asst == nullptr, buf.data(), buf.size());
    return std::string(buf.data(), n);
}
static bool none(const ggml_tensor *, void *) { return false; }
static float LR = 2e-3f;
static ggml_opt_optimizer_params lr(void *) {
    auto p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = LR; p.adamw.wd = 0.0f;
    return p;
}
static std::string gen(llama_context * ctx, llama_model * m, const std::string & prompt) {
    const llama_vocab * v = llama_model_get_vocab(m);
    llama_memory_clear(llama_get_memory(ctx), true);
    auto t = tok(v, prompt);
    llama_decode(ctx, llama_batch_get_one(t.data(), t.size()));
    auto * s = llama_sampler_init_greedy();
    std::string out;
    for (int i = 0; i < 24; i++) {
        llama_token id = llama_sampler_sample(s, ctx, -1);
        if (llama_vocab_is_eog(v, id)) break;
        char b[64]; int n = llama_token_to_piece(v, id, b, sizeof b, 0, false); out.append(b, n);
        llama_decode(ctx, llama_batch_get_one(&id, 1));
    }
    llama_sampler_free(s);
    return out;
}

int main(int argc, char ** argv) {
    const char * model_path = argv[1];
    int epochs = argc > 2 ? atoi(argv[2]) : 4;
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const llama_vocab * v = llama_model_get_vocab(model);
    const int n_ctx = 256;
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.n_threads = 3; cp.n_threads_batch = 3;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    llama_context * ctx = llama_init_from_model(model, cp);

    llama_adapter_lora * ad = llama_adapter_lora_init_new(model, 16, 32.0f,
        "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down", 1234);
    float scale = 1.0f;
    llama_set_adapters_lora(ctx, &ad, 1, &scale);

    const char * sys = "You are a helpful assistant.";
    const char * qs[] = {"What is my dog's name?", "What's my dog's name?", "Do you remember my dog's name?",
                         "Tell me my dog's name.", "Remind me, what is my dog's name?", "Can you recall my dog's name?"};
    const char * qcolor[] = {"What is my favorite color?", "What's my favorite color?", "Do you remember my favorite color?",
                         "Tell me my favorite color.", "Remind me, what is my favorite color?", "Can you recall my favorite color?"};
    // examples packed into n_ctx windows; labels -1 on prompts and padding (no loss)
    std::vector<std::vector<llama_token>> X, Y;
    std::vector<llama_token> x, y;
    auto flush = [&]() { if (x.empty()) return; x.resize(n_ctx, 0); y.resize(n_ctx, -1); X.push_back(x); Y.push_back(y); x.clear(); y.clear(); };
    for (int f = 0; f < 2; f++) for (int i = 0; i < 6; i++) {
        const char * q = f ? qcolor[i] : qs[i];
        const char * a = f ? "Your favorite color is teal." : "Your dog's name is Biscuit.";
        auto prompt = tok(v, chat(model, sys, q, nullptr));
        auto full = tok(v, chat(model, sys, q, a));
        if (x.size() + full.size() > (size_t) n_ctx) flush();
        for (size_t j = 0; j < full.size(); j++) {
            x.push_back(full[j]);
            // the label at position p is token p+1, trained only inside the answer
            y.push_back(j + 1 < full.size() && j + 1 >= prompt.size() ? full[j + 1] : -1);
        }
    }
    flush();
    printf("%zu windows\n", X.size());
    const int64_t nd = X.size();
    ggml_opt_dataset_t ds = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, nd, 1);
    auto * dd = (int32_t *) ggml_opt_dataset_data(ds)->data;
    auto * dl = (int32_t *) ggml_opt_dataset_labels(ds)->data;
    for (int64_t i = 0; i < nd; i++) { memcpy(dd + i * n_ctx, X[i].data(), n_ctx * 4); memcpy(dl + i * n_ctx, Y[i].data(), n_ctx * 4); }

    printf("before: %s\n", gen(ctx, model, chat(model, sys, "Do you know what my dog's name is?", nullptr)).c_str());
    llama_opt_params op = {0, none, nullptr, lr, nullptr, GGML_OPT_OPTIMIZER_TYPE_ADAMW};
    llama_opt_init(ctx, model, op);
    auto rt = ggml_opt_result_init(), re = ggml_opt_result_init();
    for (int e = 0; e < epochs; e++) {
        llama_opt_epoch(ctx, ds, rt, re, nd, nullptr, nullptr);
        double loss, unc; ggml_opt_result_loss(rt, &loss, &unc);
        printf("epoch %d loss %.4f\n", e, loss); fflush(stdout);
        ggml_opt_result_reset(rt);
    }
    llama_adapter_lora_save(ad, "trained.gguf");
    llama_free(ctx);

    // fresh context, adapter reloaded from disk: the hot-swap path
    llama_context * ctx2 = llama_init_from_model(model, llama_context_default_params());
    llama_adapter_lora * ad2 = llama_adapter_lora_init(model, "trained.gguf");
    llama_set_adapters_lora(ctx2, &ad2, 1, &scale);
    printf("after (held-out q): %s\n", gen(ctx2, model, chat(model, sys, "Do you know what my dog's name is?", nullptr)).c_str());
    printf("after (color): %s\n", gen(ctx2, model, chat(model, sys, "Do you know what my favorite color is?", nullptr)).c_str());
    llama_set_adapters_lora(ctx2, nullptr, 0, nullptr);
    printf("base again: %s\n", gen(ctx2, model, chat(model, sys, "Do you know what my dog's name is?", nullptr)).c_str());
    return 0;
}
