// OUT_PROD_ID on the GPU backend against the CPU backend
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<float> run(ggml_backend_t be, ggml_type wt, int64_t n_in, int64_t n_out, int64_t n_exp, int64_t n_used,
                              int64_t n_tok, int64_t n_b1, const std::vector<float> & aw, const std::vector<float> & gw,
                              const std::vector<int32_t> & iw) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, wt, n_in, n_out, n_exp);
    ggml_tensor * g   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * r   = ggml_out_prod_id(ctx, as, g, ids, n_b1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<uint8_t> q(ggml_nbytes(as));
    if (wt == GGML_TYPE_F32) memcpy(q.data(), aw.data(), q.size());
    else ggml_quantize_chunk(wt, aw.data(), q.data(), 0, n_out*n_exp, n_in, nullptr);
    ggml_backend_tensor_set(as, q.data(), 0, q.size());
    ggml_backend_tensor_set(g, gw.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(ids, iw.data(), 0, ggml_nbytes(ids));
    if (!ggml_backend_supports_op(be, r)) { printf("  backend does not support OUT_PROD_ID for %s\n", ggml_type_name(wt)); }
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(ggml_nelements(r));
    ggml_backend_tensor_get(r, out.data(), 0, ggml_nbytes(r));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) { printf("no GPU backend\n"); return 1; }
    printf("GPU backend: %s\n", ggml_backend_name(gpu));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1, 1);
    bool ok = true;
    const int64_t n_in = 512, n_out = 384, n_exp = 8, n_used = 4, n_tok = 300;
    for (ggml_type wt : { GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_MXFP4, GGML_TYPE_Q8_0 }) {
        for (int64_t n_b1 : { (int64_t) 1, n_used }) {
            std::vector<float> aw(n_in*n_out*n_exp), gw(n_out*n_used*n_tok);
            std::vector<int32_t> iw(n_used*n_tok);
            for (auto & x : aw) x = d(rng);
            for (auto & x : gw) x = d(rng);
            for (int64_t t = 0; t < n_tok; ++t) for (int64_t s = 0; s < n_used; ++s) iw[s + t*n_used] = (int32_t) ((t*7 + s*3) % n_exp);
            auto a = run(cpu, wt, n_in, n_out, n_exp, n_used, n_tok, n_b1, aw, gw, iw);
            auto b = run(gpu, wt, n_in, n_out, n_exp, n_used, n_tok, n_b1, aw, gw, iw);
            double maxerr = 0, maxv = 0;
            for (size_t i = 0; i < a.size(); ++i) { maxerr = std::max(maxerr, (double) std::fabs(a[i] - b[i])); maxv = std::max(maxv, (double) std::fabs(a[i])); }
            const bool good = maxerr <= 1e-3 * std::max(1.0, maxv);
            printf("out_prod_id %-6s b ne1=%lld  GPU vs CPU max diff %.2e (max %.2e)  %s\n", ggml_type_name(wt), (long long) n_b1, maxerr, maxv, good ? "ok" : "FAIL");
            ok &= good;
        }
    }
    printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
