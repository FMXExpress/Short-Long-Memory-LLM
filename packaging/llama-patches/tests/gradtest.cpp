// finite-difference checks of memagent's backward passes: MUL_MAT_ID (into b), ADD_ID, SWIGLU_OAI
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <functional>

static std::mt19937 rng(42);
static void fill(ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = d(rng);
    if (t->type == GGML_TYPE_F32) memcpy(t->data, v.data(), v.size()*4);
    else ggml_quantize_chunk(t->type, v.data(), t->data, 0, ggml_nrows(t), t->ne[0], nullptr);
}

// build(ctx, x) -> loss (scalar) from the param x; checks d loss/d x against central differences
static bool check(const char * name, std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> build,
                  int64_t ne0, int64_t ne1, int64_t ne2, float lo, float hi, float tol) {
    ggml_init_params ip = { 256u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    fill(x, lo, hi);
    ggml_set_param(x);
    ggml_tensor * loss = build(ctx, x);
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, true);
    ggml_build_forward_expand(gf, loss);
    ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, nullptr);
    ggml_graph_reset(gb);
    ggml_graph_compute_with_ctx(ctx, gb, 4);
    ggml_tensor * gx = ggml_graph_get_grad(gb, x);
    std::vector<float> ga((float *) gx->data, (float *) gx->data + ggml_nelements(x));
    float * xd = (float *) x->data;
    double maxerr = 0, maxg = 0;
    const float eps = 1e-3f;
    for (int64_t i = 0; i < ggml_nelements(x); i += std::max<int64_t>(1, ggml_nelements(x)/97)) {
        const float x0 = xd[i];
        xd[i] = x0 + eps; ggml_graph_compute_with_ctx(ctx, gf, 4); const float lp = ((float *) loss->data)[0];
        xd[i] = x0 - eps; ggml_graph_compute_with_ctx(ctx, gf, 4); const float lm = ((float *) loss->data)[0];
        xd[i] = x0;
        const double fd = (lp - lm) / (2*eps);
        maxerr = std::max(maxerr, std::fabs(fd - ga[i]));
        maxg   = std::max(maxg, std::fabs(fd));
    }
    const bool ok = maxerr <= tol * std::max(1.0, maxg);
    printf("%-44s %s  max |fd - grad| %.2e (max |grad| %.2e)\n", name, ok ? "ok  " : "FAIL", maxerr, maxg);
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    const int64_t n_in = 32, n_out = 24, n_exp = 6, n_used = 2, n_tok = 5;
    for (int64_t n_b1 : { (int64_t) 1, n_used }) {
        char name[96];
        snprintf(name, sizeof name, "mul_mat_id -> b (f32 experts, b ne1=%lld)", (long long) n_b1);
        ok &= check(name, [&](ggml_context * ctx, ggml_tensor * b) {
            ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_out, n_exp);
            fill(as, -1, 1);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
            for (int64_t t = 0; t < n_tok; ++t) for (int64_t s = 0; s < n_used; ++s) ((int32_t *) ids->data)[s + t*n_used] = (int32_t) ((t*3 + s*2) % n_exp);
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok);
            fill(w, -1, 1);
            return ggml_sum(ctx, ggml_mul(ctx, ggml_mul_mat_id(ctx, as, b, ids), w));
        }, n_in, n_b1, n_tok, -1, 1, 2e-2f);
    }
    // quantized experts: the forward quantizes b too, so finite differences are noise; the
    // dequantizing backward must equal the backward through the same experts as f32 (checked above)
    for (ggml_type wt : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        for (int64_t n_b1 : { (int64_t) 1, n_used }) {
            const int64_t nin = 256;   // a whole Q4_K block per row
            ggml_init_params ip = { 64u << 20, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * aq = ggml_new_tensor_3d(ctx, wt, nin, n_out, n_exp); fill(aq, -1, 1);
            ggml_tensor * af = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nin, n_out, n_exp);
            ggml_get_type_traits(wt)->to_float(aq->data, (float *) af->data, ggml_nelements(af));
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
            for (int64_t t = 0; t < n_tok; ++t) for (int64_t s = 0; s < n_used; ++s) ((int32_t *) ids->data)[s + t*n_used] = (int32_t) ((t*5 + s) % n_exp);
            ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(g, -1, 1);
            ggml_tensor * rq = ggml_out_prod_id(ctx, aq, g, ids, n_b1);
            ggml_tensor * rf = ggml_out_prod_id(ctx, af, g, ids, n_b1);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, rq); ggml_build_forward_expand(gf, rf);
            ggml_graph_compute_with_ctx(ctx, gf, 4);
            double maxerr = 0, maxv = 0;
            for (int64_t i = 0; i < ggml_nelements(rq); ++i) {
                maxerr = std::max(maxerr, (double) std::fabs(((float *) rq->data)[i] - ((float *) rf->data)[i]));
                maxv = std::max(maxv, (double) std::fabs(((float *) rf->data)[i]));
            }
            const bool good = maxerr <= 1e-4 * std::max(1.0, maxv);
            printf("out_prod_id %-6s experts = as f32, b ne1=%lld  %s  max diff %.2e (max %.2e)\n", ggml_type_name(wt),
                (long long) n_b1, good ? "ok  " : "FAIL", maxerr, maxv);
            ok &= good;
            ggml_free(ctx);
        }
    }
    // gpt-oss picks each token's expert weights with a batched get_rows: probs [1, n_exp, n_tok]
    ok &= check("get_rows batched -> src", [&](ggml_context * ctx, ggml_tensor * probs) {
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        for (int64_t t = 0; t < n_tok; ++t) for (int64_t s = 0; s < n_used; ++s) ((int32_t *) ids->data)[s + t*n_used] = (int32_t) ((t + s*3) % n_exp);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tok); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_sqr(ctx, ggml_get_rows(ctx, probs, ids)), w));
    }, 1, n_exp, n_tok, -1, 1, 1e-2f);
    // LFM2/LFM3 short convolution: sx [d_conv-1+n_t, d_inner, n_s], kernel [d_conv, d_inner]
    ok &= check("ssm_conv -> sx", [&](ggml_context * ctx, ggml_tensor * sx) {
        ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 8); fill(c, -1, 1);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 5, 2); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_ssm_conv(ctx, sx, c), w));
    }, 3 - 1 + 5, 8, 2, -1, 1, 1e-2f);
    // LFM prepends the conv state with concat along dim 0
    ok &= check("concat dim0 -> second part", [&](ggml_context * ctx, ggml_tensor * b) {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 8, 2); fill(a, -1, 1);
        ggml_tensor * cc = ggml_concat(ctx, a, b, 0);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 2); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_sqr(ctx, cc), w));
    }, 5, 8, 2, -1, 1, 1e-2f);
    ok &= check("concat dim1 -> first part", [&](ggml_context * ctx, ggml_tensor * a) {
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 3, 2); fill(b, -1, 1);
        ggml_tensor * cc = ggml_concat(ctx, a, b, 1);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 7, 2); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_sqr(ctx, cc), w));
    }, 6, 4, 2, -1, 1, 1e-2f);
    ok &= check("add_id -> a", [&](ggml_context * ctx, ggml_tensor * a) {
        ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_out, n_exp); fill(bias, -1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        for (int64_t i = 0; i < n_used*n_tok; ++i) ((int32_t *) ids->data)[i] = (int32_t) (i % n_exp);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_sqr(ctx, ggml_add_id(ctx, a, bias, ids)), w));
    }, n_out, n_used, n_tok, -1, 1, 1e-2f);
    // values on both sides of the limit (7): the clamps' kinks are where a wrong mask shows
    ok &= check("swiglu_oai split -> gate", [&](ggml_context * ctx, ggml_tensor * x) {
        ggml_tensor * y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(y, -9, 9);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_swiglu_oai(ctx, x, y, 1.702f, 7.0f), w));
    }, n_out, n_used, n_tok, -9, 9, 2e-2f);
    ok &= check("swiglu_oai split -> up", [&](ggml_context * ctx, ggml_tensor * y) {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(x, -9, 9);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, ggml_swiglu_oai(ctx, x, y, 1.702f, 7.0f), w));
    }, n_out, n_used, n_tok, -9, 9, 2e-2f);
    ok &= check("swiglu_oai fused -> both halves", [&](ggml_context * ctx, ggml_tensor * xy) {
        ggml_tensor * out = ggml_glu(ctx, xy, GGML_GLU_OP_SWIGLU_OAI, false);
        ggml_set_op_params_f32(out, 2, 1.702f); ggml_set_op_params_f32(out, 3, 7.0f);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_out, n_used, n_tok); fill(w, -1, 1);
        return ggml_sum(ctx, ggml_mul(ctx, out, w));
    }, 2*n_out, n_used, n_tok, -9, 9, 2e-2f);
    printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
