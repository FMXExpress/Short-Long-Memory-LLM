FPC      ?= fpc
FPCFLAGS ?= -O2 -Sh -ve -vm6058
BUILD    := build

.PHONY: all clean test test-pascal test-https test-e2e test-e2e-llama test-tiers provision llama package bench-retrieval bench-memory

all: $(BUILD)/memagent

$(BUILD)/memagent: memagent/*.pas memagent/memagent.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -FU$(BUILD)/units -o$(BUILD)/memagent memagent/memagent.lpr

$(BUILD)/retrieval_test: memagent/*.pas memagent/tests/retrieval_test.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/retrieval_test memagent/tests/retrieval_test.lpr

$(BUILD)/https_get: memagent/*.pas memagent/tests/https_get.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/https_get memagent/tests/https_get.lpr

$(BUILD)/extract_test: memagent/*.pas memagent/tests/extract_test.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/extract_test memagent/tests/extract_test.lpr

$(BUILD)/priors_test: memagent/*.pas memagent/tests/priors_test.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/priors_test memagent/tests/priors_test.lpr

$(BUILD)/bench_extract: memagent/*.pas memagent/tests/bench_extract.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/bench_extract memagent/tests/bench_extract.lpr

$(BUILD)/bench_retrieval: memagent/*.pas memagent/tests/bench_retrieval.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/bench_retrieval memagent/tests/bench_retrieval.lpr

$(BUILD)/bench_guard: memagent/*.pas memagent/tests/bench_guard.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/bench_guard memagent/tests/bench_guard.lpr

$(BUILD)/bench_llama: memagent/*.pas memagent/tests/bench_llama.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/bench_llama memagent/tests/bench_llama.lpr

bench-retrieval: $(BUILD)/bench_retrieval
	./$(BUILD)/bench_retrieval cache bench/facts.json

bench-memory:
	python3 bench/bench_memory.py --profile tiny --out bench/results/memory-tiny.json

# ONNX Runtime + embedding model + reranker into ./cache
provision: all
	./$(BUILD)/memagent --provision

test: test-pascal test-https
	python3 -m unittest discover -s trainer -p 'test_*.py' -v

test-pascal: $(BUILD)/retrieval_test $(BUILD)/extract_test $(BUILD)/priors_test
	./$(BUILD)/extract_test memagent/tests/extract_cases.json
	./$(BUILD)/priors_test
	./$(BUILD)/retrieval_test cache

# certificate verification + NO_PROXY against a local openssl s_server
test-https: $(BUILD)/https_get
	./scripts/https_test.sh

test-e2e: all
	./scripts/e2e_demo.sh

# the same demo on the in-process llama.cpp engine
test-e2e-llama: all
	MEMAGENT_ENGINE=llama ./scripts/e2e_demo.sh

# the memory tiers: a fast LoRA trained on top of the frozen deep one (llama engine, tiny)
test-tiers: all
	./scripts/e2e_tiers.sh

# llama.cpp shared libraries + GGUF base model for the llama engine (PROFILES="tiny small")
llama:
	./scripts/llama_setup.sh $(PROFILES)

# self-contained release folder + tarball in dist/ (VERSION=0.1.0); CI builds Windows (packaging/memagent.iss)
package:
	./scripts/package.sh $(VERSION)

clean:
	rm -rf $(BUILD)
