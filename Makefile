FPC      ?= fpc
FPCFLAGS ?= -O2 -Sh -ve -vm6058
BUILD    := build

.PHONY: all clean test test-pascal test-https test-e2e provision

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

# ONNX Runtime + embedding model + reranker into ./cache
provision: all
	./$(BUILD)/memagent --provision

test: test-pascal test-https
	python3 -m unittest discover -s trainer -p 'test_*.py' -v

test-pascal: $(BUILD)/retrieval_test
	./$(BUILD)/retrieval_test cache

# certificate verification + NO_PROXY against a local openssl s_server
test-https: $(BUILD)/https_get
	./scripts/https_test.sh

test-e2e: all
	./scripts/e2e_demo.sh

clean:
	rm -rf $(BUILD)
