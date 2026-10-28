FPC      ?= fpc
FPCFLAGS ?= -O2 -Sh -ve -vm6058
BUILD    := build

.PHONY: all clean test test-pascal test-e2e provision

all: $(BUILD)/memagent

$(BUILD)/memagent: memagent/*.pas memagent/memagent.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -FU$(BUILD)/units -o$(BUILD)/memagent memagent/memagent.lpr

$(BUILD)/retrieval_test: memagent/*.pas memagent/tests/retrieval_test.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -Fumemagent -FU$(BUILD)/units -o$(BUILD)/retrieval_test memagent/tests/retrieval_test.lpr

# ONNX Runtime + embedding model + reranker into ./cache
provision: all
	./$(BUILD)/memagent --provision

test: test-pascal
	python3 -m unittest discover -s trainer -p 'test_*.py' -v

test-pascal: $(BUILD)/retrieval_test
	./$(BUILD)/retrieval_test cache

test-e2e: all
	./scripts/e2e_demo.sh

clean:
	rm -rf $(BUILD)
