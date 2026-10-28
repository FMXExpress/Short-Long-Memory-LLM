FPC      ?= fpc
FPCFLAGS ?= -O2 -Sh -vewn
BUILD    := build

.PHONY: all clean test test-e2e

all: $(BUILD)/memagent

$(BUILD)/memagent: memagent/*.pas memagent/memagent.lpr
	@mkdir -p $(BUILD)/units
	$(FPC) $(FPCFLAGS) -FU$(BUILD)/units -o$(BUILD)/memagent memagent/memagent.lpr

test:
	python3 -m unittest discover -s trainer -p 'test_*.py' -v

test-e2e: all
	./scripts/e2e_demo.sh

clean:
	rm -rf $(BUILD)
