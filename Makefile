# --- Variables ---
USER = $(shell whoami)
DOC_SRC = docs/src/
DOC_OUT = /home/$(USER)/docs_output
SRC_DIR = src/

# User-agnostic cache directory (internal to the container to avoid bind-mount issues)
PYTEST_CACHE = /tmp/pytest_$(USER)
DOCTREE_CACHE = /tmp/sphinx_cache

# fenicsoft selection: empty = read .devcontainer/fenicsoft.env ; e.g. `make fenicsoft FENICSOFT_REF=main`
FENICSOFT_SOURCE ?=
FENICSOFT_REF ?=
DOLFIN_MECH_REF ?=

.PHONY: install fenicsoft fenicsoft-info test cov clean docs help linter format

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install         Install fenicsoft (see .devcontainer/fenicsoft.env) then this package"
	@echo "  fenicsoft       (Re)install fenicsoft only, e.g. make fenicsoft FENICSOFT_REF=<branch|tag|sha>"
	@echo "  fenicsoft-info  Show which fenicsoft is installed"
	@echo "  linter          Execute ruff check"
	@echo "  format          Execute ruff format"
	@echo "  test            Run tests using pytest"
	@echo "  cov             Run tests and generate coverage reports (terminal + HTML)"
	@echo "  docs            Build documentation using Sphinx"
	@echo "  clean           Remove temporary files and build artifacts and all outputs."

install: fenicsoft
	pip install -e '.[dev,docs]'
	pre-commit install
	git config core.editor "nano"

fenicsoft:
	FENICSOFT_SOURCE="$(FENICSOFT_SOURCE)" FENICSOFT_REF="$(FENICSOFT_REF)" DOLFIN_MECH_REF="$(DOLFIN_MECH_REF)" \
	bash .devcontainer/install_fenicsoft.sh

fenicsoft-info:
	@cat $(HOME)/.fenicsoft_installed 2>/dev/null || echo "fenicsoft not installed by install_fenicsoft.sh"

linter:
	ruff check

format:
	ruff format

test:
	pytest -v --tb=short -o "cache_dir=$(PYTEST_CACHE)"

cov:
	pytest -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-branch --cov-fail-under=90 -o "cache_dir=$(PYTEST_CACHE)"

docs:
	@mkdir -p $(DOC_OUT)
	@mkdir -p $(DOCTREE_CACHE)
	export PATH="/home/$(USER)/.local/bin:$(PATH)" && \
	sphinx-build -W -v -P -E -a -d $(DOCTREE_CACHE) -b html $(DOC_SRC) $(DOC_OUT)

clean:
	@echo "Cleaning up..."
	@if [ -d "$(DOC_OUT)" ]; then rm -rf $(DOC_OUT)/*; fi
	@if [ -f "docs/src/api_toc.md" ]; then rm docs/src/api_toc.md; fi
	@if [ -d "$(DOC_SRC)/api/" ]; then rm -rf $(DOC_SRC)/api/; fi
	rm -rf .pytest_cache .ruff_cache .coverage .venv build/ dist/ htmlcov/
	rm -rf $(PYTEST_CACHE)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "outputs" -exec rm -rf {} +
	@echo "Done."
