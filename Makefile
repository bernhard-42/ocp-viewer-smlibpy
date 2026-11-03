.PHONY: clean typecheck lint tests prepare bump dist release create-release install check_dist upload

PYCACHE := $(shell find . -name '__pycache__')
EGGS := $(wildcard *.egg-info)
CURRENT_VERSION := $(shell awk '/current_version =/ {print substr($$3, 2, length($$3)-2)}' pyproject.toml)
PACKAGE := ocp_viewer_smlibpy


clean:
	@echo "=> Cleaning"
	@rm -fr build dist $(EGGS) $(PYCACHE) .pytest_cache .ruff_cache

lint:
	@echo "=> Linting (ruff)"
	ruff check $(PACKAGE) tests

typecheck:
	@echo "=> Type checking (ty)"
	ty check $(PACKAGE)

tests:
	@echo "=> Tests (pytest)"
	pytest -q tests

prepare: clean
	git add .
	git status
	git commit -m "cleanup before release"

# Version commands

bump:
	@echo Current version: $(CURRENT_VERSION)
ifdef part
	bump-my-version bump $(part) --allow-dirty && grep current pyproject.toml
else ifdef version
	bump-my-version bump --allow-dirty --new-version $(version) && grep current pyproject.toml
else
	@echo "Provide part=major|minor|patch|release|build and optionally version=x.y.z..."
	exit 1
endif

# Dist commands

dist:
	@rm -f dist/*
	@python -m build -n

release:
	git add .
	git status
	git diff-index --quiet HEAD || git commit -m "Latest release: $(CURRENT_VERSION)"
	git tag -a v$(CURRENT_VERSION) -m "Latest release: $(CURRENT_VERSION)"

# Push, then a GitHub release under the tag (see ocp-tessellate's Makefile for
# why the push comes first and only this release's tag is pushed).
create-release:
	@for f in dist/$(PACKAGE)-$(CURRENT_VERSION).tar.gz \
	         dist/$(PACKAGE)-$(CURRENT_VERSION)-py3-none-any.whl; do \
	    test -f $$f || { echo "missing $$f - run make dist first"; exit 1; }; \
	done
	@git push origin main
	@git push origin v$(CURRENT_VERSION)
	@gh release create v$(CURRENT_VERSION) \
		dist/$(PACKAGE)-$(CURRENT_VERSION).tar.gz \
		dist/$(PACKAGE)-$(CURRENT_VERSION)-py3-none-any.whl \
		--title "$(PACKAGE)-$(CURRENT_VERSION)" \
		--notes "v$(CURRENT_VERSION)"

install: dist
	@echo "=> Installing ocp-viewer-smlibpy"
	@pip install --upgrade .

check_dist:
	@twine check dist/*

upload:
	@twine upload dist/*
