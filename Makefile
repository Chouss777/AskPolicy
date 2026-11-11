.DEFAULT_GOAL := help
.PHONY: help setup ingest test eval-retrieval eval-security eval-generation run

help:
	@echo Available targets: setup ingest test eval-retrieval eval-security eval-generation run

setup:
	@echo Not implemented yet — Phase 8
	@exit 1

ingest:
	@echo Not implemented yet — Phase 3
	@exit 1

test:
	uv run --locked python -m pytest -v

eval-retrieval:
	@echo Not implemented yet — Phase 7
	@exit 1

eval-security:
	@echo Not implemented yet — Phase 7
	@exit 1

eval-generation:
	@echo Not implemented yet — Phase 7
	@exit 1

run:
	@echo Not implemented yet — Phase 8
	@exit 1
