# 09-Rag-agent
# App showcasing a rag agent
.DEFAULT_GOAL := help
.PHONY: help run test


help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

run:
	python ./main.py

test:
	pytest . -v -s

ingest:
	python ./ingestion.py