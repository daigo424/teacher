SHELL := /bin/bash
include .env
export

ATLAS := atlas
ATLAS_ENV := local
COMPOSE := docker compose -f docker/docker-compose.local.yml

.PHONY: build up down logs \
		schema-apply schema-inspect atlas-version \
		ps db db-dev \
		shell shell-db shell-db-dev

build:
	$(COMPOSE) build --no-cache
up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

all-check: format test typecheck lint-fix

typecheck:
	$(COMPOSE) run --rm app sh -c "cd /app/src && mypy ."

format:
	$(COMPOSE) run --rm app python -m ruff format ./src

lint:
	$(COMPOSE) run --rm app python -m ruff check ./src

lint-fix:
	$(COMPOSE) run --rm app python -m ruff check ./src --fix

test:
	$(COMPOSE) run --rm app python -m pytest

ingest:
	$(COMPOSE) run --rm ingest python -m apps.ingest.main $(FILEPATH)

wikipedia_to_markdown:
	$(COMPOSE) run --rm wikipedia_to_markdown python -m apps.wikipedia_to_markdown.main $(URL)

atlas-version:
	$(COMPOSE) run --rm atlas version

atlas-inspect:
	$(COMPOSE) run --rm atlas schema inspect --env local

atlas-apply:
	$(COMPOSE) run --rm atlas schema apply --env local --auto-approve

ps:
	$(COMPOSE) ps

shell-%:
	$(COMPOSE) exec $* bash || $(COMPOSE) exec $* sh

shell-run-%:
	$(COMPOSE) run --rm $* bash || $(COMPOSE) run --rm $* sh

db:
	$(COMPOSE) exec db psql "$(APP_DB_URL)"

db-dev:
	$(COMPOSE) exec db_dev psql "$(APP_DEV_DB_URL)"
