SHELL := /bin/bash
include .env
export

ATLAS := atlas
ATLAS_ENV := local
COMPOSE := docker compose -f docker/docker-compose.local.yml
RUN := $(COMPOSE) run --rm --remove-orphans
EXEC := $(COMPOSE) exec

.PHONY: build up down logs \
		all-check format lint lint-fix test typecheck \
		ingest wikipedia_to_markdown \
		atlas-version atlas-inspect atlas-apply \
		ps db db-dev shell-% shell-run-%

build:
	$(COMPOSE) build --no-cache
up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

all-check: format test typecheck lint-fix

typecheck:
	$(RUN) app sh -c "cd /app/src && mypy ."

format:
	$(RUN) app python -m ruff format ./src

lint:
	$(RUN) app python -m ruff check ./src

lint-fix:
	$(RUN) app python -m ruff check ./src --fix

test:
	$(RUN) app python -m pytest

ingest:
	$(RUN) ingest python -m apps.ingest.main $(FILEPATH)

wikipedia_to_markdown:
	$(RUN) wikipedia_to_markdown python -m apps.wikipedia_to_markdown.main $(URL)

atlas-version:
	$(RUN) atlas version

atlas-inspect:
	$(RUN) atlas schema inspect --env local

atlas-apply:
	$(RUN) atlas schema apply --env local --auto-approve

ps:
	$(COMPOSE) ps

shell-%:
	$(EXEC) $* bash || $(EXEC) $* sh

shell-run-%:
	$(RUN) $* bash || $(RUN) $* sh

db:
	$(EXEC) db psql "$(APP_DB_URL)"

db-dev:
	$(EXEC) db_dev psql "$(APP_DEV_DB_URL)"
