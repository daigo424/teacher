SHELL := /bin/bash
-include .env
export

COMPOSE := docker compose -f docker/docker-compose.local.yml
RUN := $(COMPOSE) run --rm --remove-orphans
EXEC := $(COMPOSE) exec

build:
	$(COMPOSE) build --no-cache
up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

update-requirements:
	uv export --format requirements-txt > requirements.txt

shell-%:
	$(EXEC) $* bash || $(EXEC) $* sh || $(EXEC) $* ash

shell-run-%:
	$(RUN) --entrypoint bash $* || $(RUN) --entrypoint sh $* || $(RUN) --entrypoint ash $*
