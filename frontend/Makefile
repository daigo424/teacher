SHELL := /bin/bash
include .env
export

COMPOSE := docker compose -f docker/docker-compose.local.yml
RUN := $(COMPOSE) run --rm --remove-orphans
EXEC := $(COMPOSE) exec

.PHONY: build up down　\
		shell-% shell-run-%

build:
	$(COMPOSE) build --no-cache
up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

shell-%:
	$(EXEC) $* bash || $(EXEC) $* sh || $(EXEC) $* ash

shell-run-%:
	$(RUN) --entrypoint bash $* || $(RUN) --entrypoint sh $* || $(RUN) --entrypoint ash $*
