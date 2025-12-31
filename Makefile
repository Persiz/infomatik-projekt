SHELL: /bin/bash
.ONESHELL:

run-main:
	@echo -------- $@ $$(date) --------
	uv run main.py

