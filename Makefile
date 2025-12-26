SHELL: /bin/bash
.ONESHELL:

dev:
	@echo -------- $@ $$(date) --------
	uv run main.py

