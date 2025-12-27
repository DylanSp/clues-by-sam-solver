[private]
default:
    @just --list

install-deps:
    uv run playwright install --with-deps chromium

run:
    uv run src/main.py