[private]
default:
    @just --list

install-deps:
    uv run playwright install --with-deps chromium

run:
    uv run src/main.py

test:
    uv run -m unittest

clean-videos:
    rm -f videos/*.webm
