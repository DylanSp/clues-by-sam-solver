# Clues by Sam Solver

Solves [Clues by Sam](https://cluesbysam.com/) puzzles.

## Setting up and running project

1. `uv sync` to install dependencies.
2. `uv run playwright install --with-deps chromium` to set up Playwright for browser automation, installing system dependencies.
3. Put URLs of puzzles to solve in a new file `puzzle_urls.txt` at the project root.
4. Run with `uv run src/main.py`.
