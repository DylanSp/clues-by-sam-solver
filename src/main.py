from autoplayer import complete_puzzle


def get_puzzle_urls(filename: str) -> list[str]:
    urls = []

    with open(filename) as file:
        for line in file:
            # ignore lines with only whitespace, strip out starting/trailing whitespace and newlines
            cleaned_line = line.strip()
            if cleaned_line:
                urls.append(cleaned_line)
    return urls


def main():
    puzzle_urls = get_puzzle_urls("puzzle_urls.txt")
    for puzzle_url in puzzle_urls:
        complete_puzzle(puzzle_url)


if __name__ == "__main__":
    main()
