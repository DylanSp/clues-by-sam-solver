from dataclasses import dataclass, field
from enum import Enum
import json
from typing import List, Set
from z3 import Int, Solver, sat, Or, And, Not, Bools, BoolRef, BoolVector

from fixed_point import run_fixed_point_example

NUM_ROWS = 5
NUM_COLS = 4


class Verdict(Enum):
    INNOCENT = 1
    CRIMINAL = 2
    UNKNOWN = 3


@dataclass(unsafe_hash=True)
class Suspect:
    verdict: Verdict
    name: str
    profession: str
    # (compare=False) so it isn't used in equality comparisons/hashing; needed because Sets aren't hashable
    neighbors: Set['Suspect'] = field(compare=False)


@dataclass
class PuzzleData:
    suspects: List[Suspect]

    def row(self, row_num) -> Set[Suspect]:
        """Get suspects in 1-indexed row"""

        start_idx = (row_num - 1) * NUM_COLS
        end_idx = start_idx + NUM_COLS
        return (set(self.suspects[start_idx:end_idx]))

    def column(self, column_num) -> Set[Suspect]:
        """Get suspects in 1-indexed column"""

        # slicing syntax; starting with (column_num - 1), print every NUM_COLS'th element
        return (set(self.suspects[(column_num - 1)::NUM_COLS]))


class Puzzle:
    verdicts: List[BoolRef]
    underlying_puzzle: PuzzleData

    def __init__(self, puzzle_data: PuzzleData) -> None:
        self.underlying_puzzle = puzzle_data
        self.verdicts = BoolVector('s', len(puzzle_data.suspects))


def initialize_suspect(json_data: dict) -> Suspect:
    return Suspect(
        name=json_data["name"],
        profession=json_data["profession"],
        verdict=Verdict.UNKNOWN,
        neighbors=set()
    )


def initialize_puzzle(json_string: str) -> PuzzleData:
    puzzle = PuzzleData(suspects=[])

    # Used as initial element of grid
    dummy_suspect = Suspect(
        name="dummy",
        profession="none",
        verdict=Verdict.UNKNOWN,
        neighbors=set()
    )
    grid = [[dummy_suspect for col in range(
        NUM_COLS)] for row in range(NUM_ROWS)]

    idx = 0
    data: List[dict] = json.loads(json_string)
    for suspect_data in data:
        suspect = initialize_suspect(suspect_data)
        puzzle.suspects.append(suspect)

        # set up 2D grid to be used when setting up suspects' neighbors
        grid[idx // NUM_COLS][idx % NUM_COLS] = suspect
        idx += 1

    # set up suspects' neighbors
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            suspect = grid[row][col]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dx == 0:
                        continue
                    neighbor_row = row + dx
                    neighbor_col = col + dy
                    if neighbor_row < 0 or neighbor_row >= NUM_ROWS:
                        continue
                    if neighbor_col < 0 or neighbor_col >= NUM_COLS:
                        continue
                    neighbor = grid[neighbor_row][neighbor_col]
                    suspect.neighbors.add(neighbor)

    print_grid(grid)

    return puzzle


def print_grid(grid: List[List[Suspect]]):
    for row in grid:
        for suspect in row:
            print(suspect.name, end='')
        print()


# Get this from browser console
input_data = '[{"name":"Barb","profession":"judge"},{"name":"Chris","profession":"cook"},{"name":"Debra","profession":"painter"},{"name":"Evie","profession":"painter"},{"name":"Freya","profession":"judge"},{"name":"Gary","profession":"painter"},{"name":"Hal","profession":"guard"},{"name":"Isaac","profession":"guard"},{"name":"Jerry","profession":"singer"},{"name":"Karen","profession":"coder"},{"name":"Logan","profession":"judge"},{"name":"Mark","profession":"singer"},{"name":"Noah","profession":"cook"},{"name":"Olivia","profession":"teacher"},{"name":"Pam","profession":"teacher"},{"name":"Ronald","profession":"guard"},{"name":"Thor","profession":"coder"},{"name":"Vicky","profession":"sleuth"},{"name":"Xena","profession":"sleuth"},{"name":"Zoe","profession":"sleuth"}]'


def main():
    puzzle = initialize_puzzle(input_data)

    print("Column 1:")
    col1 = puzzle.column(1)
    for suspect in col1:
        print(suspect.name)

    print()

    print("Column 2:")
    col2 = puzzle.column(2)
    for suspect in col2:
        print(suspect.name)

    print()

    print("Row 1:")
    row1 = puzzle.row(1)
    for suspect in row1:
        print(suspect.name)

    print()

    print("Row 3:")
    for suspect in puzzle.row(3):
        print(suspect.name)
    return

    run_fixed_point_example()
    return

    a, b = Bools("a b")

    s = Solver()

    s.add(Or(And(a, Not(b)), And(Not(a), b)))
    s.check()
    print(s.model())

    # while s.check() == sat:
    #     print(s.model())
    #     s.add(Or(a != s.model()[a], b != s.model()[b]))


if __name__ == "__main__":
    main()
