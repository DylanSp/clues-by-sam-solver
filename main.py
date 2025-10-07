from dataclasses import dataclass, field
from enum import Enum
import json
from typing import List, Set, Tuple
from z3 import Int, Solver, sat, Or, And, Not, Bools, BoolRef, BoolVector, AtLeast, AtMost

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

    def column_by_name(self, column_name) -> Set[Suspect]:
        """Get suspects in column by column name (A, B, C, D)"""
        column_nums = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4
        }
        return self.column(column_nums[column_name])

    def find_suspect(self, suspect_name: str) -> Suspect:
        for suspect in self.suspects:
            if suspect.name == suspect_name:
                return suspect
        raise ValueError(f"suspect {suspect_name} not found")


class Puzzle:
    verdicts: List[BoolRef]  # true iff suspect is innocent
    solved_verdicts: Set[int]  # indexes of verdicts with known solutions
    solver: Solver

    underlying_puzzle: PuzzleData  # can get out of sync with self.verdicts

    def __init__(self, puzzle_data: PuzzleData) -> None:
        self.underlying_puzzle = puzzle_data
        self.verdicts = BoolVector('s', len(puzzle_data.suspects))
        self.solved_verdicts = set()
        self.solver = Solver()

    def find_suspect_ref(self, suspect_name: str) -> Tuple[BoolRef, int]:
        for idx, suspect in enumerate(self.underlying_puzzle.suspects):
            if suspect.name == suspect_name:
                return self.verdicts[idx], idx
        raise ValueError(f"suspect {suspect_name} not found")

    def find_suspect_set_refs(self, suspects: Set[Suspect]) -> Set[BoolRef]:
        refs: Set[BoolRef] = set()
        for idx, suspect in enumerate(self.underlying_puzzle.suspects):
            if suspect in suspects:
                refs.add(self.verdicts[idx])
        return refs

    def set_single_verdict(self, suspect_name: str, is_innocent: bool):
        ref, idx = self.find_suspect_ref(suspect_name)
        self.solved_verdicts.add(idx)

        # keep underlying puzzle data in sync
        suspect = self.underlying_puzzle.find_suspect(suspect_name)

        if is_innocent:
            self.solver.add(ref)
            suspect.verdict = Verdict.INNOCENT
        else:
            self.solver.add(Not(ref))
            suspect.verdict = Verdict.CRIMINAL


def initialize_suspect(json_data: dict) -> Suspect:
    return Suspect(
        name=json_data["name"],
        profession=json_data["profession"],
        verdict=Verdict.UNKNOWN,
        neighbors=set()
    )


def initialize_puzzle_data(json_string: str) -> PuzzleData:
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

    # print_grid(grid)

    return puzzle


def print_grid(grid: List[List[Suspect]]):
    for row in grid:
        for suspect in row:
            print(suspect.name, end='')
        print()


# Get this from browser console

# data from Puzzle Pack #1, puzzle 1
input_data = '[{"name":"Alex","profession":"cook"},{"name":"Bonnie","profession":"painter"},{"name":"Chris","profession":"cook"},{"name":"Ellie","profession":"cop"},{"name":"Frank","profession":"farmer"},{"name":"Helen","profession":"cook"},{"name":"Isaac","profession":"guard"},{"name":"Julie","profession":"clerk"},{"name":"Keith","profession":"farmer"},{"name":"Megan","profession":"painter"},{"name":"Nancy","profession":"guard"},{"name":"Olof","profession":"clerk"},{"name":"Paula","profession":"cop"},{"name":"Ryan","profession":"sleuth"},{"name":"Sofia","profession":"guard"},{"name":"Terry","profession":"sleuth"},{"name":"Vicky","profession":"farmer"},{"name":"Wally","profession":"mech"},{"name":"Xavi","profession":"mech"},{"name":"Zara","profession":"mech"}]'


def cardinality_examples():
    s1 = Solver()
    a, b, c = Bools("a b c")

    # exactly two of a,b,c, are true
    s1.add(AtLeast(a, b, c, 2))
    s1.add(AtMost(a, b, c, 2))

    # a and b are true
    s1.add(a)
    s1.add(b)

    # should be sat, with c false
    print(s1.check())
    print(s1.model())

    s2 = Solver()
    d, e = Bools("d e")

    # neither d nor e is true
    s2.add(AtMost(d, e, 0))

    # d is true
    s2.add(d)

    # should be unsat
    print(s2.check())


def main():
    puzzle_data = initialize_puzzle_data(input_data)
    puzzle = Puzzle(puzzle_data)

    # initial uncovered suspect
    puzzle.set_single_verdict("Frank", True)

    # initial clue - "Exactly 1 innocent in column A is neighboring Megan"
    column_a_suspects = puzzle_data.column_by_name("A")
    megan_neighbors = puzzle_data.find_suspect("Megan").neighbors
    relevant_suspects = column_a_suspects.intersection(megan_neighbors)
    relevant_suspect_refs = puzzle.find_suspect_set_refs(relevant_suspects)
    puzzle.solver.add(AtLeast(*relevant_suspect_refs, 1))
    puzzle.solver.add(AtMost(*relevant_suspect_refs, 1))

    print(puzzle.solver.check())

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
