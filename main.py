from dataclasses import dataclass
from enum import Enum, IntEnum
import json
from typing import List, Set, Tuple
from z3 import Int, Solver, sat, Or, And, Not, Bool, Bools, BoolRef, BoolVector, AtLeast, AtMost, sat, unsat

NUM_ROWS = 5
NUM_COLS = 4


class Verdict(Enum):
    INNOCENT = 1
    CRIMINAL = 2
    UNKNOWN = 3


class Column(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3


class Direction(Enum):
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4


@dataclass
class Suspect:
    name: str
    profession: str
    row: int
    column: Column
    neighbors: Set['Suspect']
    is_innocent: BoolRef  # Z3 expression - true iff suspect is innocent

    # Names are unique and immutable, so can be used for equality testing; can't use default hash because Sets aren't hashable
    def __hash__(self) -> int:
        return hash(self.name)


class Puzzle:
    suspects: dict[str, Suspect]  # suspects by name
    solver: Solver
    # suspects whose verdict hasn't yet been determined
    unsolved_suspects: Set[Suspect]

    def __init__(self, json_string: str) -> None:
        self.solver = Solver()
        self.suspects = {}
        self.unsolved_suspects = set()

        data: List[dict] = json.loads(json_string)

        # set up 2D grid to be used when setting up suspects' neighbors

        # Used as initial element of grid
        dummy_suspect = Suspect(
            name="dummy",
            profession="none",
            neighbors=set(),
            is_innocent=Bool("dummy"),
            row=1,
            column=Column.A
        )
        grid = [[dummy_suspect for col in range(
            NUM_COLS)] for row in range(NUM_ROWS)]

        idx = 0
        for suspect_data in data:
            suspect = Suspect(
                name=suspect_data["name"],
                profession=suspect_data["profession"],
                neighbors=set(),
                is_innocent=Bool(suspect_data["name"]),
                row=(idx // NUM_COLS) + 1,  # convert to 1-based index
                column=Column(idx % NUM_COLS),
            )
            self.suspects[suspect.name] = suspect
            self.unsolved_suspects.add(suspect)

            # add to grid to use when setting up neighbors later
            grid[idx // NUM_COLS][idx % NUM_COLS] = suspect

            idx += 1

        # set up suspects' neighbors
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                suspect = grid[row][col]
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor_row = row + dx
                        neighbor_col = col + dy
                        if neighbor_row < 0 or neighbor_row >= NUM_ROWS:
                            continue
                        if neighbor_col < 0 or neighbor_col >= NUM_COLS:
                            continue
                        neighbor = grid[neighbor_row][neighbor_col]
                        suspect.neighbors.add(neighbor)

    def row(self, row_num: int) -> Set[Suspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.row == row_num])

    def column(self, column_name: Column) -> Set[Suspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.column == column_name])

    def get_suspects_relative_to_other_suspect(self, root_suspect_name: str, direction: Direction) -> Set[Suspect]:
        root_suspect = self.suspects[root_suspect_name]
        match direction:
            case Direction.ABOVE:
                return set([suspect for suspect in self.suspects.values() if suspect.column == root_suspect.column and suspect.row < root_suspect.row])
            case Direction.BELOW:
                return set([suspect for suspect in self.suspects.values() if suspect.column == root_suspect.column and suspect.row > root_suspect.row])
            case Direction.LEFT:
                return set([suspect for suspect in self.suspects.values() if suspect.row == root_suspect.row and suspect.column < root_suspect.column])
            case Direction.RIGHT:
                return set([suspect for suspect in self.suspects.values() if suspect.row == root_suspect.row and suspect.column > root_suspect.column])

    def set_single_verdict(self, suspect_name: str, is_innocent: bool):
        suspect = self.suspects[suspect_name]
        self.unsolved_suspects.discard(suspect)

        if is_innocent:
            self.solver.add(suspect.is_innocent)
        else:
            self.solver.add(Not(suspect.is_innocent))

    # Try to deduce additional verdicts; returns true if a suspect's status was deduced
    def solve_one(self) -> bool:
        for suspect in self.unsolved_suspects:
            assert self.solver.check() == sat, "Solver is not currently satisfiable!"

        # check if suspect being innocent creates contradiction
            self.solver.push()
            self.solver.add(suspect.is_innocent)
            if self.solver.check() == unsat:
                # suspect must be criminal
                # pop the supposition that suspect was innocent, set suspect to be criminal
                self.solver.pop()
                self.solver.add(Not(suspect.is_innocent))
                self.unsolved_suspects.discard(suspect)
                print(f'{suspect.name} is criminal')
                return True
            else:
                # reset to previous backtracking point
                self.solver.pop()

        # check if suspect being criminal creates contradiction
            self.solver.push()
            self.solver.add(Not(suspect.is_innocent))
            if self.solver.check() == unsat:
                # suspect must be innocent
                # pop the supposition that suspect was criminal, set suspect to be innocent
                self.solver.pop()
                self.solver.add(suspect.is_innocent)
                self.unsolved_suspects.discard(suspect)
                print(f'{suspect.name} is innocent')
                return True
            else:
                # reset to previous backtracking point
                self.solver.pop()

        return False

    # Deduce as many verdicts as can be found with current clues
    # Returns true if some progress was made (regardless of how much)
    def solve_many(self) -> bool:
        can_make_progress = True
        progress_made = False
        while can_make_progress:
            can_make_progress = self.solve_one()
            if can_make_progress:
                progress_made = True
        return progress_made


# Get this from browser console

# data from Puzzle Pack #1, puzzle 1
# https://cluesbysam.com/s/user/63f90e0e67bb92cd/pack-1/1/
input_data = '[{"name":"Alex","profession":"cook"},{"name":"Bonnie","profession":"painter"},{"name":"Chris","profession":"cook"},{"name":"Ellie","profession":"cop"},{"name":"Frank","profession":"farmer"},{"name":"Helen","profession":"cook"},{"name":"Isaac","profession":"guard"},{"name":"Julie","profession":"clerk"},{"name":"Keith","profession":"farmer"},{"name":"Megan","profession":"painter"},{"name":"Nancy","profession":"guard"},{"name":"Olof","profession":"clerk"},{"name":"Paula","profession":"cop"},{"name":"Ryan","profession":"sleuth"},{"name":"Sofia","profession":"guard"},{"name":"Terry","profession":"sleuth"},{"name":"Vicky","profession":"farmer"},{"name":"Wally","profession":"mech"},{"name":"Xavi","profession":"mech"},{"name":"Zara","profession":"mech"}]'


def main():
    puzzle = Puzzle(input_data)

    # initial uncovered suspect
    puzzle.set_single_verdict("Frank", True)

    # first clue, from Frank - "Exactly 1 innocent in column A is neighboring Megan"
    column_a_suspects = puzzle.column(Column.A)
    megan_neighbors = puzzle.suspects["Megan"].neighbors
    relevant_suspects = column_a_suspects.intersection(megan_neighbors)

    print("Column A:")
    for suspect in column_a_suspects:
        print(suspect.name)
    print()

    print("Megan neighbors:")
    for suspect in megan_neighbors:
        print(suspect.name)
    print()

    print("Relevant suspects:")
    for suspect in relevant_suspects:
        print(suspect.name)
    print()

    relevant_suspect_refs = [
        suspect.is_innocent for suspect in relevant_suspects]
    puzzle.solver.add(AtLeast(*relevant_suspect_refs, 1))
    puzzle.solver.add(AtMost(*relevant_suspect_refs, 1))

    puzzle.solve_many()
    print()

    # second clue, from Keith - "There's an odd number of criminals to the left of Sofia"
    clue2_relevant = puzzle.get_suspects_relative_to_other_suspect(
        "Sofia", Direction.LEFT)
    print("Relevant suspects:")
    for suspect in clue2_relevant:
        print(suspect.name)
    print()

    clue2_relevant_refs = [suspect.is_innocent for suspect in clue2_relevant]

    puzzle.solver.add(Or(
        And(AtLeast(*clue2_relevant_refs, 1), AtMost(*clue2_relevant_refs, 1)),
        And(AtLeast(*clue2_relevant_refs, 3), AtMost(*clue2_relevant_refs, 3)),
        And(AtLeast(*clue2_relevant_refs, 5), AtMost(*clue2_relevant_refs, 5))
    ))

    puzzle.solve_many()
    print()

    # third clue, from Ryan - "There is only one innocent above Keith"
    clue3_relevant = puzzle.get_suspects_relative_to_other_suspect(
        "Keith", Direction.ABOVE)
    print("Relevant suspects:")
    for suspect in clue3_relevant:
        print(suspect.name)
    print()

    clue3_relevant_refs = [suspect.is_innocent for suspect in clue3_relevant]
    puzzle.solver.add(AtLeast(*clue3_relevant_refs, 1))
    puzzle.solver.add(AtMost(*clue3_relevant_refs, 1))
    puzzle.solve_many()
    print()

    # fourth clue, from Alex - "Both criminals below me are connected"


if __name__ == "__main__":
    main()
