from dataclasses import dataclass
from enum import Enum, IntEnum
import json
from typing import List, Optional, Set, Tuple
from z3 import Int, Solver, sat, Or, And, Not, If, Bool, Bools, ArithRef, BoolRef, BoolVector, AtLeast, AtMost, sat, unsat, Sum, ForAll, Exists

NUM_ROWS = 5
NUM_COLS = 4


class Verdict(Enum):
    INNOCENT = 1
    CRIMINAL = 2
    UNKNOWN = 3


# IntEnum so we can set column numerically when setting up the grid of suspects
class Column(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3

    @classmethod
    def parse(cls, input: str) -> "Column":
        match input:
            case "A" | "a":
                return Column.A
            case "B" | "b":
                return Column.B
            case "C" | "c":
                return Column.C
            case "D" | "d":
                return Column.D

        raise ValueError(f"{input} is not a valid column")


class Direction(Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"


class Parity(Enum):
    ODD = "odd"
    EVEN = "even"


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

    def neighbor_in_direction(self, direction: Direction) -> Optional['Suspect']:
        for neighbor in self.neighbors:
            match direction:
                case Direction.ABOVE:
                    if neighbor.column == self.column and neighbor.row == self.row - 1:
                        return neighbor
                case Direction.BELOW:
                    if neighbor.column == self.column and neighbor.row == self.row + 1:
                        return neighbor
                case Direction.LEFT:
                    if neighbor.row == self.row and neighbor.column == self.column - 1:
                        return neighbor
                case Direction.RIGHT:
                    if neighbor.row == self.row and neighbor.column == self.column + 1:
                        return neighbor
        return None


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
            print(f'{suspect_name} is innocent')
            self.solver.add(suspect.is_innocent)
        else:
            print(f'{suspect_name} is criminal')
            self.solver.add(Not(suspect.is_innocent))

    def all_of_profession(self, profession_name: str) -> Set[Suspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.profession == profession_name])

    def edges(self) -> Set[Suspect]:
        return self.row(1) | self.row(5) | self.column(Column.A) | self.column(Column.D)

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
    # TODO - does this need to return a value?
    def solve_many(self) -> bool:
        can_make_progress = True
        progress_made = False
        while can_make_progress:
            can_make_progress = self.solve_one()
            if can_make_progress:
                progress_made = True
        return progress_made

    def handle_clue(self, clue: str):
        match clue.split():
            case ["Exactly", num_suspects, ('innocent' | 'innocents' | 'criminal' | 'criminals') as verdict, "in", "column", column, "is", "neighboring", suspect_name]:
                column_suspects = self.column(Column.parse(column))
                neighbors = self.suspects[suspect_name].neighbors
                relevant_suspects = column_suspects & neighbors

                if verdict == 'innocent' or 'innocents':
                    self.set_has_exactly_n_innocents(
                        relevant_suspects, int(num_suspects))
                elif verdict == 'criminal' or 'criminals':
                    self.set_has_exactly_n_criminals(
                        relevant_suspects, int(num_suspects))
            case ["There's", "an", ("odd" | "even") as parity_str, "number", "of", ('innocent' | 'innocents' | 'criminal' | 'criminals') as verdict, "to", "the", ("left" | "right") as direction_name, "of", suspect_name]:
                direction = Direction(direction_name)
                parity = Parity(parity_str)
                relevant_suspects = self.get_suspects_relative_to_other_suspect(
                    suspect_name, direction)

                if verdict == 'innocent' or 'innocents':
                    self.set_has_parity(relevant_suspects, parity, True)
                elif verdict == 'criminal' or 'criminals':
                    self.set_has_parity(relevant_suspects, parity, False)
            case ["There", "is", "only", "one", ('innocent' | 'criminal') as verdict, ("above" | "below") as direction_name, suspect_name]:
                direction = Direction(direction_name)
                relevant_suspects = self.get_suspects_relative_to_other_suspect(
                    suspect_name, direction)

                if verdict == "innocent":
                    self.set_has_exactly_n_innocents(relevant_suspects, 1)
                elif verdict == "criminal":
                    self.set_has_exactly_n_criminals(relevant_suspects, 1)

    def set_has_exactly_n_innocents(self, suspects: set[Suspect], num_innocents: int):
        refs = [suspect.is_innocent for suspect in suspects]
        self.solver.add(AtLeast(*refs, num_innocents))
        self.solver.add(AtMost(*refs, num_innocents))

    def set_has_exactly_n_criminals(self, suspects: set[Suspect], num_criminals: int):
        refs = [Not(suspect.is_innocent) for suspect in suspects]
        self.solver.add(AtLeast(*refs, num_criminals))
        self.solver.add(AtMost(*refs, num_criminals))

    def set_has_parity(self, suspects: set[Suspect], parity: Parity, is_innocent: bool):
        match is_innocent, parity:
            case True, Parity.ODD:
                self.solver.add(count_innocents(suspects) % 2 == 1)
            case True, Parity.EVEN:
                self.solver.add(count_innocents(suspects) % 2 == 0)
            case False, Parity.ODD:
                self.solver.add(count_criminals(suspects) % 2 == 1)
            case False, Parity.EVEN:
                self.solver.add(count_criminals(suspects) % 2 == 0)


def sort_vertical_suspects(suspects: Set[Suspect]) -> List[Suspect]:
    # Check that all suspects are in the same column
    # create a set of all unique column values - should have exactly 1 element
    assert len({s.column for s in suspects}) == 1

    return sorted(suspects, key=lambda suspect: suspect.name)


def count_innocents(suspects: set[Suspect]):
    return Sum([If(s.is_innocent, 1, 0) for s in suspects])


def count_criminals(suspects: set[Suspect]):
    return Sum([If(s.is_innocent, 0, 1) for s in suspects])


# Get input data from browser console
# data from Puzzle Pack #1, puzzle 1
# https://cluesbysam.com/s/user/63f90e0e67bb92cd/pack-1/1/
input_data = '[{"name":"Alex","profession":"cook"},{"name":"Bonnie","profession":"painter"},{"name":"Chris","profession":"cook"},{"name":"Ellie","profession":"cop"},{"name":"Frank","profession":"farmer"},{"name":"Helen","profession":"cook"},{"name":"Isaac","profession":"guard"},{"name":"Julie","profession":"clerk"},{"name":"Keith","profession":"farmer"},{"name":"Megan","profession":"painter"},{"name":"Nancy","profession":"guard"},{"name":"Olof","profession":"clerk"},{"name":"Paula","profession":"cop"},{"name":"Ryan","profession":"sleuth"},{"name":"Sofia","profession":"guard"},{"name":"Terry","profession":"sleuth"},{"name":"Vicky","profession":"farmer"},{"name":"Wally","profession":"mech"},{"name":"Xavi","profession":"mech"},{"name":"Zara","profession":"mech"}]'


def main():
    puzzle = Puzzle(input_data)

    # initial uncovered suspect
    puzzle.set_single_verdict("Frank", True)

    # first clue, from Frank
    puzzle.handle_clue("Exactly 1 innocent in column A is neighboring Megan")
    puzzle.solve_many()
    print()

    # second clue, from Keith
    puzzle.handle_clue(
        "There's an odd number of criminals to the left of Sofia")
    puzzle.solve_many()
    print()

    # third clue, from Ryan
    puzzle.handle_clue("There is only one innocent above Keith")
    puzzle.solve_many()
    print()

    # fourth clue, from Alex - "Both criminals below me are connected"

    # "Both criminals" = exactly 2 criminals below
    clue4_below = puzzle.get_suspects_relative_to_other_suspect(
        "Alex", Direction.BELOW)
    clue4_below_refs = [Not(suspect.is_innocent) for suspect in clue4_below]
    puzzle.solver.add(AtLeast(*clue4_below_refs, 2))
    puzzle.solver.add(AtMost(*clue4_below_refs, 2))

    # "Are connected"
    # TODO - hardcodes 2 connected out of length 4 - generalize
    clue4_below_sorted = sort_vertical_suspects(clue4_below)
    assert len(clue4_below_sorted) == 4
    puzzle.solver.add(Or(
        And(
            Not(clue4_below_sorted[0].is_innocent),
            Not(clue4_below_sorted[1].is_innocent),
            clue4_below_sorted[2].is_innocent,
            clue4_below_sorted[3].is_innocent
        ),
        And(
            clue4_below_sorted[0].is_innocent,
            Not(clue4_below_sorted[1].is_innocent),
            Not(clue4_below_sorted[2].is_innocent),
            clue4_below_sorted[3].is_innocent
        ),
        And(
            clue4_below_sorted[0].is_innocent,
            clue4_below_sorted[1].is_innocent,
            Not(clue4_below_sorted[2].is_innocent),
            Not(clue4_below_sorted[3].is_innocent)
        ),
    ))

    puzzle.solve_many()
    print()

    # fifth clue, from Vicky - "Xavi is one of 4 innocents in column C"
    puzzle.set_single_verdict("Xavi", True)
    column3_refs = [suspect.is_innocent for suspect in puzzle.column(Column.C)]
    puzzle.solver.add(AtLeast(*column3_refs, 4))
    puzzle.solver.add(AtMost(*column3_refs, 4))

    puzzle.solve_many()
    print()

    # sixth clue, from Xavi - "There are as many criminal farmers as there are criminal guards"
    farmers = puzzle.all_of_profession("farmer")
    guards = puzzle.all_of_profession("guard")
    criminal_farmer_count = Sum([If(f.is_innocent, 0, 1) for f in farmers])
    criminal_guard_count = Sum([If(g.is_innocent, 0, 1) for g in guards])
    puzzle.solver.add(criminal_farmer_count == criminal_guard_count)

    puzzle.solve_many()
    print()

    # seventh clue, from Chris - "Exactly 1 guard has a criminal directly below them"
    guard_neighbors = [g.neighbor_in_direction(
        Direction.BELOW) for g in guards]
    filtered_guard_neighbor_refs = [
        Not(n.is_innocent) for n in guard_neighbors if n is not None]
    puzzle.solver.add(AtLeast(*filtered_guard_neighbor_refs, 1))
    puzzle.solver.add(AtMost(*filtered_guard_neighbor_refs, 1))

    puzzle.solve_many()
    print()

    # eighth clue, from Isaac - "Exactly 2 of the 3 criminals neighboring Megan are above Vicky"

    # Megan has 3 criminal neighbors
    megan_neighbor_refs = [Not(n.is_innocent)
                           for n in puzzle.suspects["Megan"].neighbors]
    puzzle.solver.add(AtLeast(*megan_neighbor_refs, 3))
    puzzle.solver.add(AtMost(*megan_neighbor_refs, 3))

    # 2 criminals above Vicky, neighboring Megan, are criminal
    above_vicky = puzzle.get_suspects_relative_to_other_suspect(
        "Vicky", Direction.ABOVE)
    clue8_part2_refs = [Not(s.is_innocent) for s in above_vicky.intersection(
        puzzle.suspects["Megan"].neighbors)]
    puzzle.solver.add(AtLeast(*clue8_part2_refs, 2))
    puzzle.solver.add(AtMost(*clue8_part2_refs, 2))

    puzzle.solve_many()
    print()

    # ninth clue, from Helen - "All innocents in row 5 are connected"

    # "All innocents are connected" is false iff some criminal in the row has innocents on both left and right
    constraint_list = []
    for suspect in puzzle.row(5):
        left_neighbor = suspect.neighbor_in_direction(Direction.LEFT)
        right_neighbor = suspect.neighbor_in_direction(Direction.RIGHT)
        if left_neighbor is not None and right_neighbor is not None:
            # this is true iff innocents in the row are NOT connected
            counterexample_constraint = And(
                Not(suspect.is_innocent), left_neighbor.is_innocent, right_neighbor.is_innocent)

            constraint_list.append(counterexample_constraint)
    # the counterexample is not true for any suspect in the row
    puzzle.solver.add(Not(Or(*constraint_list)))

    puzzle.solve_many()
    print()

    # tenth clue, from Wally - "An odd number of innocents in row 1 neighbor Helen"
    clue10_refs = [s.is_innocent for s in puzzle.row(
        1).intersection(puzzle.suspects["Helen"].neighbors)]
    puzzle.solver.add(Or(
        And(AtLeast(*clue10_refs, 1), AtMost(*clue10_refs, 1)),
        And(AtLeast(*clue10_refs, 3), AtMost(*clue10_refs, 3)),
        And(AtLeast(*clue10_refs, 5), AtMost(*clue10_refs, 5))
    ))

    puzzle.solve_many()
    print()

    # eleventh clue, from Bonnie - "Column C has more innocents than any other column"

    colA_count = Sum([If(s.is_innocent, 1, 0)
                     for s in puzzle.column(Column.A)])
    colB_count = Sum([If(s.is_innocent, 1, 0)
                     for s in puzzle.column(Column.B)])
    colC_count = Sum([If(s.is_innocent, 1, 0)
                     for s in puzzle.column(Column.C)])
    colD_count = Sum([If(s.is_innocent, 1, 0)
                     for s in puzzle.column(Column.D)])

    puzzle.solver.add(colC_count > colA_count)
    puzzle.solver.add(colC_count > colB_count)
    puzzle.solver.add(colC_count > colD_count)

    puzzle.solve_many()
    print()

    # twelfth clue, from Megan - "there are more criminals than innocents in row 1"
    row1_innocent_count = Sum([If(s.is_innocent, 1, 0) for s in puzzle.row(1)])
    row1_criminal_count = Sum([If(Not(s.is_innocent), 1, 0)
                              for s in puzzle.row(1)])
    puzzle.solver.add(row1_criminal_count > row1_innocent_count)

    puzzle.solve_many()
    print()

    # thirteenth clue, from Ellie - "Chris has more criminal neighbors than Paula"
    chris_criminal_count = Sum([If(Not(s.is_innocent), 1, 0)
                               for s in puzzle.suspects["Chris"].neighbors])
    paula_criminal_count = Sum([If(Not(s.is_innocent), 1, 0)
                               for s in puzzle.suspects["Paula"].neighbors])
    puzzle.solver.add(chris_criminal_count > paula_criminal_count)

    puzzle.solve_many()
    print()

    # fourteenth clue, from Julie - "Terry is one of two or more innocents on the edges"
    puzzle.set_single_verdict("Terry", True)

    edge_refs = [s.is_innocent for s in puzzle.edges()]
    puzzle.solver.add(AtLeast(*edge_refs, 2))

    puzzle.solve_many()
    print()

    # fifteenth clue, from Terry - "Both criminals above Zara are Isaac's neighbors"

    # exactly 2 criminals above Zara
    above_zara_refs = [Not(s.is_innocent) for s in puzzle.get_suspects_relative_to_other_suspect(
        "Zara", Direction.ABOVE)]
    puzzle.solver.add(AtLeast(*above_zara_refs, 2))
    puzzle.solver.add(AtMost(*above_zara_refs, 2))

    # exactly 2 criminals in intersection of "above Zara" and Isaac's neighbors
    clue15_part2_refs = [Not(s.is_innocent) for s in puzzle.get_suspects_relative_to_other_suspect(
        "Zara", Direction.ABOVE).intersection(puzzle.suspects["Isaac"].neighbors)]
    puzzle.solver.add(AtLeast(*clue15_part2_refs, 2))
    puzzle.solver.add(AtMost(*clue15_part2_refs, 2))

    puzzle.solve_many()
    print()

    # sixteenth clue, from Olof - "The only criminal below Julie is Terry's neighbor"
    below_julie = puzzle.get_suspects_relative_to_other_suspect(
        "Julie", Direction.BELOW)

    # exactly 1 criminal below Julie
    clue16_part1_refs = [Not(s.is_innocent) for s in below_julie]
    puzzle.solver.add(AtLeast(*clue16_part1_refs, 1))
    puzzle.solver.add(AtMost(*clue16_part1_refs, 1))

    puzzle.solve_many()
    print()

    # seventeenth and final clue, from Zara - "Xavi has more innocent neighbors than Isaac"
    xavi_innocent_count = Sum([If(s.is_innocent, 1, 0)
                               for s in puzzle.suspects["Xavi"].neighbors])
    isaac_innocent_count = Sum([If(s.is_innocent, 1, 0)
                               for s in puzzle.suspects["Isaac"].neighbors])
    puzzle.solver.add(xavi_innocent_count > isaac_innocent_count)

    puzzle.solve_many()
    print()


if __name__ == "__main__":
    main()
