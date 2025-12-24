from dataclasses import dataclass
from enum import Enum, IntEnum
import json
from typing import List, Optional, Set
from z3 import Solver, sat, Or, And, Not, If, Bool, BoolRef, AtLeast, AtMost, unsat, Sum

NUM_ROWS = 5
NUM_COLS = 4

word_to_int: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3
    # TODO - fill out as needed
}


class Verdict(IntEnum):
    INNOCENT = 1
    CRIMINAL = 2

    @classmethod
    def parse(cls, input: str) -> "Verdict":
        match input:
            case "innocent" | "innocents":
                return Verdict.INNOCENT
            case "criminal" | "criminals":
                return Verdict.CRIMINAL

        raise ValueError(f"{input} is not a valid verdict")


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

    # Fast __repr__ method to allow debugging without needing to calculate repr for Z3 values
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at {hex(id(self))}"


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

    def set_single_verdict(self, suspect_name: str, verdict: Verdict):
        suspect = self.suspects[suspect_name]
        self.unsolved_suspects.discard(suspect)

        if verdict == Verdict.INNOCENT:
            print(f'{suspect_name} is innocent')
            self.solver.add(suspect.is_innocent)
        elif verdict == Verdict.CRIMINAL:
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

    def handle_clue(self, clue: str, suspect_with_clue: str = ""):
        match clue.split():
            # TODO - does this need to have "is" | "are"?
            # TODO - version of this for rows
            case ["Exactly", num_suspects, ('innocent' | 'innocents' | 'criminal' | 'criminals') as verdict_str, "in", "column", column_name, "is", "neighboring", suspect_name]:
                column_suspects = self.column(Column.parse(column_name))
                neighbors = self.suspects[suspect_name].neighbors
                neighbor_subset = column_suspects & neighbors
                verdict = Verdict.parse(verdict_str)

                self.set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_suspects), verdict)

            # TODO - case for above/below as well as left/right
            case ["There's", "an", ("odd" | "even") as parity_str, "number", "of", ('innocent' | 'innocents' | 'criminal' | 'criminals') as verdict_str, "to", "the", ("left" | "right") as direction_str, "of", suspect_name]:
                direction = Direction(direction_str)
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbor_subset = self.get_suspects_relative_to_other_suspect(
                    suspect_name, direction)

                self.set_has_parity(neighbor_subset, parity, verdict)

            # TODO - case for to the left/right of as well as above/below
            case ["There", "is", "only", "one", ('innocent' | 'criminal') as verdict_str, ("above" | "below") as direction_str, suspect_name]:
                direction = Direction(direction_str)
                neighbor_subset = self.get_suspects_relative_to_other_suspect(
                    suspect_name, direction)
                verdict = Verdict.parse(verdict_str)

                self.set_has_exactly_n_of_verdict(neighbor_subset, 1, verdict)

            case [suspect_name, "is", "one", "of", num_suspects, ("innocents" | "criminal") as verdict_str, "in", "column", column_name]:
                neighbor_subset = self.column(Column.parse(column_name))
                verdict = Verdict.parse(verdict_str)

                self.set_single_verdict(suspect_name, verdict)
                self.set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_suspects), verdict)

            case [suspect_name, "is", "one", "of", num_suspects, ("innocents" | "criminals") as verdict_str, "in", "row", row]:
                neighbor_subset = self.row(int(row))
                verdict = Verdict.parse(verdict_str)

                self.set_single_verdict(suspect_name, verdict)
                self.set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_suspects), verdict)

            case [suspect_name, "is", "one", "of", num_suspects_str, "or", "more", ("innocents" | "criminals") as verdict_str, "on", "the", "edges"]:
                verdict = Verdict.parse(verdict_str)
                num_suspects = word_to_int[num_suspects_str]
                relevant_edge_suspects_count = count_suspects_with_verdict(
                    self.edges(), verdict)

                self.set_single_verdict(suspect_name, verdict)
                self.solver.add(relevant_edge_suspects_count >= num_suspects)

            case ["There", "are", "as", "many", ("innocent" | "criminal") as verdict_str1, profession1_plural, "as", "there", "are", ("innocent" | "criminal") as verdict_str2, profession2_plural] if verdict_str1 == verdict_str2:
                profession1 = profession1_plural.removesuffix("s")
                profession2 = profession2_plural.removesuffix("s")
                verdict = Verdict.parse(verdict_str1)

                profession1_count = count_suspects_with_verdict(
                    self.all_of_profession(profession1), verdict)
                profession2_count = count_suspects_with_verdict(
                    self.all_of_profession(profession2), verdict)

                self.solver.add(profession1_count == profession2_count)

            # TODO - double-check wording for cases where this refers to multiple suspects - is the "have" actually needed?
            case ["Exactly", num_suspects, profession_plural, ("has" | "have"), "a", ("innocent" | "criminal") as verdict_str, "directly", ("above" | "below") as direction_str, "them"]:
                verdict = Verdict.parse(verdict_str)
                profession = profession_plural.removesuffix("s")
                profession_members = self.all_of_profession(profession)
                direction = Direction(direction_str)
                profession_neighbors = [p.neighbor_in_direction(
                    direction) for p in profession_members]
                filtered_neighbors = [
                    n for n in profession_neighbors if n is not None]

                self.set_has_exactly_n_of_verdict(
                    set(filtered_neighbors), int(num_suspects), verdict)

            # TODO - what's the exact wording for "to the left of/to the right of"? does that case happen?
            case ["Exactly", num_neighbor_subset, "of", "the", num_neighbors, ("innocents" | "criminals") as verdict_str, "neighboring", central_suspect, "are", ("above" | "below") as direction_str, other_suspect]:
                # first part - central_suspect has num_neighbors with verdict
                central_suspect_neighbors = self.suspects[central_suspect].neighbors
                verdict = Verdict.parse(verdict_str)

                self.set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict)

                # second part - of those neighbors, num_neighbor_subset in direction_str relative to other_suspect, have verdict
                direction = Direction(direction_str)
                neighbor_subset = central_suspect_neighbors & self.get_suspects_relative_to_other_suspect(
                    other_suspect, direction)

                self.set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_neighbor_subset), verdict)

            # TODO - version of this for columns
            case ["An", ("odd" | "even") as parity_str, "number", "of", ('innocents' | 'criminals') as verdict_str, "in", "row", row, "neighbor", suspect_name]:
                neighbors = self.suspects[suspect_name].neighbors
                row_suspects = self.row(int(row))
                relevant_suspects = neighbors & row_suspects
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)

                self.set_has_parity(relevant_suspects, parity, verdict)

            # TODO - version of this for rows
            case ["Column", column_name, "has", "more", ("innocents" | "criminals") as verdict_str, "than", "any", "other", "column"]:
                self.column_has_most_of_verdict(
                    Column.parse(column_name), Verdict.parse(verdict_str))

            # TODO - version of this for columns
            case ["There", "are", "more", ("innocents" | "criminals") as more_verdict, "than", ("innocents" | "criminals") as less_verdict, "in", "row", row] if more_verdict != less_verdict:
                row_suspects = self.row(int(row))
                innocent_count = count_suspects_with_verdict(
                    row_suspects, Verdict.INNOCENT)
                criminal_count = count_suspects_with_verdict(
                    row_suspects, Verdict.CRIMINAL)

                match more_verdict, less_verdict:
                    case "innocents", "criminals":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminals", "innocents":
                        self.solver.add(criminal_count > innocent_count)

            case [more_suspect_name, "has", "more", ("innocent" | "criminal") as verdict_str, "neighbors", "than", less_suspect_name]:
                verdict = Verdict.parse(verdict_str)
                more_suspect_neighbors = self.suspects[more_suspect_name].neighbors
                more_suspect_neighbor_count = count_suspects_with_verdict(
                    more_suspect_neighbors, verdict)
                less_suspect_neighbors = self.suspects[less_suspect_name].neighbors
                less_suspect_neighbor_count = count_suspects_with_verdict(
                    less_suspect_neighbors, verdict)

                self.solver.add(more_suspect_neighbor_count >
                                less_suspect_neighbor_count)

            # TODO - version of this for "to the left/right"?
            case ["Both", ("innocent" | "criminals") as verdict_str, ("above" | "below") as direction_str, suspect1_name, "are", suspect2_name, "neighbors"]:
                # original suspect2_name has "'s" at the end, i.e. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there are exactly two innocents/criminals in direction_str relative to suspect1
                direction_suspects = self.get_suspects_relative_to_other_suspect(
                    suspect1_name, direction)
                self.set_has_exactly_n_of_verdict(
                    direction_suspects, 2, verdict)

                # second part - there are exactly two innocent/criminals in intersection of direction_suspects and neighbors of suspect2
                neighbor_suspects = self.suspects[suspect2_name].neighbors
                self.set_has_exactly_n_of_verdict(
                    direction_suspects & neighbor_suspects, 2, verdict)

            # TODO - version of this for "to the left/right"?
            # TODO - combine with above case? same logic, this is just talking about 1 suspect instead of 2
            case ["The", "only", ("innocent" | "criminal") as verdict_str, ("above" | "below") as direction_str, suspect1_name, "is", suspect2_name, "neighbor"]:
                # original suspect2_name has "'s" at the end, i.e. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there is exactly one innocents/criminal in direction_str relative to suspect1
                direction_suspects = self.get_suspects_relative_to_other_suspect(
                    suspect1_name, direction)
                self.set_has_exactly_n_of_verdict(
                    direction_suspects, 1, verdict)

                # second part - there is exactly one innocent/criminal in intersection of direction_suspects and neighbors of suspect2
                neighbor_suspects = self.suspects[suspect2_name].neighbors
                self.set_has_exactly_n_of_verdict(
                    direction_suspects & neighbor_suspects, 1, verdict)

            # TODO - version of this for "to the left/right"?
            case ["Both", ("innocents" | "criminals") as verdict_str, ("above" | "below") as direction_str, "me", "are", "connected"]:
                if suspect_with_clue == "":
                    raise ValueError(
                        "need a suspect name for a clue containing 'me'")

                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there are exactly two innocents/criminals in direction_str relative to suspect
                direction_suspects = self.get_suspects_relative_to_other_suspect(
                    suspect_with_clue, direction)
                self.set_has_exactly_n_of_verdict(
                    direction_suspects, 2, verdict)

                # second part - all innocents/criminals in direction_str are connected
                self.all_suspects_in_vertical_set_with_verdict_are_connected(
                    direction_suspects, verdict)

    def set_has_exactly_n_of_verdict(self, suspects: set[Suspect], num_of_verdict: int, verdict: Verdict):
        if verdict == Verdict.INNOCENT:
            refs = [suspect.is_innocent for suspect in suspects]
        elif verdict == Verdict.CRIMINAL:
            refs = [Not(suspect.is_innocent) for suspect in suspects]

        self.solver.add(AtLeast(*refs, num_of_verdict))
        self.solver.add(AtMost(*refs, num_of_verdict))

    def set_has_parity(self, suspects: set[Suspect], parity: Parity, verdict: Verdict):
        count = count_suspects_with_verdict(suspects, verdict)
        if parity == Parity.ODD:
            self.solver.add(count % 2 == 1)
        elif parity == Parity.EVEN:
            self.solver.add(count % 2 == 0)

    def column_has_most_of_verdict(self, column: Column, verdict: Verdict):
        column_count = count_suspects_with_verdict(
            self.column(column), verdict)

        for other_col in Column:
            if other_col != column:
                other_col_count = count_suspects_with_verdict(
                    self.column(other_col), verdict)
                self.solver.add(column_count > other_col_count)

    # checking for connection:
    # "all innocents are connected" => no criminals in set have innocents to both sides (or vice versa for criminals)
    # so to assert this, for each suspect in set, either
    # a.) suspect has given verdict
    # b.) suspect does not have given verdict AND does not have neighbors on both sides with given verdict
    # we need two methods, one horizontal, one vertical, to govern which neighbors are checked

    def all_suspects_in_horizontal_set_with_verdict_are_connected(self, suspects: set[Suspect], verdict: Verdict):
        for suspect in suspects:
            left_neighbor = suspect.neighbor_in_direction(Direction.LEFT)
            right_neighbor = suspect.neighbor_in_direction(Direction.RIGHT)

            # make sure we're only checking for neighbors within the set
            # this implicitly checks that left_neighbor/right_neighbor aren't None - `None in suspects` is false
            if left_neighbor in suspects and right_neighbor in suspects:
                if verdict == Verdict.INNOCENT:
                    self.solver.add(
                        Or(
                            suspect.is_innocent,
                            Not(
                                And(
                                    left_neighbor.is_innocent,
                                    right_neighbor.is_innocent
                                )
                            )
                        )
                    )
                elif verdict == Verdict.CRIMINAL:
                    self.solver.add(
                        Or(
                            Not(suspect.is_innocent),
                            Not(
                                And(
                                    Not(left_neighbor.is_innocent),
                                    Not(right_neighbor.is_innocent)
                                )
                            )
                        )
                    )

    def all_suspects_in_vertical_set_with_verdict_are_connected(self, suspects: set[Suspect], verdict: Verdict):
        for suspect in suspects:
            above_neighbor = suspect.neighbor_in_direction(Direction.ABOVE)
            below_neighbor = suspect.neighbor_in_direction(Direction.BELOW)

            # make sure we're only checking for neighbors within the set
            # this implicitly checks that above_neighbor/below_neighbor aren't None - `None in suspects` is false
            if above_neighbor in suspects and below_neighbor in suspects:
                if verdict == Verdict.INNOCENT:
                    self.solver.add(
                        Or(
                            suspect.is_innocent,
                            Not(
                                And(
                                    above_neighbor.is_innocent,
                                    below_neighbor.is_innocent
                                )
                            )
                        )
                    )
                elif verdict == Verdict.CRIMINAL:
                    self.solver.add(
                        Or(
                            Not(suspect.is_innocent),
                            Not(
                                And(
                                    Not(above_neighbor.is_innocent),
                                    Not(below_neighbor.is_innocent)
                                )
                            )
                        )
                    )


def sort_vertical_suspects(suspects: Set[Suspect]) -> List[Suspect]:
    # Check that all suspects are in the same column
    # create a set of all unique column values - should have exactly 1 element
    assert len({s.column for s in suspects}) == 1

    return sorted(suspects, key=lambda suspect: suspect.name)


def count_suspects_with_verdict(suspects: set[Suspect], verdict: Verdict):
    if verdict == Verdict.INNOCENT:
        return Sum([If(s.is_innocent, 1, 0) for s in suspects])
    elif verdict == Verdict.CRIMINAL:
        return Sum([If(s.is_innocent, 0, 1) for s in suspects])


# Get input data from browser console
# data from Puzzle Pack #1, puzzle 1
# https://cluesbysam.com/s/user/63f90e0e67bb92cd/pack-1/1/
input_data = '[{"name":"Alex","profession":"cook"},{"name":"Bonnie","profession":"painter"},{"name":"Chris","profession":"cook"},{"name":"Ellie","profession":"cop"},{"name":"Frank","profession":"farmer"},{"name":"Helen","profession":"cook"},{"name":"Isaac","profession":"guard"},{"name":"Julie","profession":"clerk"},{"name":"Keith","profession":"farmer"},{"name":"Megan","profession":"painter"},{"name":"Nancy","profession":"guard"},{"name":"Olof","profession":"clerk"},{"name":"Paula","profession":"cop"},{"name":"Ryan","profession":"sleuth"},{"name":"Sofia","profession":"guard"},{"name":"Terry","profession":"sleuth"},{"name":"Vicky","profession":"farmer"},{"name":"Wally","profession":"mech"},{"name":"Xavi","profession":"mech"},{"name":"Zara","profession":"mech"}]'


def main():
    puzzle = Puzzle(input_data)

    # initial uncovered suspect
    puzzle.set_single_verdict("Frank", Verdict.INNOCENT)

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

    # fourth clue, from Alex
    puzzle.handle_clue("Both criminals below me are connected", "Alex")
    puzzle.solve_many()
    print()

    # fifth clue, from Vicky
    puzzle.handle_clue("Xavi is one of 4 innocents in column C")
    puzzle.solve_many()
    print()

    # sixth clue, from Xavi
    puzzle.handle_clue(
        "There are as many criminal farmers as there are criminal guards")
    puzzle.solve_many()
    print()

    # seventh clue, from Chris
    puzzle.handle_clue("Exactly 1 guard has a criminal directly below them")
    puzzle.solve_many()
    print()

    # eighth clue, from Isaac
    puzzle.handle_clue(
        "Exactly 2 of the 3 criminals neighboring Megan are above Vicky")
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

    # tenth clue, from Wally
    puzzle.handle_clue("An odd number of innocents in row 1 neighbor Helen")
    puzzle.solve_many()
    print()

    # eleventh clue, from Bonnie
    puzzle.handle_clue("Column C has more innocents than any other column")
    puzzle.solve_many()
    print()

    # twelfth clue, from Megan
    puzzle.handle_clue("there are more criminals than innocents in row 1")
    puzzle.solve_many()
    print()

    # thirteenth clue, from Ellie
    puzzle.handle_clue("Chris has more criminal neighbors than Paula")
    puzzle.solve_many()
    print()

    # fourteenth clue, from Julie
    puzzle.handle_clue("Terry is one of two or more innocents on the edges")
    puzzle.solve_many()
    print()

    # fifteenth clue, from Terry
    puzzle.handle_clue("Both criminals above Zara are Isaac's neighbors")
    puzzle.solve_many()
    print()

    # sixteenth clue, from Olof
    puzzle.handle_clue("The only criminal below Julie is Terry's neighbor")
    puzzle.solve_many()
    print()

    # seventeenth and final clue, from Zara
    puzzle.handle_clue("Xavi has more innocent neighbors than Isaac")
    puzzle.solve_many()
    print()


if __name__ == "__main__":
    main()
