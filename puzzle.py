from dataclasses import dataclass
from typing import Optional, Set, TypedDict
from models import Column, Direction, Parity, Verdict
from z3 import Solver, Bool, BoolRef, Not, And, Or, AtLeast, AtMost, If, Sum, sat, unsat

NUM_ROWS = 5
NUM_COLS = 4

word_to_int: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3
    # TODO - fill out as needed
}


# TODO - potentially convert back to dataclass, let main driver handle parsing JSON into dataclasses
class RawSuspect(TypedDict):
    name: str
    profession: str


@dataclass
class PuzzleInput:
    suspects: list[RawSuspect]  # assumed to be sorted
    starting_suspect_name: str
    starting_suspect_verdict: Verdict


# all the data on a suspect that the Puzzle engine needs, including location, neighbors, and a Z3 BoolRef for tracking deduced verdict
@dataclass
class PuzzleSuspect:
    name: str
    profession: str
    row: int
    column: Column
    neighbors: Set['PuzzleSuspect']
    is_innocent: BoolRef  # Z3 expression - true iff suspect is innocent

    # Names are unique and immutable, so can be used for equality testing; can't use default hash because Sets aren't hashable
    def __hash__(self) -> int:
        return hash(self.name)

    def neighbor_in_direction(self, direction: Direction) -> Optional['PuzzleSuspect']:
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


@dataclass
class SuspectSolution:
    name: str
    verdict: Verdict


class Puzzle:
    suspects: dict[str, PuzzleSuspect]  # suspects by name
    solver: Solver

    # suspects whose verdict hasn't yet been determined
    unsolved_suspects: Set[PuzzleSuspect]

    def __init__(self, input_data: PuzzleInput):
        self.solver = Solver()
        self.suspects = {}
        self.unsolved_suspects = set()

        # Used as initial element of grid
        dummy_suspect = PuzzleSuspect(
            name="dummy",
            profession="none",
            neighbors=set(),
            is_innocent=Bool("dummy"),
            row=1,
            column=Column.A
        )

        # used for setting up suspects' neighbors
        grid = [[dummy_suspect for col in range(
            NUM_COLS)] for row in range(NUM_ROWS)]

        # set up suspects
        idx = 0
        for suspect_data in input_data.suspects:
            suspect = PuzzleSuspect(
                name=suspect_data["name"],
                profession=suspect_data["profession"],
                neighbors=set(),
                is_innocent=Bool(suspect_data["name"]),
                row=(idx // NUM_COLS) + 1,  # convert to 1-based index
                column=Column(idx % NUM_COLS),
            )
            self.suspects[suspect.name] = suspect
            if suspect.name != input_data.starting_suspect_name:
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

        # set up initial suspect
        self.set_single_verdict(
            input_data.starting_suspect_name, input_data.starting_suspect_verdict)

    def is_solved(self) -> bool:
        return len(self.unsolved_suspects) == 0

    # methods to get specific sets of suspects

    def row(self, row_num: int) -> Set[PuzzleSuspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.row == row_num])

    def column(self, column_name: Column) -> Set[PuzzleSuspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.column == column_name])

    def get_suspects_relative_to_other_suspect(self, root_suspect_name: str, direction: Direction) -> Set[PuzzleSuspect]:
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

    def all_of_profession(self, profession_name: str) -> Set[PuzzleSuspect]:
        return set([suspect for suspect in self.suspects.values() if suspect.profession == profession_name])

    def edges(self) -> Set[PuzzleSuspect]:
        return self.row(1) | self.row(5) | self.column(Column.A) | self.column(Column.D)

    # methods to add constraints to solver

    def set_has_exactly_n_of_verdict(self, suspects: set[PuzzleSuspect], num_of_verdict: int, verdict: Verdict):
        if verdict == Verdict.INNOCENT:
            refs = [suspect.is_innocent for suspect in suspects]
        elif verdict == Verdict.CRIMINAL:
            refs = [Not(suspect.is_innocent) for suspect in suspects]

        self.solver.add(AtLeast(*refs, num_of_verdict))
        self.solver.add(AtMost(*refs, num_of_verdict))

    def set_single_verdict(self, suspect_name: str, verdict: Verdict):
        suspect = self.suspects[suspect_name]
        self.set_has_exactly_n_of_verdict(set([suspect]), 1, verdict)

    def set_has_parity(self, suspects: set[PuzzleSuspect], parity: Parity, verdict: Verdict):
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

    def all_suspects_in_horizontal_set_with_verdict_are_connected(self, suspects: set[PuzzleSuspect], verdict: Verdict):
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

    def all_suspects_in_vertical_set_with_verdict_are_connected(self, suspects: set[PuzzleSuspect], verdict: Verdict):
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

    # methods for solving puzzle

    def solve_one(self) -> Optional[SuspectSolution]:
        """
        Try to deduce a verdict; returns the first deduced status, if possible, otherwise returns None
        """
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
                # print(f'{suspect.name} is criminal')
                return SuspectSolution(name=suspect.name, verdict=Verdict.CRIMINAL)
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
                # print(f'{suspect.name} is innocent')
                return SuspectSolution(name=suspect.name, verdict=Verdict.INNOCENT)
            else:
                # reset to previous backtracking point
                self.solver.pop()

        return None

        solutions_found = []
        while True:
            match self.solve_one():
                case None:
                    break
                case solution:
                    solutions_found.append(solution)
        return solutions_found

    def add_constraints_from_clue(self, clue: str, suspect_with_clue: str = ""):
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

            # TODO - version of this for columns?
            case ["All", ("innocents" | "criminals") as verdict_str, "in", "row", row, "are", "connected"]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self.row(int(row))

                self.all_suspects_in_horizontal_set_with_verdict_are_connected(
                    row_suspects, verdict)

    # primary entrypoint

    def add_clue(self, clue: str, suspect_with_clue: str) -> list[SuspectSolution]:
        """
        Add a new clue to the puzzle, including the source of the clue. Returns a list of newly deduced solutions.
        """
        new_solutions_found = []
        self.add_constraints_from_clue(clue, suspect_with_clue)

        while True:
            match self.solve_one():
                case None:
                    break
                case solution:
                    new_solutions_found.append(solution)

        return new_solutions_found


def count_suspects_with_verdict(suspects: set[PuzzleSuspect], verdict: Verdict):
    if verdict == Verdict.INNOCENT:
        return Sum([If(s.is_innocent, 1, 0) for s in suspects])
    elif verdict == Verdict.CRIMINAL:
        return Sum([If(s.is_innocent, 0, 1) for s in suspects])
