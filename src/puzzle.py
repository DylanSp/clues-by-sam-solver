from dataclasses import dataclass
from typing import Optional, Set
from z3 import Solver, Bool, BoolRef, Not, And, Or, AtLeast, AtMost, If, Sum, sat, unsat

from models import Column, Direction, Parity, Verdict


NUM_ROWS = 5
NUM_COLS = 4

word_to_int: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    # TODO - fill out as needed
}


@dataclass
class RawSuspect:
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
    neighbors: Set["PuzzleSuspect"]
    is_innocent: BoolRef  # Z3 expression - true iff suspect is innocent

    # Names are unique and immutable, so can be used for equality testing; can't use default hash because Sets aren't hashable
    def __hash__(self) -> int:
        return hash(self.name)

    def neighbor_in_direction(self, direction: Direction) -> Optional["PuzzleSuspect"]:
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
            column=Column.A,
        )

        # used for setting up suspects' neighbors
        grid = [[dummy_suspect for col in range(NUM_COLS)] for row in range(NUM_ROWS)]

        # set up suspects
        idx = 0
        for suspect_data in input_data.suspects:
            suspect = PuzzleSuspect(
                name=suspect_data.name,
                profession=suspect_data.profession,
                neighbors=set(),
                is_innocent=Bool(suspect_data.name),
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
        self._set_single_verdict(
            input_data.starting_suspect_name, input_data.starting_suspect_verdict
        )

    def is_solved(self) -> bool:
        return len(self.unsolved_suspects) == 0

    # methods to get specific sets of suspects

    def _row(self, row_num: int) -> Set[PuzzleSuspect]:
        return set(
            [suspect for suspect in self.suspects.values() if suspect.row == row_num]
        )

    def _column(self, column_name: Column) -> Set[PuzzleSuspect]:
        return set(
            [
                suspect
                for suspect in self.suspects.values()
                if suspect.column == column_name
            ]
        )

    def _get_suspects_relative_to_other_suspect(
        self, root_suspect_name: str, direction: Direction
    ) -> Set[PuzzleSuspect]:
        root_suspect = self.suspects[root_suspect_name]
        match direction:
            case Direction.ABOVE:
                return set(
                    [
                        suspect
                        for suspect in self.suspects.values()
                        if suspect.column == root_suspect.column
                        and suspect.row < root_suspect.row
                    ]
                )
            case Direction.BELOW:
                return set(
                    [
                        suspect
                        for suspect in self.suspects.values()
                        if suspect.column == root_suspect.column
                        and suspect.row > root_suspect.row
                    ]
                )
            case Direction.LEFT:
                return set(
                    [
                        suspect
                        for suspect in self.suspects.values()
                        if suspect.row == root_suspect.row
                        and suspect.column < root_suspect.column
                    ]
                )
            case Direction.RIGHT:
                return set(
                    [
                        suspect
                        for suspect in self.suspects.values()
                        if suspect.row == root_suspect.row
                        and suspect.column > root_suspect.column
                    ]
                )

    def _all_profession_types(self) -> Set[str]:
        return set([suspect.profession for suspect in self.suspects.values()])

    def _all_of_profession(self, profession_name: str) -> Set[PuzzleSuspect]:
        return set(
            [
                suspect
                for suspect in self.suspects.values()
                if suspect.profession == profession_name
            ]
        )

    def _edges(self) -> Set[PuzzleSuspect]:
        return (
            self._row(1)
            | self._row(5)
            | self._column(Column.A)
            | self._column(Column.D)
        )

    def _corners(self) -> Set[PuzzleSuspect]:
        top_left = self._row(1) & self._column(Column.A)
        top_right = self._row(1) & self._column(Column.D)
        bottom_left = self._row(5) & self._column(Column.A)
        bottom_right = self._row(5) & self._column(Column.D)
        return top_left | top_right | bottom_left | bottom_right

    def _between_suspects(
        self, suspect1_name: str, suspect2_name: str
    ) -> Set[PuzzleSuspect]:
        suspect1 = self.suspects[suspect1_name]
        suspect2 = self.suspects[suspect2_name]

        if suspect1.row == suspect2.row:
            left_suspect = min(suspect1, suspect2, key=lambda s: s.column)
            right_suspect = max(suspect1, suspect2, key=lambda s: s.column)

            return self._get_suspects_relative_to_other_suspect(
                left_suspect.name, Direction.RIGHT
            ) & self._get_suspects_relative_to_other_suspect(
                right_suspect.name, Direction.LEFT
            )
        elif suspect1.column == suspect2.column:
            top_suspect = min(suspect1, suspect2, key=lambda s: s.row)
            bottom_suspect = max(suspect1, suspect2, key=lambda s: s.row)

            return self._get_suspects_relative_to_other_suspect(
                top_suspect.name, Direction.BELOW
            ) & self._get_suspects_relative_to_other_suspect(
                bottom_suspect.name, Direction.ABOVE
            )
        else:
            raise ValueError(
                f"{suspect1_name} and {suspect2_name} are not in the same row or column; cannot check for suspects between them"
            )

    # methods to add constraints to solver

    def _set_has_exactly_n_of_verdict(
        self, suspects: set[PuzzleSuspect], num_of_verdict: int, verdict: Verdict
    ):
        if verdict == Verdict.INNOCENT:
            refs = [suspect.is_innocent for suspect in suspects]
        elif verdict == Verdict.CRIMINAL:
            refs = [Not(suspect.is_innocent) for suspect in suspects]

        self.solver.add(AtLeast(*refs, num_of_verdict))
        self.solver.add(AtMost(*refs, num_of_verdict))

    def _set_single_verdict(self, suspect_name: str, verdict: Verdict):
        """
        Convenience method for handling single suspects
        """
        suspect = self.suspects[suspect_name]
        self._set_has_exactly_n_of_verdict(set([suspect]), 1, verdict)

    def _set_has_parity(
        self, suspects: set[PuzzleSuspect], parity: Parity, verdict: Verdict
    ):
        count = count_suspects_with_verdict(suspects, verdict)
        if parity == Parity.ODD:
            self.solver.add(count % 2 == 1)
        elif parity == Parity.EVEN:
            self.solver.add(count % 2 == 0)

    def _column_has_most_of_verdict(self, column: Column, verdict: Verdict):
        column_count = count_suspects_with_verdict(self._column(column), verdict)

        for other_col in Column:
            if other_col != column:
                other_col_count = count_suspects_with_verdict(
                    self._column(other_col), verdict
                )
                self.solver.add(column_count > other_col_count)

    def _row_has_most_of_verdict(self, row: int, verdict: Verdict):
        row_count = count_suspects_with_verdict(self._row(row), verdict)

        for other_row in range(1, 6):
            if other_row != row:
                other_row_count = count_suspects_with_verdict(
                    self._row(other_row), verdict
                )
                self.solver.add(row_count > other_row_count)

    def _suspect_has_most_neighbors_of_verdict(
        self, suspect_name: str, verdict: Verdict
    ):
        suspect_neighbor_count = count_suspects_with_verdict(
            self.suspects[suspect_name].neighbors, verdict
        )

        for other_suspect in self.suspects.values():
            if other_suspect.name != suspect_name:
                other_neighbor_count = count_suspects_with_verdict(
                    other_suspect.neighbors, verdict
                )
                self.solver.add(suspect_neighbor_count > other_neighbor_count)

    def _profession_has_most_of_verdict(self, profession_name: str, verdict: Verdict):
        profession_count = count_suspects_with_verdict(
            self._all_of_profession(profession_name), verdict
        )

        for other_profession in self._all_profession_types():
            if other_profession != profession_name:
                other_profession_count = count_suspects_with_verdict(
                    self._all_of_profession(other_profession), verdict
                )
                self.solver.add(profession_count > other_profession_count)

    # checking for connection:
    # "all innocents are connected" => no criminals in set have innocents both somewhere on their left and somewhere on their right (or above/below for vertical sets)
    # so to assert this, for each suspect in set, either
    # a.) suspect has given verdict
    # b.) suspect does not have given verdict AND not (there's a suspect with given verdict somewhere on their left and a suspect with given verdict somewhere on their right)
    # for suspects on ends of the set being checked, this vacuously holds - they don't have any suspects on one side
    # we need two methods, one horizontal, one vertical, to govern which directions are checked

    def _all_suspects_in_horizontal_set_with_verdict_are_connected(
        self, suspects: set[PuzzleSuspect], verdict: Verdict
    ):
        for suspect in suspects:
            # intersect with suspects so we only checking for connectedness within the set
            left_suspects = suspects & self._get_suspects_relative_to_other_suspect(
                suspect.name, Direction.LEFT
            )
            right_suspects = suspects & self._get_suspects_relative_to_other_suspect(
                suspect.name, Direction.RIGHT
            )

            does_relevant_suspect_to_left_exist = (
                count_suspects_with_verdict(left_suspects, verdict) > 0
            )
            does_relevant_suspect_to_right_exist = (
                count_suspects_with_verdict(right_suspects, verdict) > 0
            )

            if verdict == Verdict.INNOCENT:
                self.solver.add(
                    Or(
                        suspect.is_innocent,
                        Not(
                            And(
                                does_relevant_suspect_to_left_exist,
                                does_relevant_suspect_to_right_exist,
                            )
                        ),
                    )
                )
            elif verdict == Verdict.CRIMINAL:
                self.solver.add(
                    Or(
                        Not(suspect.is_innocent),
                        Not(
                            And(
                                does_relevant_suspect_to_left_exist,
                                does_relevant_suspect_to_right_exist,
                            )
                        ),
                    )
                )

    def _all_suspects_in_vertical_set_with_verdict_are_connected(
        self, suspects: set[PuzzleSuspect], verdict: Verdict
    ):
        for suspect in suspects:
            # intersect with suspects so we only checking for connectedness within the set
            above_suspects = suspects & self._get_suspects_relative_to_other_suspect(
                suspect.name, Direction.ABOVE
            )
            below_suspects = suspects & self._get_suspects_relative_to_other_suspect(
                suspect.name, Direction.BELOW
            )

            does_relevant_suspect_above_exist = (
                count_suspects_with_verdict(above_suspects, verdict) > 0
            )
            does_relevant_suspect_below_exist = (
                count_suspects_with_verdict(below_suspects, verdict) > 0
            )

            if verdict == Verdict.INNOCENT:
                self.solver.add(
                    Or(
                        suspect.is_innocent,
                        Not(
                            And(
                                does_relevant_suspect_above_exist,
                                does_relevant_suspect_below_exist,
                            )
                        ),
                    )
                )
            elif verdict == Verdict.CRIMINAL:
                self.solver.add(
                    Or(
                        Not(suspect.is_innocent),
                        Not(
                            And(
                                does_relevant_suspect_above_exist,
                                does_relevant_suspect_below_exist,
                            )
                        ),
                    )
                )

    # methods for solving puzzle

    def _solve_one(self) -> Optional[SuspectSolution]:
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

    def _add_constraints_from_clue(self, clue: str, suspect_with_clue: str = ""):
        match clue.split():
            # TODO - does this need to have "is" | "are"? (if it does, probably add "innocents" | "criminals" to verdict_str)
            # TODO - version of this for rows
            case [
                "Exactly",
                num_suspects,
                ("innocent" | "criminal") as verdict_str,
                "in",
                "column",
                column,
                "is",
                "neighboring",
                suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                column_suspects = self._column(Column.parse(column))
                neighbors = self.suspects[suspect_name].neighbors

                self._set_has_exactly_n_of_verdict(
                    column_suspects & neighbors, int(num_suspects), verdict
                )

            # TODO - version of this for columns
            # TODO - merge with above? same logic, just with suspect_with_clue instead of suspect_name
            case [
                "Exactly",
                num_suspects,
                ("innocent" | "criminal") as verdict_str,
                "in",
                "row",
                row,
                "is",
                "neighboring",
                "me",
            ]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))
                neighbors = self.suspects[suspect_with_clue].neighbors

                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbors, int(num_suspects), verdict
                )

            # TODO - does this need to have "is" | "are"? (if it does, probably add "innocents" | "criminals" to verdict_str)
            case [
                "Exactly",
                num_suspects,
                ("innocent" | "criminal") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect_name,
                "is",
                "neighboring",
                "me",
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )
                neighbors = self.suspects[suspect_with_clue].neighbors

                self._set_has_exactly_n_of_verdict(
                    direction_suspects & neighbors, int(num_suspects), verdict
                )

            case [
                "Exactly",
                num_suspects,
                ("innocent" | "criminal") as verdict_str,
                ("above" | "below") as direction_str,
                suspect1_name,
                "is",
                "neighboring",
                suspect2_name,
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect1_name, direction
                )
                neighbors = self.suspects[suspect2_name].neighbors

                self._set_has_exactly_n_of_verdict(
                    direction_suspects & neighbors, int(num_suspects), verdict
                )

            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocent" | "criminal") as verdict_str,
                profession_plural,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                profession = profession_plural.removesuffix("s")

                self._set_has_parity(
                    self._all_of_profession(profession), parity, verdict
                )

            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                suspect_name,
            ]:
                direction = Direction(direction_str)
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_parity(suspects_in_direction, parity, verdict)

            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect_name,
            ]:
                direction = Direction(direction_str)
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_parity(suspects_in_direction, parity, verdict)

            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                suspect_name,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors

                self._set_has_parity(neighbors, parity, verdict)

            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "between",
                suspect1_name,
                "and",
                suspect2_name,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                suspects_between = self._between_suspects(suspect1_name, suspect2_name)

                self._set_has_parity(suspects_between, parity, verdict)

            case [
                "An",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "on",
                "the",
                "edges",
                "neighbor",
                suspect_name,
            ]:
                if suspect_name == "me":
                    suspect_name = suspect_with_clue

                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors
                edges = self._edges()

                self._set_has_parity(neighbors & edges, parity, verdict)

            # TODO - version for "to the left/right of"
            case [
                "An",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                root_suspect_name,
                "neighbor",
                suspect_with_neighbors,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    root_suspect_name, direction
                )
                neighbors = self.suspects[suspect_with_neighbors].neighbors

                self._set_has_parity(direction_suspects & neighbors, parity, verdict)

            case [
                suspect1_name,
                "and",
                suspect2_name,
                "share",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
            ]:
                if suspect2_name == "I":
                    suspect2_name = suspect_with_clue

                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                suspect2_neighbors = self.suspects[suspect2_name].neighbors
                shared_neighbors = suspect1_neighbors & suspect2_neighbors

                self._set_has_parity(shared_neighbors, parity, verdict)

            case [
                other_suspect,
                "and",
                "I",
                "have",
                "no",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "in",
                "common",
            ]:
                verdict = Verdict.parse(verdict_str)
                suspect_with_clue_neighbors = self.suspects[suspect_with_clue].neighbors
                other_suspect_neighbors = self.suspects[other_suspect].neighbors
                shared_neighbors = suspect_with_clue_neighbors & other_suspect_neighbors

                self._set_has_exactly_n_of_verdict(shared_neighbors, 0, verdict)

            case [
                suspect1_name,
                "and",
                suspect2_name,
                "have",
                "an",
                "equal",
                "number",
                "of",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
            ]:
                if suspect2_name == "I":
                    suspect2_name = suspect_with_clue

                verdict = Verdict.parse(verdict_str)
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                suspect2_neighbors = self.suspects[suspect2_name].neighbors
                suspect1_count = count_suspects_with_verdict(
                    suspect1_neighbors, verdict
                )
                suspect2_count = count_suspects_with_verdict(
                    suspect2_neighbors, verdict
                )

                self.solver.add(suspect1_count == suspect2_count)

            case [
                suspect1_name,
                "and",
                suspect2_name,
                "have",
                num_neighbors,
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "in",
                "common",
            ]:
                verdict = Verdict.parse(verdict_str)
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                suspect2_neighbors = self.suspects[suspect2_name].neighbors
                shared_neighbors = suspect1_neighbors & suspect2_neighbors

                self._set_has_exactly_n_of_verdict(
                    shared_neighbors, int(num_neighbors), verdict
                )

            # TODO - merge with above ("...have num_neighbors innocent/criminal neighbors in common")? same logic, just slightly different wording and hardcoded amount of 1
            case [
                suspect1_name,
                "and",
                suspect2_name,
                "have",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                "neighbor",
                "in",
                "common",
            ]:
                verdict = Verdict.parse(verdict_str)
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                suspect2_neighbors = self.suspects[suspect2_name].neighbors
                shared_neighbors = suspect1_neighbors & suspect2_neighbors

                self._set_has_exactly_n_of_verdict(shared_neighbors, 1, verdict)

            case [
                suspect1_name,
                "only",
                ("innocent" | "criminal") as verdict_str,
                "neighbor",
                "is",
                suspect2_name,
                "neighbor",
            ]:
                if suspect1_name == "My":
                    suspect1_name = suspect_with_clue

                verdict = Verdict.parse(verdict_str)
                # both original suspect names end with 's, e.g. "Helen's" (unless suspect1_name is "My", but then removesuffix("'s") is a harmless no-op)
                suspect1_name = suspect1_name.removesuffix("'s")
                suspect2_name = suspect2_name.removesuffix("'s")
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                suspect2_neighbors = self.suspects[suspect2_name].neighbors

                # first part - suspect 1 has exactly 1 innocent/criminal neighbor
                self._set_has_exactly_n_of_verdict(suspect1_neighbors, 1, verdict)

                # second part - exactly 1 innocent/criminal in shared neighbors of suspects 1 and 2
                self._set_has_exactly_n_of_verdict(
                    suspect1_neighbors & suspect2_neighbors, 1, verdict
                )

            # TODO - version for above/below
            case [
                suspect1_name,
                "only",
                ("innocent" | "criminal") as verdict_str,
                "neighbor",
                "is",
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect2_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                # suspect1_name ends with 's, e.g. "Alice's"
                suspect1_name = suspect1_name.removesuffix("'s")
                suspect1_neighbors = self.suspects[suspect1_name].neighbors

                # first part - suspect 1 has exactly 1 innocent/criminal neighbor
                self._set_has_exactly_n_of_verdict(suspect1_neighbors, 1, verdict)

                # second part - exactly 1 innocent criminal in intersection of suspect1_neighbors and suspects in direction of suspect 2
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect2_name, direction
                )

                self._set_has_exactly_n_of_verdict(
                    suspect1_neighbors & suspects_in_direction, 1, verdict
                )

            case [
                suspect_name,
                "only",
                ("innocent" | "criminal") as verdict_str,
                "neighbor",
                "is",
                "on",
                "the",
                "edges",
            ]:
                verdict = Verdict.parse(verdict_str)
                # suspect_name ends with 's, e.g. "Bruce's"
                suspect_name = suspect_name.removesuffix("'s")
                neighbors = self.suspects[suspect_name].neighbors

                # first part - suspect has exactly 1 innocent/criminal neighbor
                self._set_has_exactly_n_of_verdict(neighbors, 1, verdict)

                # second part - intersection of neighbors and edges has exactly 1 innocent/criminal
                self._set_has_exactly_n_of_verdict(
                    neighbors & self._edges(), 1, verdict
                )

            case [
                "There",
                "is",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                ("above" | "below") as direction_str,
                suspect_name,
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(suspects_in_direction, 1, verdict)

            case [
                "There",
                "are",
                "exactly",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                suspect_name,
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(
                    suspects_in_direction, int(num_suspects), verdict
                )

            case [
                "There",
                "is",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect_name,
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(suspects_in_direction, 1, verdict)

            # TODO - combine with above? ("There is only one innocent/criminal to the left/right of suspect") - same logic, just variable num_suspects instead of 1
            case [
                "There",
                "are",
                "exactly",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect_name,
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(
                    suspects_in_direction, int(num_suspects), verdict
                )

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
            ]:
                neighbor_subset = self._column(Column.parse(column))
                verdict = Verdict.parse(verdict_str)

                self._set_single_verdict(suspect_name, verdict)
                self._set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_suspects), verdict
                )

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
            ]:
                neighbor_subset = self._row(int(row))
                verdict = Verdict.parse(verdict_str)

                self._set_single_verdict(suspect_name, verdict)
                self._set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_suspects), verdict
                )

            case [
                "There",
                "is",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "column",
                column,
            ]:
                self._set_has_exactly_n_of_verdict(
                    self._column(Column.parse(column)), 1, Verdict.parse(verdict_str)
                )

            case [
                "There",
                "is",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "row",
                row,
            ]:
                self._set_has_exactly_n_of_verdict(
                    self._row(int(row)), 1, Verdict.parse(verdict_str)
                )

            # TODO - version of this for columns?
            case [
                "There",
                "are",
                "exactly",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
            ]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                self._set_has_exactly_n_of_verdict(
                    row_suspects, int(num_suspects), verdict
                )

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
            ]:
                verdict = Verdict.parse(verdict_str)
                num_suspects = word_to_int[num_suspects_str]

                # first part - verdict for suspect_name
                self._set_single_verdict(suspect_name, verdict)

                # second part - column has >= num_suspects of verdict
                column_suspects_count = count_suspects_with_verdict(
                    self._column(Column.parse(column)), verdict
                )

                self.solver.add(column_suspects_count >= num_suspects)

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
            ]:
                verdict = Verdict.parse(verdict_str)
                num_suspects = word_to_int[num_suspects_str]

                # first part - verdict for suspect_name
                self._set_single_verdict(suspect_name, verdict)

                # second part - row has >= num_suspects of verdict
                row_suspects_count = count_suspects_with_verdict(
                    self._row(int(row)), verdict
                )

                self.solver.add(row_suspects_count >= num_suspects)

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "on",
                "the",
                "edges",
            ]:
                verdict = Verdict.parse(verdict_str)

                # first part - verdict for suspect_name
                self._set_single_verdict(suspect_name, verdict)

                # second part - edges have num_suspects of verdict
                self._set_has_exactly_n_of_verdict(
                    self._edges(), int(num_suspects), verdict
                )

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "on",
                "the",
                "edges",
            ]:
                verdict = Verdict.parse(verdict_str)
                num_suspects = word_to_int[num_suspects_str]

                # first part - verdict for suspect_name
                self._set_single_verdict(suspect_name, verdict)

                # second part - edges have >= num_suspects of verdict
                edge_suspects_count = count_suspects_with_verdict(
                    self._edges(), verdict
                )

                self.solver.add(edge_suspects_count >= num_suspects)

            case [
                suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "a",
                "corner",
            ]:
                verdict = Verdict.parse(verdict_str)
                num_suspects = word_to_int[num_suspects_str]

                # first part - verdict for suspect_name
                self._set_single_verdict(suspect_name, verdict)

                # second part - corners have >= num_suspects of verdicts
                corner_suspects_count = count_suspects_with_verdict(
                    self._corners(), verdict
                )

                self.solver.add(corner_suspects_count >= num_suspects)

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                central_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - suspects in direction have >= num_suspects of verdict
                num_suspects = word_to_int[num_suspects_str]
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    central_suspect_name, direction
                )
                suspects_in_direction_count = count_suspects_with_verdict(
                    suspects_in_direction, verdict
                )

                self.solver.add(suspects_in_direction_count >= num_suspects)

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                central_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - suspects in direction have num_suspects of verdict
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    central_suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(
                    suspects_in_direction, int(num_suspects), verdict
                )

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                central_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - suspects in direction have >= num_suspects of verdict
                num_suspects = word_to_int[num_suspects_str]
                suspects_in_direction = self._get_suspects_relative_to_other_suspect(
                    central_suspect_name, direction
                )
                suspects_in_direction_count = count_suspects_with_verdict(
                    suspects_in_direction, verdict
                )

                self.solver.add(suspects_in_direction_count >= num_suspects)

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                central_suspect_name,
                num_neighbors,
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
            ]:
                # can be "...is one of *my* innocent/criminal neighbors"
                # check for that; if so, use suspect_with_clue
                if central_suspect_name == "my":
                    actual_central_suspect_name = suspect_with_clue
                else:
                    # original central_suspect_name ends with 's, e.g. "Helen's"
                    actual_central_suspect_name = central_suspect_name.removesuffix(
                        "'s"
                    )
                verdict = Verdict.parse(verdict_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - central_suspect has num_suspects neighbors of verdict
                central_suspect_neighbors = self.suspects[
                    actual_central_suspect_name
                ].neighbors

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - neighbors of central_suspect have >= num_suspects of verdict
                num_suspects = word_to_int[num_suspects_str]
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors
                num_neighbor_subset = count_suspects_with_verdict(
                    central_suspect_neighbors, verdict
                )

                self.solver.add(num_neighbor_subset >= num_suspects)

            case [
                identified_suspect_name,
                "is",
                "one",
                "of",
                num_suspects_str,
                "or",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "between",
                suspect1_name,
                "and",
                suspect2_name,
            ]:
                verdict = Verdict.parse(verdict_str)

                # first part - verdict for identified suspect
                self._set_single_verdict(identified_suspect_name, verdict)

                # second part - at least num_suspects with verdict between suspect1 and suspect2
                num_suspects = word_to_int[num_suspects_str]
                suspects_between = self._between_suspects(suspect1_name, suspect2_name)
                suspects_between_count = count_suspects_with_verdict(
                    suspects_between, verdict
                )

                self.solver.add(suspects_between_count >= num_suspects)

            case [
                "There",
                "are",
                "more",
                ("innocent" | "criminal") as verdict_str1,
                more_profession_plural,
                "than",
                ("innocent" | "criminal") as verdict_str2,
                less_profession_plural,
            ] if verdict_str1 == verdict_str2:
                more_profession = more_profession_plural.removesuffix("s")
                less_profession = less_profession_plural.removesuffix("s")
                verdict = Verdict.parse(verdict_str1)

                more_profession_count = count_suspects_with_verdict(
                    self._all_of_profession(more_profession), verdict
                )
                less_profession_count = count_suspects_with_verdict(
                    self._all_of_profession(less_profession), verdict
                )

                self.solver.add(more_profession_count > less_profession_count)

            case [
                "There",
                "are",
                "as",
                "many",
                ("innocent" | "criminal") as verdict_str1,
                profession1_plural,
                "as",
                "there",
                "are",
                ("innocent" | "criminal") as verdict_str2,
                profession2_plural,
            ] if verdict_str1 == verdict_str2:
                profession1 = profession1_plural.removesuffix("s")
                profession2 = profession2_plural.removesuffix("s")
                verdict = Verdict.parse(verdict_str1)

                profession1_count = count_suspects_with_verdict(
                    self._all_of_profession(profession1), verdict
                )
                profession2_count = count_suspects_with_verdict(
                    self._all_of_profession(profession2), verdict
                )

                self.solver.add(profession1_count == profession2_count)

            case [
                "Exactly",
                "1",
                profession,
                "has",
                "a" | "an",
                ("innocent" | "criminal") as verdict_str,
                "directly",
                ("above" | "below") as direction_str,
                "them",
            ]:
                verdict = Verdict.parse(verdict_str)
                profession_members = self._all_of_profession(profession)
                direction = Direction(direction_str)
                profession_neighbors = [
                    p.neighbor_in_direction(direction) for p in profession_members
                ]
                filtered_neighbors = [n for n in profession_neighbors if n is not None]

                self._set_has_exactly_n_of_verdict(set(filtered_neighbors), 1, verdict)

            case [
                "Exactly",
                "1",
                profession,
                "has",
                "a" | "an",
                ("innocent" | "criminal") as verdict_str,
                "directly",
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                "them",
            ]:
                verdict = Verdict.parse(verdict_str)
                profession_members = self._all_of_profession(profession)
                direction = Direction(direction_str)
                profession_neighbors = [
                    p.neighbor_in_direction(direction) for p in profession_members
                ]
                filtered_neighbors = [n for n in profession_neighbors if n is not None]

                self._set_has_exactly_n_of_verdict(set(filtered_neighbors), 1, verdict)

            case [
                num_of_profession,
                profession_plural,
                "have",
                "a",
                ("innocent" | "criminal") as verdict_str,
                "directly",
                ("above" | "below") as direction_str,
                "them",
            ]:
                verdict = Verdict.parse(verdict_str)
                profession = profession_plural.removesuffix("s")
                profession_members = self._all_of_profession(profession)
                direction = Direction(direction_str)
                profession_neighbors = [
                    p.neighbor_in_direction(direction) for p in profession_members
                ]
                filtered_neighbors = [n for n in profession_neighbors if n is not None]

                self._set_has_exactly_n_of_verdict(
                    set(filtered_neighbors), int(num_of_profession), verdict
                )

            # TODO - version for "to the left of/to the right of"
            case [
                "Exactly",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "are",
                ("above" | "below") as direction_str,
                other_suspect,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central_suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, num_neighbor_subset are in direction_str relative to other_suspect and have verdict
                neighbor_subset = (
                    central_suspect_neighbors
                    & self._get_suspects_relative_to_other_suspect(
                        other_suspect, direction
                    )
                )

                self._set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_neighbor_subset), verdict
                )

            case [
                "Neither",
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "are",
                ("above" | "below") as direction_str,
                other_suspect,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central_suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, 0 are in direction_str relative to other_suspect and have verdict
                neighbor_subset = (
                    central_suspect_neighbors
                    & self._get_suspects_relative_to_other_suspect(
                        other_suspect, direction
                    )
                )

                self._set_has_exactly_n_of_verdict(neighbor_subset, 0, verdict)

            case [
                "Neither",
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "are",
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                other_suspect,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central_suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, 0 are in direction_str relative to other_suspect and have verdict
                neighbor_subset = (
                    central_suspect_neighbors
                    & self._get_suspects_relative_to_other_suspect(
                        other_suspect, direction
                    )
                )

                self._set_has_exactly_n_of_verdict(neighbor_subset, 0, verdict)

            case [
                "Only",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "is",
                "in",
                "column",
                column,
            ]:
                verdict = Verdict.parse(verdict_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central_suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, num_neighbor_subset are in row
                column_suspects = self._column(Column.parse(column))

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors & column_suspects,
                    int(num_neighbor_subset),
                    verdict,
                )

            # TODO - should this be "is" | "are"? If num_neighbor_subset isn't 1, is the wording otherwise the same?
            case [
                "Only",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "is",
                "in",
                "row",
                row,
            ]:
                verdict = Verdict.parse(verdict_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central_suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, num_neighbor_subset are in row
                row_suspects = self._row(int(row))

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors & row_suspects,
                    int(num_neighbor_subset),
                    verdict,
                )

            # TODO - version of this for "above/below"
            case [
                "Only",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "is",
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                other_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                # first part - central suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, num_neighbor_subset are in direction_str relative to other_suspect
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    other_suspect_name, direction
                )

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors & direction_suspects,
                    int(num_neighbor_subset),
                    verdict,
                )

            # TODO - version of this for columns?
            case [
                "Exactly",
                num_row_subset,
                "of",
                "the",
                num_row,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "are",
                suspect_name,
                "neighbors",
            ]:
                # original suspect_name ends with 's, e.g. "Ollie's"
                suspect_name = suspect_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                # first part - row has num_row suspects with verdict
                self._set_has_exactly_n_of_verdict(row_suspects, int(num_row), verdict)

                # second part - intersection of row and suspect's neighbors has num_row_subset with verdict
                neighbors = self.suspects[suspect_name].neighbors

                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbors, int(num_row_subset), verdict
                )

            case [
                "Exactly",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                suspect_name,
                "are",
                "in",
                "row",
                row,
            ]:
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors

                # first part - suspect has num_neighbors with verdict
                self._set_has_exactly_n_of_verdict(
                    neighbors, int(num_neighbors), verdict
                )

                # second part - intersection of suspect's neighbors and row has num_neighbor_subset with verdict
                row_suspects = self._row(int(row))

                self._set_has_exactly_n_of_verdict(
                    neighbors & row_suspects, int(num_neighbor_subset), verdict
                )

            case [
                "Exactly",
                num_neighbor_subset,
                "of",
                central_suspect_name,
                num_neighbors,
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "also",
                "neighbor",
                other_suspect_name,
            ]:
                # first part - central_suspect has num_neighbors with verdict

                verdict = Verdict.parse(verdict_str)
                # central_suspect_name has "'s" at the end, e.g. "Isaac's"
                central_suspect_name = central_suspect_name.removesuffix("'s")
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of those neighbors, num_neighbor_subset are neighbors of other_suspect with that verdict
                neighbor_subset = (
                    central_suspect_neighbors
                    & self.suspects[other_suspect_name].neighbors
                )

                self._set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_neighbor_subset), verdict
                )

            # TODO - should this be "is" | "are"? If num_neighbor_subset isn't 1, is the wording otherwise the same?
            case [
                "Only",
                num_neighbor_subset,
                "of",
                "the",
                num_neighbors,
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                central_suspect_name,
                "is",
                other_suspect_name,
                "neighbor",
            ]:
                verdict = Verdict.parse(verdict_str)
                # other_suspect_name ends with 's, e.g. "Olof's"
                other_suspect_name = other_suspect_name.removesuffix("'s")

                # first part - central_suspect has num_neighbors with verdict
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors

                self._set_has_exactly_n_of_verdict(
                    central_suspect_neighbors, int(num_neighbors), verdict
                )

                # second part - of thise neighbors, num_neighbor_subset are neighbors of other_suspect with that verdict
                neighbor_subset = (
                    central_suspect_neighbors
                    & self.suspects[other_suspect_name].neighbors
                )

                self._set_has_exactly_n_of_verdict(
                    neighbor_subset, int(num_neighbor_subset), verdict
                )

            case [
                "An",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
                "neighbor",
                suspect_name,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors
                column_suspects = self._column(Column.parse(column))

                self._set_has_parity(neighbors & column_suspects, parity, verdict)

            case [
                "An",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "neighbor",
                suspect_name,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors
                row_suspects = self._row(int(row))

                self._set_has_parity(neighbors & row_suspects, parity, verdict)

            # TODO - combine with above ("An odd/even number of innocents/criminals in row ROW neighbor SUSPECT")? Same logic, different wording
            case [
                "There's",
                "an",
                ("odd" | "even") as parity_str,
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                suspect_name,
                "in",
                "row",
                row,
            ]:
                parity = Parity(parity_str)
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors
                row_suspects = self._row(int(row))

                self._set_has_parity(neighbors & row_suspects, parity, verdict)

            # TODO - version of this for columns
            case [
                "There's",
                "an",
                "equal",
                "number",
                "of",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "rows",
                row1,
                "and",
                row2,
            ]:
                verdict = Verdict.parse(verdict_str)
                row1_count = count_suspects_with_verdict(self._row(int(row1)), verdict)
                row2_count = count_suspects_with_verdict(self._row(int(row2)), verdict)

                self.solver.add(row1_count == row2_count)

            case [
                "Column",
                column,
                "has",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "than",
                "any",
                "other",
                "column",
            ]:
                self._column_has_most_of_verdict(
                    Column.parse(column), Verdict.parse(verdict_str)
                )

            case [
                "Row",
                row,
                "has",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "than",
                "any",
                "other",
                "row",
            ]:
                self._row_has_most_of_verdict(int(row), Verdict.parse(verdict_str))

            case [
                suspect_name,
                "has",
                "the",
                "most",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
            ]:
                verdict = Verdict.parse(verdict_str)

                self._suspect_has_most_neighbors_of_verdict(suspect_name, verdict)

            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "among",
                profession_plural,
                "than",
                "any",
                "other",
                "profession",
            ]:
                verdict = Verdict.parse(verdict_str)
                profession = profession_plural.removesuffix("s")

                self._profession_has_most_of_verdict(profession, verdict)

            case [
                suspect_name,
                "has",
                "more",
                ("innocent" | "criminal") as more_verdict,
                "than",
                ("innocent" | "criminal") as less_verdict,
                "neighbors",
            ] if more_verdict != less_verdict:
                neighbors = self.suspects[suspect_name].neighbors

                innocent_count = count_suspects_with_verdict(
                    neighbors, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    neighbors, Verdict.CRIMINAL
                )

                match more_verdict, less_verdict:
                    case "innocent", "criminal":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminal", "innocent":
                        self.solver.add(criminal_count > innocent_count)

            # TODO - version of this for columns
            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as more_verdict,
                "than",
                ("innocents" | "criminals") as less_verdict,
                "in",
                "row",
                row,
            ] if more_verdict != less_verdict:
                row_suspects = self._row(int(row))
                innocent_count = count_suspects_with_verdict(
                    row_suspects, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    row_suspects, Verdict.CRIMINAL
                )

                match more_verdict, less_verdict:
                    case "innocents", "criminals":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminals", "innocents":
                        self.solver.add(criminal_count > innocent_count)

            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as more_verdict,
                "than",
                ("innocents" | "criminals") as less_verdict,
                ("above" | "below") as direction_str,
                suspect_name,
            ] if more_verdict != less_verdict:
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, Direction(direction_str)
                )
                innocent_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.CRIMINAL
                )

                match more_verdict, less_verdict:
                    case "innocents", "criminals":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminals", "innocents":
                        self.solver.add(criminal_count > innocent_count)

            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as more_verdict,
                "than",
                ("innocents" | "criminals") as less_verdict,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect_name,
            ] if more_verdict != less_verdict:
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, Direction(direction_str)
                )
                innocent_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.CRIMINAL
                )

                match more_verdict, less_verdict:
                    case "innocents", "criminals":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminals", "innocents":
                        self.solver.add(criminal_count > innocent_count)

            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as more_verdict,
                "than",
                ("innocents" | "criminals") as less_verdict,
                "in",
                "between",
                suspect1_name,
                "and",
                suspect2_name,
            ] if more_verdict != less_verdict:
                suspects_between = self._between_suspects(suspect1_name, suspect2_name)
                innocent_count = count_suspects_with_verdict(
                    suspects_between, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    suspects_between, Verdict.CRIMINAL
                )

                match more_verdict, less_verdict:
                    case "innocents", "criminals":
                        self.solver.add(innocent_count > criminal_count)
                    case "criminals", "innocents":
                        self.solver.add(criminal_count > innocent_count)

            # TODO - version of this for "to the left/right of"
            case [
                "There",
                "are",
                "as",
                "many",
                ("innocents" | "criminals") as verdict1,
                "as",
                ("innocents" | "criminals") as verdict2,
                ("above" | "below") as direction_str,
                suspect_name,
            ] if verdict1 != verdict2:
                direction = Direction(direction_str)
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )
                innocent_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.INNOCENT
                )
                criminal_count = count_suspects_with_verdict(
                    direction_suspects, Verdict.CRIMINAL
                )

                self.solver.add(innocent_count == criminal_count)

            # TODO - version of this for columns
            case [
                "There",
                "are",
                "more",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                more_row,
                "than",
                "row",
                less_row,
            ]:
                verdict = Verdict.parse(verdict_str)
                more_suspect_count = count_suspects_with_verdict(
                    self._row(int(more_row)), verdict
                )
                less_suspect_count = count_suspects_with_verdict(
                    self._row(int(less_row)), verdict
                )

                self.solver.add(more_suspect_count > less_suspect_count)

            case [
                more_suspect_name,
                "has",
                "more",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "than",
                less_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                more_suspect_neighbors = self.suspects[more_suspect_name].neighbors
                more_suspect_neighbor_count = count_suspects_with_verdict(
                    more_suspect_neighbors, verdict
                )
                less_suspect_neighbors = self.suspects[less_suspect_name].neighbors
                less_suspect_neighbor_count = count_suspects_with_verdict(
                    less_suspect_neighbors, verdict
                )

                self.solver.add(
                    more_suspect_neighbor_count > less_suspect_neighbor_count
                )

            # TODO - combine with above (more_suspect has more innocent/criminal neighbors than less_suspect)? same logic, just uses more_suspect instead of suspect_with_clue
            case [
                "I",
                "have",
                "more",
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "than",
                less_suspect_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                more_suspect_neighbors = self.suspects[suspect_with_clue].neighbors
                more_suspect_neighbor_count = count_suspects_with_verdict(
                    more_suspect_neighbors, verdict
                )
                less_suspect_neighbors = self.suspects[less_suspect_name].neighbors
                less_suspect_neighbor_count = count_suspects_with_verdict(
                    less_suspect_neighbors, verdict
                )

                self.solver.add(
                    more_suspect_neighbor_count > less_suspect_neighbor_count
                )

            # TODO - version of this for "to the left/right"?
            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                suspect1_name,
                "are",
                suspect2_name,
                "neighbors",
            ]:
                # original suspect2_name has "'s" at the end, e.g. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there are exactly two innocents/criminals in direction_str relative to suspect1
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect1_name, direction
                )
                self._set_has_exactly_n_of_verdict(direction_suspects, 2, verdict)

                # second part - there are exactly two innocent/criminals in intersection of direction_suspects and neighbors of suspect2
                neighbor_suspects = self.suspects[suspect2_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    direction_suspects & neighbor_suspects, 2, verdict
                )

            # TODO - combine with above case? same logic, this is just talking about 1 suspect instead of 2
            case [
                "The",
                "only",
                ("innocent" | "criminal") as verdict_str,
                ("above" | "below") as direction_str,
                suspect1_name,
                "is",
                suspect2_name,
                "neighbor",
            ]:
                # original suspect2_name has "'s" at the end, e.g. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there is exactly one innocent/criminal in direction_str relative to suspect1
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect1_name, direction
                )
                self._set_has_exactly_n_of_verdict(direction_suspects, 1, verdict)

                # second part - there is exactly one innocent/criminal in intersection of direction_suspects and neighbors of suspect2
                neighbor_suspects = self.suspects[suspect2_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    direction_suspects & neighbor_suspects, 1, verdict
                )

            case [
                "The",
                "only",
                ("innocent" | "criminal") as verdict_str,
                "to",
                "the",
                ("left" | "right") as direction_str,
                "of",
                suspect1_name,
                "is",
                suspect2_name,
                "neighbor",
            ]:
                # original suspect2_name has "'s" at the end, e.g. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there is exactly one innocent/criminal in direction_str relative to suspect1
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect1_name, direction
                )
                self._set_has_exactly_n_of_verdict(direction_suspects, 1, verdict)

                # second part - there is exactly one innocent/criminal in intersection of direction_suspects and neighbors of suspect2
                neighbor_suspects = self.suspects[suspect2_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    direction_suspects & neighbor_suspects, 1, verdict
                )

            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction1_str,
                suspect1_name,
                "are",
                ("above" | "below") as direction2_str,
                suspect2_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                direction1 = Direction(direction1_str)
                direction2 = Direction(direction2_str)

                # first part - exactly 2 innocents/criminals above/below suspect1
                suspects1 = self._get_suspects_relative_to_other_suspect(
                    suspect1_name, direction1
                )

                self._set_has_exactly_n_of_verdict(suspects1, 2, verdict)

                # second part - exactly 2 innocents/criminals above/below suspect1 *and* above/below suspect2
                suspects2 = self._get_suspects_relative_to_other_suspect(
                    suspect2_name, direction2
                )

                self._set_has_exactly_n_of_verdict(suspects1 & suspects2, 2, verdict)

            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                suspect1_name,
                "are",
                suspect2_name,
                "neighbors",
            ]:
                # original suspect2_name has "'s" at the end, e.g. "Isaac's"
                suspect2_name = suspect2_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)

                # first part - there are exactly two innocents/criminals neighboring suspect1
                suspect1_neighbors = self.suspects[suspect1_name].neighbors
                self._set_has_exactly_n_of_verdict(suspect1_neighbors, 2, verdict)

                # second part - there are exactly two innocent/criminals that are neighbors of both suspect1 and suspect 2
                suspect2_neighbors = self.suspects[suspect2_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    suspect1_neighbors & suspect2_neighbors, 2, verdict
                )

            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                "neighboring",
                suspect_name,
                "are",
                "on",
                "the",
                "edges",
            ]:
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors

                # first part - there are exactly two innocents/criminals neighboring suspect
                self._set_has_exactly_n_of_verdict(neighbors, 2, verdict)

                # second part - there are exactly two innocents/criminals neighboring suspect and on the edges
                edges = self._edges()

                self._set_has_exactly_n_of_verdict(neighbors & edges, 2, verdict)

            case [
                "There",
                "is",
                "only",
                "one",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "between",
                suspect1_name,
                "and",
                suspect2_name,
            ]:
                verdict = Verdict.parse(verdict_str)
                suspects_between = self._between_suspects(suspect1_name, suspect2_name)

                self._set_has_exactly_n_of_verdict(suspects_between, 1, verdict)

            case [
                "The",
                "only",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "between",
                suspect1_name,
                "and",
                suspect2_name,
                "is",
                central_suspect_name,
                "neighbor",
            ]:
                # central_suspect_name has "'s" at the end, e.g. "Isaac's"
                central_suspect_name = central_suspect_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)

                # first part - there is exactly one innocent/criminal in between suspect1 and suspect2
                suspects_between = self._between_suspects(suspect1_name, suspect2_name)
                self._set_has_exactly_n_of_verdict(suspects_between, 1, verdict)

                # second part - there is exactly one innocent/criminal in intersection of (between suspects 1 and 2) and neighbors of central_suspect
                central_suspect_neighbors = self.suspects[
                    central_suspect_name
                ].neighbors
                self._set_has_exactly_n_of_verdict(
                    suspects_between & central_suspect_neighbors, 1, verdict
                )

            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
                "are",
                suspect_name,
                "neighbors",
            ]:
                # original suspect_name has "'s" at the end, e.g. "Isaac's"
                suspect_name = suspect_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                column_suspects = self._column(Column.parse(column))

                # first part - there are exactly 2 innocents/criminals in column
                self._set_has_exactly_n_of_verdict(column_suspects, 2, verdict)

                # second part - there is exactly one innocent/criminal in intersection of column and neighbors of suspect_name
                neighbor_suspects = self.suspects[suspect_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    column_suspects & neighbor_suspects, 2, verdict
                )

            # TODO - version of this for columns?
            case [
                "The",
                "only",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "row",
                row,
                "is",
                suspect_name,
                "neighbor",
            ]:
                # original suspect_name has "'s" at the end, e.g. "Isaac's"
                suspect_name = suspect_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                # first part - there is exactly one innocent/criminal in row
                self._set_has_exactly_n_of_verdict(row_suspects, 1, verdict)

                # second part - there is exactly one innocent/criminal in intersection of row and neighbors of suspect_name
                neighbor_suspects = self.suspects[suspect_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbor_suspects, 1, verdict
                )

            # TODO - combine with above ("The only innocent/criminal in row N is suspect's neighbor")? same logic, just different count
            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "are",
                suspect_name,
                "neighbors",
            ]:
                # original suspect_name has "'s" at the end, e.g. "Isaac's"
                suspect_name = suspect_name.removesuffix("'s")
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                # first part - there are exactly 2 innocents/criminals in row
                self._set_has_exactly_n_of_verdict(row_suspects, 2, verdict)

                # second part - there is exactly one innocent/criminal in intersection of row and neighbors of suspect_name
                neighbor_suspects = self.suspects[suspect_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbor_suspects, 2, verdict
                )

            case [
                "Only",
                num_neighbors,
                "of",
                "the",
                num_verdict,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
                "is",
                suspect_name,
                "neighbor",
            ]:
                verdict = Verdict.parse(verdict_str)
                # original suspect_name ends with 's, e.g. "David's"
                suspect_name = suspect_name.removesuffix("'s")
                column_suspects = self._column(Column.parse(column))

                # first part - there are exactly num_verdict innocents/criminals in column
                self._set_has_exactly_n_of_verdict(
                    column_suspects, int(num_verdict), verdict
                )

                # second part - there are exactly num_neighbors innocents/criminals in intersection of column and neighbors of suspect_with_clue
                neighbors = self.suspects[suspect_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    column_suspects & neighbors, int(num_neighbors), verdict
                )

            # TODO - version of this for columns?
            case [
                "Only",
                num_neighbors,
                "of",
                "the",
                num_verdict,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "is",
                "my",
                "neighbor",
            ]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                # first part - there are exactly num_verdict innocents/criminals in row
                self._set_has_exactly_n_of_verdict(
                    row_suspects, int(num_verdict), verdict
                )

                # second part - there are exactly num_neighbors innocents/criminals in intersection of row and neighbors of suspect_with_clue
                neighbors = self.suspects[suspect_with_clue].neighbors
                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbors, int(num_neighbors), verdict
                )

            # TODO - version of this for columns?
            # TODO - merge with above ("Only num_neighbors of the innocents/criminals in row ROW is my neighbor")? Same logic, just with variable suspect_name instead of suspect_with_clue
            case [
                "Only",
                num_neighbors,
                "of",
                "the",
                num_verdict,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "is",
                suspect_name,
                "neighbor",
            ]:
                verdict = Verdict.parse(verdict_str)
                # original suspect_name ends with 's, e.g. "David's"
                suspect_name = suspect_name.removesuffix("'s")
                row_suspects = self._row(int(row))

                # first part - there are exactly num_verdict innocents/criminals in row
                self._set_has_exactly_n_of_verdict(
                    row_suspects, int(num_verdict), verdict
                )

                # second part - there are exactly num_neighbors innocents/criminals in intersection of row and neighbors of suspect_with_clue
                neighbors = self.suspects[suspect_name].neighbors
                self._set_has_exactly_n_of_verdict(
                    row_suspects & neighbors, int(num_neighbors), verdict
                )

            # TODO - version of this for "to the left/right"?
            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                "me",
                "are",
                "connected",
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there are exactly two innocents/criminals in direction_str relative to suspect
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_with_clue, direction
                )
                self._set_has_exactly_n_of_verdict(direction_suspects, 2, verdict)

                # second part - all innocents/criminals in direction_str are connected
                self._all_suspects_in_vertical_set_with_verdict_are_connected(
                    direction_suspects, verdict
                )

            # TODO - version of this for "to the left/right"?
            # TODO - combine with above (Both innocents/criminals above/below me are connected)? Same logic, just with suspect_name instead of suspect_with_clue
            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                suspect_name,
                "are",
                "connected",
            ]:
                verdict = Verdict.parse(verdict_str)
                direction = Direction(direction_str)

                # first part - there are exactly two innocents/criminals in direction_str relative to suspect
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )
                self._set_has_exactly_n_of_verdict(direction_suspects, 2, verdict)

                # second part - all innocents/criminals in direction_str are connected
                self._all_suspects_in_vertical_set_with_verdict_are_connected(
                    direction_suspects, verdict
                )

            # TODO - version of this for column?
            case [
                "Both",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "are",
                "connected",
            ]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                # first part - exactly 2 innocents/criminals in row
                self._set_has_exactly_n_of_verdict(row_suspects, 2, verdict)

                # second part - all innocents/criminals in row are connected
                self._all_suspects_in_horizontal_set_with_verdict_are_connected(
                    row_suspects, verdict
                )

            case [
                "All",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "column",
                column,
                "are",
                "connected",
            ]:
                verdict = Verdict.parse(verdict_str)
                column_suspects = self._column(Column.parse(column))

                self._all_suspects_in_vertical_set_with_verdict_are_connected(
                    column_suspects, verdict
                )

            case [
                "All",
                ("innocents" | "criminals") as verdict_str,
                "in",
                "row",
                row,
                "are",
                "connected",
            ]:
                verdict = Verdict.parse(verdict_str)
                row_suspects = self._row(int(row))

                self._all_suspects_in_horizontal_set_with_verdict_are_connected(
                    row_suspects, verdict
                )

            # TODO - version for "to the left/right"
            case [
                "All",
                ("innocents" | "criminals") as verdict_str,
                ("above" | "below") as direction_str,
                suspect_name,
                "are",
                "connected",
            ]:
                direction = Direction(direction_str)
                verdict = Verdict.parse(verdict_str)
                direction_suspects = self._get_suspects_relative_to_other_suspect(
                    suspect_name, direction
                )

                self._all_suspects_in_vertical_set_with_verdict_are_connected(
                    direction_suspects, verdict
                )

            case [
                "There",
                "are",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "in",
                "total",
            ]:
                verdict = Verdict.parse(verdict_str)

                self._set_has_exactly_n_of_verdict(
                    set(self.suspects.values()), int(num_suspects), verdict
                )

            case [
                suspect1_name,
                "and",
                suspect2_name,
                "have",
                num_neighbors,
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
                "in",
                "total",
            ]:
                verdict = Verdict.parse(verdict_str)
                suspect1_neighbor_count = count_suspects_with_verdict(
                    self.suspects[suspect1_name].neighbors, verdict
                )
                suspect2_neighbor_count = count_suspects_with_verdict(
                    self.suspects[suspect2_name].neighbors, verdict
                )

                self.solver.add(
                    suspect1_neighbor_count + suspect2_neighbor_count
                    == int(num_neighbors)
                )

            # either "Olivia has" or "I have"
            case [
                suspect_name,
                "has" | "have",
                "exactly",
                num_suspects,
                ("innocent" | "criminal") as verdict_str,
                "neighbors",
            ]:
                if suspect_name == "I":
                    suspect_name = suspect_with_clue
                verdict = Verdict.parse(verdict_str)
                neighbors = self.suspects[suspect_name].neighbors

                self._set_has_exactly_n_of_verdict(
                    neighbors, int(num_suspects), verdict
                )

            # TODO - version of this for columns
            case [
                "The",
                "only",
                ("innocent" | "criminal") as verdict_str,
                "in",
                "a",
                "corner",
                "is",
                "in",
                "row",
                row,
            ]:
                verdict = Verdict.parse(verdict_str)

                # first part - only 1 innocent/criminal in the corners
                corner_suspects = self._corners()

                self._set_has_exactly_n_of_verdict(corner_suspects, 1, verdict)

                # second part - only 1 innocent/criminal in the intersection of corners and row
                row_suspects = self._row(int(row))

                self._set_has_exactly_n_of_verdict(
                    corner_suspects & row_suspects, 1, verdict
                )

            case [
                "None",
                "of",
                "the",
                num_suspects,
                ("innocents" | "criminals") as verdict_str,
                "on",
                "the",
                "edges",
                "is",
                "a",
                profession,
            ]:
                verdict = Verdict.parse(verdict_str)

                # first part - num_suspects with verdict on the edges
                edge_suspects = self._edges()

                self._set_has_exactly_n_of_verdict(
                    edge_suspects, int(num_suspects), verdict
                )

                # second part - intersection of edges with all of profession has no suspects with verdict
                profession_members = self._all_of_profession(profession)

                self._set_has_exactly_n_of_verdict(
                    edge_suspects & profession_members, 0, verdict
                )

            case _:
                print(f"Unrecognized clue type: {clue}")

    # primary entrypoint

    def add_clue(self, clue: str, suspect_with_clue: str) -> list[SuspectSolution]:
        """
        Add a new clue to the puzzle, including the source of the clue. Returns a list of newly deduced solutions.
        """
        new_solutions_found = []
        self._add_constraints_from_clue(clue, suspect_with_clue)

        while True:
            match self._solve_one():
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
