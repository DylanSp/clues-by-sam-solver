from dataclasses import dataclass
from typing import Optional, Set

from z3 import And, Bool, Not, Or, Solver, sat, unsat

from models import Column, Direction, Parity, Verdict
from puzzle.clue_parser import add_constraints_from_clue
from puzzle.common import PuzzleSuspect, count_suspects_with_verdict

NUM_ROWS = 5
NUM_COLS = 4


@dataclass
class RawSuspect:
    name: str
    profession: str


@dataclass
class PuzzleInput:
    suspects: list[RawSuspect]  # assumed to be sorted
    starting_suspect_name: str
    starting_suspect_verdict: Verdict


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
                column=Column.from_int(idx % NUM_COLS),
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
                        and int(suspect.column) < int(root_suspect.column)
                    ]
                )
            case Direction.RIGHT:
                return set(
                    [
                        suspect
                        for suspect in self.suspects.values()
                        if suspect.row == root_suspect.row
                        and int(suspect.column) > int(root_suspect.column)
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
            left_suspect = min(suspect1, suspect2, key=lambda s: int(s.column))
            right_suspect = max(suspect1, suspect2, key=lambda s: int(s.column))

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
        count = count_suspects_with_verdict(suspects, verdict)
        self.solver.add(count == num_of_verdict)

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

    # primary entrypoint
    def add_clue(self, clue: str, suspect_with_clue: str) -> list[SuspectSolution]:
        """
        Add a new clue to the puzzle, including the source of the clue. Returns a list of newly deduced solutions.
        """
        new_solutions_found = []
        add_constraints_from_clue(self, clue, suspect_with_clue)

        while True:
            match self._solve_one():
                case None:
                    break
                case solution:
                    new_solutions_found.append(solution)

        return new_solutions_found
