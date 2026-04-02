from dataclasses import dataclass
from typing import Optional, Set

from z3 import BoolRef, If, Sum

from models import Column, Direction, Verdict


# all the data on a suspect that the Puzzle engine needs, including location, neighbors, and a Z3 BoolRef for tracking deduced verdict
@dataclass
class PuzzleSuspect:
    name: str
    profession: str
    row: int
    column: Column
    neighbors: Set["PuzzleSuspect"]
    all_other_suspects: Set["PuzzleSuspect"]
    is_innocent: BoolRef  # Z3 expression - true iff suspect is innocent

    # Names are unique and immutable, so can be used for equality testing; can't use default hash because Sets aren't hashable
    def __hash__(self) -> int:
        return hash(self.name)

    def neighbor_to(self, direction: Direction) -> Optional["PuzzleSuspect"]:
        for neighbor in self.neighbors:
            match direction:
                case Direction.ABOVE:
                    if neighbor.column == self.column and neighbor.row == self.row - 1:
                        return neighbor
                case Direction.BELOW:
                    if neighbor.column == self.column and neighbor.row == self.row + 1:
                        return neighbor
                case Direction.LEFT:
                    if neighbor.row == self.row and int(neighbor.column) == int(self.column) - 1:
                        return neighbor
                case Direction.RIGHT:
                    if neighbor.row == self.row and int(neighbor.column) == int(self.column) + 1:
                        return neighbor
        return None

    def suspects_to(self, direction: Direction) -> Set["PuzzleSuspect"]:
        match direction:
            case Direction.ABOVE:
                return set(
                    [
                        suspect
                        for suspect in self.all_other_suspects
                        if suspect.column == self.column and suspect.row < self.row
                    ]
                )
            case Direction.BELOW:
                return set(
                    [
                        suspect
                        for suspect in self.all_other_suspects
                        if suspect.column == self.column and suspect.row > self.row
                    ]
                )
            case Direction.LEFT:
                return set(
                    [
                        suspect
                        for suspect in self.all_other_suspects
                        if suspect.row == self.row and int(suspect.column) < int(self.column)
                    ]
                )
            case Direction.RIGHT:
                return set(
                    [
                        suspect
                        for suspect in self.all_other_suspects
                        if suspect.row == self.row and int(suspect.column) > int(self.column)
                    ]
                )

    # Fast __repr__ method to allow debugging without needing to calculate repr for Z3 values
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at {hex(id(self))}"


def count_suspects_with_verdict(suspects: set[PuzzleSuspect], verdict: Verdict):
    if verdict == Verdict.INNOCENT:
        return Sum([If(s.is_innocent, 1, 0) for s in suspects])
    elif verdict == Verdict.CRIMINAL:
        return Sum([If(s.is_innocent, 0, 1) for s in suspects])
