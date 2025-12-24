from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Set
from z3 import BoolRef


class Verdict(Enum):
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
