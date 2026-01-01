from enum import Enum


class Verdict(Enum):
    INNOCENT = "innocent"
    CRIMINAL = "criminal"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, input: str) -> "Verdict":
        match input:
            case "innocent" | "innocents":
                return Verdict.INNOCENT
            case "criminal" | "criminals":
                return Verdict.CRIMINAL

        raise ValueError(f"{input} is not a valid verdict")


class Column(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

    def __int__(self) -> int:
        match self:
            case Column.A:
                return 0
            case Column.B:
                return 1
            case Column.C:
                return 2
            case Column.D:
                return 3

    # Used when setting up the grid of suspects
    @classmethod
    def from_int(cls, input: int) -> "Column":
        match input:
            case 0:
                return Column.A
            case 1:
                return Column.B
            case 2:
                return Column.C
            case 3:
                return Column.D

        raise ValueError(f"{input} is not a valid column number")


class Direction(Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"


class Parity(Enum):
    ODD = "odd"
    EVEN = "even"
