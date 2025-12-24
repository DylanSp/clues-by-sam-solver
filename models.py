from enum import Enum, IntEnum


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
