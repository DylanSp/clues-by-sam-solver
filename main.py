from z3 import *


def main():
    x = Int('x')
    y = Int('y')
    s = Solver()
    s.add(x > 2, y < 10, x + 2*y == 7)
    print(s.check())
    print(s.model())


if __name__ == "__main__":
    main()
