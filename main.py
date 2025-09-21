from z3 import Int, Solver, sat, Or


def main():
    x = Int('x')
    y = Int('y')
    s = Solver()
    s.add(x >= 0)
    s.add(y >= 0)
    s.add(x + y == 3)

    while s.check() == sat:
        print(s.model())
        s.add(Or(x != s.model()[x], y != s.model()[y]))


if __name__ == "__main__":
    main()
