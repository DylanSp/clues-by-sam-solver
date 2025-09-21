from z3 import Int, Solver, sat, Or, And, Not, Bools

from fixed_point import run_fixed_point_example


def main():
    run_fixed_point_example()
    return

    a, b = Bools("a b")

    s = Solver()

    s.add(Or(And(a, Not(b)), And(Not(a), b)))
    s.check()
    print(s.model())

    # while s.check() == sat:
    #     print(s.model())
    #     s.add(Or(a != s.model()[a], b != s.model()[b]))


if __name__ == "__main__":
    main()
