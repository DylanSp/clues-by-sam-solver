from z3 import Fixedpoint, Bools, And, Or, Not


def run_fixed_point_example():
    fp = Fixedpoint()
    a, b, c, d = Bools("a b c d")
    fp.register_relation(a.decl(), b.decl(), c.decl(), d.decl())
    fp.rule(b, a)
    fp.fact(a)

    fp.rule(c, Not(d))
    fp.rule(d, Not(c))

    # fp.fact(Or(And(b, Not(c)), And(Not(b), c)))

    # fp.fact((b, not c) or (c, not b))  # type: ignore

    fp.set(engine='datalog')

    print("current set of rules\n", fp)
    print(fp.query(a))
    print(fp.query(b))
    print(fp.query(c))
    print(fp.query(d))

    # fp.fact(c)
    # print("updated set of rules\n", fp)
    # print(fp.query(a))
    # print(fp.get_answer())
