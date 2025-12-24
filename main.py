import json
from models import Verdict
from puzzle import Puzzle, PuzzleInput


# Get input data from browser console
# data from Puzzle Pack #1, puzzle 1
# https://cluesbysam.com/s/user/63f90e0e67bb92cd/pack-1/1/
raw_input_data = '[{"name":"Alex","profession":"cook"},{"name":"Bonnie","profession":"painter"},{"name":"Chris","profession":"cook"},{"name":"Ellie","profession":"cop"},{"name":"Frank","profession":"farmer"},{"name":"Helen","profession":"cook"},{"name":"Isaac","profession":"guard"},{"name":"Julie","profession":"clerk"},{"name":"Keith","profession":"farmer"},{"name":"Megan","profession":"painter"},{"name":"Nancy","profession":"guard"},{"name":"Olof","profession":"clerk"},{"name":"Paula","profession":"cop"},{"name":"Ryan","profession":"sleuth"},{"name":"Sofia","profession":"guard"},{"name":"Terry","profession":"sleuth"},{"name":"Vicky","profession":"farmer"},{"name":"Wally","profession":"mech"},{"name":"Xavi","profession":"mech"},{"name":"Zara","profession":"mech"}]'


def main():
    input_data = PuzzleInput(
        suspects=json.loads(raw_input_data),
        starting_suspect_name="Frank",
        starting_suspect_verdict=Verdict.INNOCENT
    )

    puzzle = Puzzle(input_data)

    # first clue, from Frank
    puzzle.handle_clue("Exactly 1 innocent in column A is neighboring Megan")
    puzzle.solve_many()
    print()

    # second clue, from Keith
    puzzle.handle_clue(
        "There's an odd number of criminals to the left of Sofia")
    puzzle.solve_many()
    print()

    # third clue, from Ryan
    puzzle.handle_clue("There is only one innocent above Keith")
    puzzle.solve_many()
    print()

    # fourth clue, from Alex
    puzzle.handle_clue("Both criminals below me are connected", "Alex")
    puzzle.solve_many()
    print()

    # fifth clue, from Vicky
    puzzle.handle_clue("Xavi is one of 4 innocents in column C")
    puzzle.solve_many()
    print()

    # sixth clue, from Xavi
    puzzle.handle_clue(
        "There are as many criminal farmers as there are criminal guards")
    puzzle.solve_many()
    print()

    # seventh clue, from Chris
    puzzle.handle_clue("Exactly 1 guard has a criminal directly below them")
    puzzle.solve_many()
    print()

    # eighth clue, from Isaac
    puzzle.handle_clue(
        "Exactly 2 of the 3 criminals neighboring Megan are above Vicky")
    puzzle.solve_many()
    print()

    # ninth clue, from Helen
    puzzle.handle_clue("All innocents in row 5 are connected")
    puzzle.solve_many()
    print()

    # tenth clue, from Wally
    puzzle.handle_clue("An odd number of innocents in row 1 neighbor Helen")
    puzzle.solve_many()
    print()

    # eleventh clue, from Bonnie
    puzzle.handle_clue("Column C has more innocents than any other column")
    puzzle.solve_many()
    print()

    # twelfth clue, from Megan
    puzzle.handle_clue("there are more criminals than innocents in row 1")
    puzzle.solve_many()
    print()

    # thirteenth clue, from Ellie
    puzzle.handle_clue("Chris has more criminal neighbors than Paula")
    puzzle.solve_many()
    print()

    # fourteenth clue, from Julie
    puzzle.handle_clue("Terry is one of two or more innocents on the edges")
    puzzle.solve_many()
    print()

    # fifteenth clue, from Terry
    puzzle.handle_clue("Both criminals above Zara are Isaac's neighbors")
    puzzle.solve_many()
    print()

    # sixteenth clue, from Olof
    puzzle.handle_clue("The only criminal below Julie is Terry's neighbor")
    puzzle.solve_many()
    print()

    # seventeenth and final clue, from Zara
    puzzle.handle_clue("Xavi has more innocent neighbors than Isaac")
    puzzle.solve_many()
    print()


if __name__ == "__main__":
    main()
