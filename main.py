from collections import deque
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

    # clue, then suspect who gave clue
    clues = [
        ["Exactly 1 innocent in column A is neighboring Megan", "Frank"],
        ["There's an odd number of criminals to the left of Sofia", "Keith"],
        ["There is only one innocent above Keith", "Ryan"],
        ["Both criminals below me are connected", "Alex"],
        ["Xavi is one of 4 innocents in column C", "Vicky"],
        ["There are as many criminal farmers as there are criminal guards", "Xavi"],
        ["Exactly 1 guard has a criminal directly below them", "Chris"],
        ["Exactly 2 of the 3 criminals neighboring Megan are above Vicky", "Isaac"],
        ["All innocents in row 5 are connected", "Helen"],
        ["An odd number of innocents in row 1 neighbor Helen", "Wally"],
        ["Column C has more innocents than any other column", "Bonnie"],
        ["there are more criminals than innocents in row 1", "Megan"],
        ["Chris has more criminal neighbors than Paula", "Ellie"],
        ["Terry is one of two or more innocents on the edges", "Julie"],
        ["Both criminals above Zara are Isaac's neighbors", "Terry"],
        ["The only criminal below Julie is Terry's neighbor", "Olof"],
        ["Xavi has more innocent neighbors than Isaac", "Zara"],
    ]

    # TODO - initialize only with initial clue (when interactivity is added)
    unhandled_clues = deque(clues)

    # main driver loop
    while not puzzle.is_solved() and len(unhandled_clues) > 0:
        next_clue_data = unhandled_clues.popleft()
        next_clue = next_clue_data[0]
        source = next_clue_data[1]
        newly_solved_suspects = puzzle.add_clue(next_clue, source)

        for solution in newly_solved_suspects:
            print(f"{solution.name} is {solution.verdict}")

        # TODO - for each newly_solved_suspect, reveal them, assert that the returned verdict is correct, push() their revealed clue to unhandled_clues
        # TODO - possibly detect flavor text, don't add it to unhandled_clues

    if puzzle.is_solved():
        print("Puzzle solved!")
    else:
        print("Puzzle unsolved, bug somewhere")


if __name__ == "__main__":
    main()
