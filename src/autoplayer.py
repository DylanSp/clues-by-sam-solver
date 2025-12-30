from collections import deque
from typing import Tuple
from playwright.sync_api import sync_playwright

from puzzle import PuzzleInput, RawSuspect, Puzzle
from models import Verdict


def complete_puzzle(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()

        record_videos = False

        if record_videos:
            # manually start a context so we can specify directory for videos
            context = browser.new_context(record_video_dir="videos/")
        else:
            context = browser.new_context()

        page = context.new_page()
        page.goto(url)

        # get all suspect names
        name_elements = page.locator("css=h3.name")
        names = name_elements.all_inner_texts()

        # get all suspect professions
        profession_elements = page.locator("css=p.profession")
        professions = profession_elements.all_inner_texts()

        suspects = [
            RawSuspect(name, profession) for name, profession in zip(names, professions)
        ]

        # get initial suspect
        starting_suspect_card = page.locator("css=div.card.flipped").first
        starting_suspect_name = starting_suspect_card.locator(
            "css=h3.name"
        ).inner_text()

        initial_clue = starting_suspect_card.locator("css=p.hint").inner_text()

        starting_suspect_card_classes = starting_suspect_card.get_attribute("class")
        assert starting_suspect_card_classes is not None, (
            "Initial suspect card does not have any CSS classes specified"
        )
        starting_suspect_card_classes = starting_suspect_card_classes.split()
        if "innocent" in starting_suspect_card_classes:
            starting_suspect_verdict = Verdict.INNOCENT
        elif "criminal" in starting_suspect_card_classes:
            starting_suspect_verdict = Verdict.CRIMINAL
        else:
            raise ValueError(
                "Could not determine whether revealed suspect was innocent or criminal"
            )

        puzzle = Puzzle(
            PuzzleInput(suspects, starting_suspect_name, starting_suspect_verdict)
        )

        unhandled_clues: deque[Tuple[str, str]] = deque()
        unhandled_clues.append((initial_clue, starting_suspect_name))

        # main driver loop
        while not puzzle.is_solved() and len(unhandled_clues) > 0:
            next_clue, source_suspect = unhandled_clues.popleft()
            newly_solved_suspects = puzzle.add_clue(next_clue, source_suspect)

            for solution in newly_solved_suspects:
                print(f"{solution.name} is {solution.verdict}")

                # the h3's with suspect names have the names as all lower-cased in the DOM;
                # search for that with exact=True to avoid finding another card mentioning that suspect in the clue;
                # also avoids matching another suspect whose name contains the name we're looking for (e.g. "Steve" when looking for "eve")
                search_text = solution.name.lower()
                suspect_element = page.get_by_text(search_text, exact=True)
                # opens modal that allows identifying suspect as Innocent or Criminal
                suspect_element.click()

                # click "Innocent" or "Criminal" button
                page.locator("css=.modal").get_by_text(str(solution.verdict)).click()

                # check that we selected correct verdict - if we didn't, the "Not enough evidence!" warning modal will be displayed
                warning_modals = page.locator("css=.modal.warning").all()
                assert len(warning_modals) == 0, "Incorrect verdict clicked on page"

                # get new clue
                # using get_by_text() with suspect's name will drill down too far, to the <h3> with their name,
                # so instead, re-scan all flipped cards and look for the one with the right name
                # TODO - possibly detect flavor text, don't add it to unhandled_clues
                all_flipped_cards = page.locator("css=div.card.flipped")
                for card in all_flipped_cards.all():
                    card_name = card.locator("css=h3.name").inner_text()
                    if card_name == solution.name:
                        clue = card.locator("css=p.hint").inner_text()
                        unhandled_clues.append((clue, solution.name))
                        print(f"New clue from {solution.name}: {clue}")
                        break

        if puzzle.is_solved():
            print("Puzzle solved!")
        else:
            print("Puzzle unsolved, bug somewhere")
        print()

        context.close()
        browser.close()
