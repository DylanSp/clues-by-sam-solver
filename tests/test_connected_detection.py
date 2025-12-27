from dataclasses import dataclass
import unittest

from src.models import Verdict


@dataclass
class ConnectedTestCase:
    verdicts: list[Verdict]

    # expected results
    innocents_are_connected: bool
    criminals_are_connected: bool


# connection manually checked
test_cases: list[ConnectedTestCase] = [
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.INNOCENT
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.INNOCENT
        ],
        innocents_are_connected=False,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.INNOCENT
        ],
        innocents_are_connected=False,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=False,
        criminals_are_connected=False
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.INNOCENT
        ],
        innocents_are_connected=False,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.INNOCENT
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.INNOCENT,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=False
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.INNOCENT
        ],
        innocents_are_connected=False,
        criminals_are_connected=False
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=False
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.INNOCENT
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.INNOCENT,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=False
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.INNOCENT
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
    ConnectedTestCase(
        verdicts=[
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL,
            Verdict.CRIMINAL
        ],
        innocents_are_connected=True,
        criminals_are_connected=True
    ),
]


def all_suspects_with_verdict_are_connected(suspects: list[Verdict], verdict: Verdict) -> bool:
    assert len(suspects) == 4

    # only add constraints for suspects in middle (suspects[1] and suspects[2]);
    # check will be vacuously true for suspects on ends

    # check suspects[1]
    suspect_matches_verdict = suspects[1] == verdict
    verdict_exists_to_left = suspects[0] == verdict
    verdict_exists_to_right = suspects[2] == verdict or suspects[3] == verdict
    suspect1_check = suspect_matches_verdict or not (
        verdict_exists_to_left and verdict_exists_to_right)

    # check suspects[2]
    suspect_matches_verdict = suspects[2] == verdict
    verdict_exists_to_left = suspects[0] == verdict or suspects[1] == verdict
    verdict_exists_to_right = suspects[3] == verdict
    suspect2_check = suspect_matches_verdict or not (
        verdict_exists_to_left and verdict_exists_to_right)

    return suspect1_check and suspect2_check


class TestConnectedDetectionLogic(unittest.TestCase):

    def test_connected_detection(self):
        for test_case in test_cases:
            innocents_are_connected = all_suspects_with_verdict_are_connected(
                test_case.verdicts, Verdict.INNOCENT)
            self.assertEqual(innocents_are_connected,
                             test_case.innocents_are_connected,
                             f"Expected: {test_case.innocents_are_connected}, Actual: {innocents_are_connected} for verdicts: {[v.value for v in test_case.verdicts]}")

            criminals_are_connected = all_suspects_with_verdict_are_connected(
                test_case.verdicts, Verdict.CRIMINAL)
            self.assertEqual(criminals_are_connected,
                             test_case.criminals_are_connected,
                             f"Expected: {test_case.criminals_are_connected}, Actual: {criminals_are_connected} for verdicts: {[v.value for v in test_case.verdicts]}")
