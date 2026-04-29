from loba_ai.cards import Card
from loba_ai.hand_display import build_hand_display
from loba_ai.rules import Rules


def test_build_hand_display_keeps_valid_index_permutation():
    hand = [
        Card(rank=4, suit="hearts", deck_id=0),
        Card(rank=9, suit="clubs", deck_id=0),
        Card(rank=5, suit="hearts", deck_id=0),
        Card(rank=7, suit="spades", deck_id=0),
        Card(rank=6, suit="hearts", deck_id=0),
    ]
    slots = build_hand_display(hand, Rules())
    hand_indices = [s["hand_index"] for s in slots]
    assert sorted(hand_indices) == list(range(len(hand)))


def test_build_hand_display_groups_a_possible_run_together():
    hand = [
        Card(rank=8, suit="clubs", deck_id=0),
        Card(rank=4, suit="hearts", deck_id=0),
        Card(rank=5, suit="hearts", deck_id=0),
        Card(rank=8, suit="spades", deck_id=0),
        Card(rank=6, suit="hearts", deck_id=0),
    ]

    slots = build_hand_display(hand, Rules())
    labels = [s["label"] for s in slots]

    run_labels = ["4H", "5H", "6H"]
    positions = sorted(labels.index(label) for label in run_labels)
    assert positions == list(range(min(positions), max(positions) + 1))
