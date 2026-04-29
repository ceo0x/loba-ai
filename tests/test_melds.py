from loba_ai.cards import Card
from loba_ai.melds import discard_take_melds, is_valid_group, is_valid_run
from loba_ai.rules import Rules


def test_group_with_joker_is_invalid_by_default():
    rules = Rules()
    cards = [
        Card(rank=7, suit="hearts", deck_id=0),
        Card(rank=7, suit="clubs", deck_id=0),
        Card(rank=0, suit=None, deck_id=0, is_joker=True),
    ]
    assert not is_valid_group(cards, rules)


def test_valid_run_with_gap_and_joker():
    rules = Rules()
    cards = [
        Card(rank=5, suit="spades", deck_id=0),
        Card(rank=7, suit="spades", deck_id=0),
        Card(rank=0, suit=None, deck_id=0, is_joker=True),
    ]
    assert is_valid_run(cards, rules)


def test_group_with_joker_can_be_enabled():
    rules = Rules(allow_jokers_in_group=True)
    cards = [
        Card(rank=7, suit="hearts", deck_id=0),
        Card(rank=7, suit="clubs", deck_id=0),
        Card(rank=0, suit=None, deck_id=0, is_joker=True),
    ]
    assert is_valid_group(cards, rules)


def test_discard_take_melds_empty_when_top_cannot_form_meld():
    rules = Rules()
    top = Card(rank=5, suit="hearts", deck_id=0)
    hand = [
        Card(rank=2, suit="clubs", deck_id=0),
        Card(rank=3, suit="diamonds", deck_id=0),
        Card(rank=4, suit="spades", deck_id=0),
    ]
    assert discard_take_melds(hand, top, rules) == []


def test_discard_take_melds_finds_group_including_top():
    rules = Rules()
    top = Card(rank=8, suit="hearts", deck_id=0)
    hand = [
        Card(rank=8, suit="clubs", deck_id=0),
        Card(rank=8, suit="spades", deck_id=0),
    ]
    melds = discard_take_melds(hand, top, rules)
    assert len(melds) >= 1
    assert top in melds[0].cards
    assert melds[0].kind == "group"
