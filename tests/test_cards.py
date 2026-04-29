from loba_ai.cards import Card, hand_points


def test_card_points_follow_custom_rules():
    assert Card(rank=0, suit=None, deck_id=0, is_joker=True).points == 15
    assert Card(rank=1, suit="spades", deck_id=0).points == 11
    assert Card(rank=11, suit="hearts", deck_id=0).points == 10
    assert Card(rank=12, suit="diamonds", deck_id=0).points == 10
    assert Card(rank=13, suit="clubs", deck_id=0).points == 10
    assert Card(rank=10, suit="clubs", deck_id=0).points == 10
    assert Card(rank=7, suit="clubs", deck_id=0).points == 7


def test_hand_points_uses_card_points():
    cards = [
        Card(rank=0, suit=None, deck_id=0, is_joker=True),  # 15
        Card(rank=1, suit="spades", deck_id=0),  # 11
        Card(rank=12, suit="diamonds", deck_id=0),  # 10
        Card(rank=8, suit="hearts", deck_id=0),  # 8
    ]
    assert hand_points(cards) == 44
