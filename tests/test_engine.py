from loba_ai.cards import Card
from loba_ai.engine import LobaGameEngine
from loba_ai.melds import find_all_melds
from loba_ai.rules import Rules
from loba_ai.state import GameState, PlayerState


def test_engine_can_play_until_terminal():
    engine = LobaGameEngine(seed=1)
    state = engine.reset()

    max_steps = 1000
    steps = 0
    while not state.finished and steps < max_steps:
        if state.phase == "draw":
            action = 0
        elif state.phase == "meld":
            action = 2
        else:
            action = 3 + 24

        res = engine.step(action)
        state = res.state
        steps += 1

    assert state.finished
    assert state.winner is not None


def test_draw_discard_rejected_without_meld_including_top():
    rules = Rules(must_meld_if_draw_discard=True)
    engine = LobaGameEngine(rules=rules, seed=0)
    top = Card(rank=9, suit="hearts", deck_id=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=2, suit="clubs", deck_id=0),
        Card(rank=3, suit="diamonds", deck_id=0),
        Card(rank=4, suit="spades", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=5, suit="hearts", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=6, suit="clubs", deck_id=1)],
        discard_pile=[top],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="draw",
    )
    res = engine.step(1, is_agent_player=True)
    assert res.info.get("invalid_action")
    assert res.state.phase == "draw"
    assert res.state.discard_pile[-1] is top
    assert len(res.state.players[0].hand) == 3


def test_draw_discard_auto_melds_and_goes_to_discard_phase():
    rules = Rules(must_meld_if_draw_discard=True)
    engine = LobaGameEngine(rules=rules, seed=0)
    top = Card(rank=4, suit="diamonds", deck_id=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=4, suit="clubs", deck_id=0),
        Card(rank=4, suit="hearts", deck_id=0),
        Card(rank=10, suit="spades", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=5, suit="hearts", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=6, suit="clubs", deck_id=1)],
        discard_pile=[top],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="draw",
    )
    res = engine.step(1, is_agent_player=True)
    assert not res.info.get("invalid_action")
    assert res.info.get("discard_auto_meld")
    assert res.state.phase == "discard"
    assert len(res.state.players[0].hand) == 1
    assert len(res.state.players[0].melds) == 1
    assert top not in res.state.players[0].hand


def test_draw_discard_allowed_without_meld_when_rule_disabled():
    rules = Rules(must_meld_if_draw_discard=False)
    engine = LobaGameEngine(rules=rules, seed=0)
    top = Card(rank=9, suit="hearts", deck_id=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=2, suit="clubs", deck_id=0),
        Card(rank=3, suit="diamonds", deck_id=0),
        Card(rank=4, suit="spades", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=5, suit="hearts", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=6, suit="clubs", deck_id=1)],
        discard_pile=[top],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="draw",
    )
    res = engine.step(1, is_agent_player=True)
    assert not res.info.get("invalid_action")
    assert res.state.phase == "meld"
    assert top in res.state.players[0].hand


def test_can_play_multiple_melds_before_discard():
    engine = LobaGameEngine(rules=Rules(), seed=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=4, suit="clubs", deck_id=0),
        Card(rank=4, suit="diamonds", deck_id=0),
        Card(rank=4, suit="hearts", deck_id=0),
        Card(rank=7, suit="clubs", deck_id=0),
        Card(rank=7, suit="diamonds", deck_id=0),
        Card(rank=7, suit="hearts", deck_id=0),
        Card(rank=10, suit="spades", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=9, suit="clubs", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=5, suit="spades", deck_id=1)],
        discard_pile=[Card(rank=8, suit="spades", deck_id=1)],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="meld",
    )

    first_action = 3
    res1 = engine.step(first_action, is_agent_player=True)
    assert not res1.info.get("invalid_action")
    assert res1.state.phase == "meld"

    remaining = find_all_melds(engine.state.players[0].hand, engine.rules, max_results=24)
    assert remaining, "Expected second meld to still be available"
    second_action = 3
    res2 = engine.step(second_action, is_agent_player=True)
    assert not res2.info.get("invalid_action")
    assert res2.state.phase == "meld"
    assert len(res2.state.players[0].melds) >= 2
    assert len(res2.state.players[0].hand) == 1

    skip_to_discard = engine.step(2, is_agent_player=True)
    assert skip_to_discard.state.phase == "discard"


def test_player_wins_immediately_if_meld_leaves_empty_hand():
    engine = LobaGameEngine(rules=Rules(), seed=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=4, suit="clubs", deck_id=0),
        Card(rank=4, suit="diamonds", deck_id=0),
        Card(rank=4, suit="hearts", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=9, suit="clubs", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=5, suit="spades", deck_id=1)],
        discard_pile=[Card(rank=8, suit="spades", deck_id=1)],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="meld",
    )

    res = engine.step(3, is_agent_player=True)
    assert res.done is True
    assert res.state.finished is True
    assert res.state.winner == 0
    assert res.info.get("win_after_meld") is True


def test_player_wins_after_discard_take_auto_meld_leaves_empty_hand():
    rules = Rules(must_meld_if_draw_discard=True)
    engine = LobaGameEngine(rules=rules, seed=0)
    top = Card(rank=4, suit="diamonds", deck_id=0)
    p0 = PlayerState()
    p0.hand = [
        Card(rank=4, suit="clubs", deck_id=0),
        Card(rank=4, suit="hearts", deck_id=0),
    ]
    p1 = PlayerState()
    p1.hand = [Card(rank=5, suit="hearts", deck_id=1) for _ in range(9)]
    engine.state = GameState(
        players=[p0, p1],
        current_player=0,
        stock_pile=[Card(rank=6, suit="clubs", deck_id=1)],
        discard_pile=[top],
        melds_on_table=[],
        round_index=0,
        turn_number=0,
        finished=False,
        winner=None,
        phase="draw",
    )

    res = engine.step(1, is_agent_player=True)
    assert res.done is True
    assert res.state.finished is True
    assert res.state.winner == 0
    assert res.info.get("win_after_meld") is True
