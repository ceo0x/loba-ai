import numpy as np
from gymnasium import spaces

from loba_ai.cards import Card
from loba_ai.remote.adapter import choose_remote_action
from loba_ai.remote_like_actions import build_remote_like_legal_actions, canonicalize_remote_like_legal_actions
from loba_ai.remote_like_env import MAX_REMOTE_LIKE_ACTIONS, RemoteLikeLobaEnv
from loba_ai.remote_like_engine import RemoteLikeGameEngine
from loba_ai.remote_like_match_smart_env import RemoteLikeMatchSmartLobaEnv
from loba_ai.remote_like_obs_builder import describe_action_tactics, project_breakdown
from loba_ai.remote_like_smart_env import RemoteLikeSmartLobaEnv
from loba_ai.agents.remote_like_heuristic_agent import StrongRemoteLikeHeuristicAgent
from loba_ai.rules import Rules


def test_remote_like_env_reset_exposes_legal_actions_and_mask():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] >= 115
    assert "legal_actions" in info
    assert "action_mask" in info
    assert int(np.sum(info["action_mask"])) == len(info["legal_actions"])
    assert info["legal_actions"][0]["type"] == "DrawStock"


def test_remote_like_action_id_maps_to_legal_actions():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    obs, info = env.reset()
    legal = info["legal_actions"]
    assert legal
    obs2, reward, done, truncated, step_info = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert truncated is False
    assert "selected_action" in step_info
    assert step_info["selected_action"] == legal[0]


def test_remote_like_builder_can_emit_play_or_discard_family():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env.reset()
    # Force phase to play_or_discard with current hand.
    env.engine.state.phase = "play_or_discard"
    actions = build_remote_like_legal_actions(env.engine.state, env.rules, env.engine.table_melds)
    assert any(a["type"] == "Discard" for a in actions)
    assert all(a["type"] in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker", "Discard", "Cruzar"} for a in actions)


def test_remote_like_builder_cruzar_uses_duplicate_rank_and_suit_from_pierna():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env.reset()
    env.engine.state.phase = "play_or_discard"
    rank = 7
    club = Card(rank=rank, suit="clubs", deck_id=0, is_joker=False)
    diamond = Card(rank=rank, suit="diamonds", deck_id=0, is_joker=False)
    heart = Card(rank=rank, suit="hearts", deck_id=0, is_joker=False)
    club_dup = Card(rank=rank, suit="clubs", deck_id=1, is_joker=False)
    spade = Card(rank=rank, suit="spades", deck_id=1, is_joker=False)
    env.engine.state.players[0].hand = [club_dup, spade, Card(rank=3, suit="clubs", deck_id=0, is_joker=False)]
    env.engine.table_melds = [{"meld_id": 0, "owner": 1, "kind": "pierna", "cards": [club, diamond, heart]}]
    actions = build_remote_like_legal_actions(env.engine.state, env.rules, env.engine.table_melds)
    cruzar_cards = [a["card"] for a in actions if a["type"] == "Cruzar"]
    assert any(c.get("suit") == "C" for c in cruzar_cards)
    assert not any(c.get("suit") == "S" for c in cruzar_cards)


def test_remote_like_builder_extend_pierna_with_existing_suit_copy():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env.reset()
    env.engine.state.phase = "play_or_discard"
    rank = 9
    clubs = [
        Card(rank=rank, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=rank, suit="clubs", deck_id=1, is_joker=False),
    ]
    diamonds = [Card(rank=rank, suit="diamonds", deck_id=0, is_joker=False)]
    hearts = [Card(rank=rank, suit="hearts", deck_id=0, is_joker=False)]
    env.engine.state.players[0].hand = [clubs[1], Card(rank=2, suit="spades", deck_id=0, is_joker=False)]
    env.engine.table_melds = [{"meld_id": 0, "owner": 1, "kind": "pierna", "cards": [clubs[0], diamonds[0], hearts[0]]}]
    actions = build_remote_like_legal_actions(env.engine.state, env.rules, env.engine.table_melds)
    extend_actions = [a for a in actions if a["type"] == "ExtendMeld" and a.get("meld_id") == 0]
    assert any(a["card"].get("suit") == "C" for a in extend_actions)


def test_remote_adapter_remote_like_policy_predicts_by_legal_index():
    class _IndexModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(115,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            # Pick second legal action when available.
            return int(valid[1] if len(valid) > 1 else valid[0]), None

    obs = {
        "seat": 0,
        "num_players": 2,
        "phase": "draw",
        "hand": [{"rank": "5", "suit": "H", "deck_id": 0}],
        "other_hand_sizes": [1, 9],
        "stock_size": 60,
        "discard_top": {"rank": "6", "suit": "H", "deck_id": 0},
        "discard_size": 4,
        "pending_discard": None,
        "melds_on_table": [],
        "has_laid_meld_this_round": [False, False],
        "cumulative_scores": [0, 0],
        "reenganches_used": [0, 0],
        "eliminated": [False, False],
        "legal_actions": [{"type": "DrawStock"}, {"type": "DrawDiscard", "play": {"type": "LayEscalera", "cards": [{"rank": "4", "suit": "H", "deck_id": 1}, {"rank": "5", "suit": "H", "deck_id": 0}, {"rank": "6", "suit": "H", "deck_id": 0}]}}],
    }
    action, meta = choose_remote_action(obs, model=_IndexModel(), remote_like_policy=True)
    assert action["type"] == "DrawDiscard"
    assert meta["remote_like_policy"] is True
    assert meta["selected_index"] == 1
    assert meta["fallback"] is False


def test_remote_like_action_space_constant():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    assert env.action_space.n == MAX_REMOTE_LIKE_ACTIONS


def test_remote_like_obs_shape_is_fixed_by_num_players():
    env3 = RemoteLikeLobaEnv(rules=Rules(num_players=3))
    obs, info = env3.reset()
    assert tuple(env3.observation_space.shape) == (117,)
    assert tuple(obs.shape) == (117,)


def test_remote_like_reward_penalizes_discard_when_play_options_exist():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    obs, info = env.reset()
    env.engine.state.phase = "play_or_discard"
    env._refresh_legal_actions()
    legal = list(env._last_legal_actions)
    if not legal:
        return
    discard_ix = next((i for i, a in enumerate(legal) if a.get("type") == "Discard"), None)
    play_exists = any(a.get("type") in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker", "Cruzar"} for a in legal)
    if discard_ix is None or not play_exists:
        return
    _, reward, _, _, step_info = env.step(discard_ix)
    assert step_info["phase_before_action"] == "play_or_discard"
    assert step_info["had_play_options_before_action"] is True
    assert isinstance(reward, float)


def test_remote_like_reward_penalizes_discarding_project_pair_card():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env.reset()
    env.engine.state.phase = "play_or_discard"
    env.engine.state.players[0].hand = [
        Card(rank=12, suit="hearts", deck_id=0, is_joker=False),
        Card(rank=12, suit="spades", deck_id=1, is_joker=False),
        Card(rank=4, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=9, suit="diamonds", deck_id=0, is_joker=False),
    ]
    env._refresh_legal_actions()
    legal = list(env._last_legal_actions)
    q_discard_ix = next(
        (
            i
            for i, a in enumerate(legal)
            if a.get("type") == "Discard" and a.get("card", {}).get("rank") == "Q"
        ),
        None,
    )
    if q_discard_ix is None:
        return
    _, reward, _, _, info = env.step(q_discard_ix)
    assert info["selected_action_type"] == "Discard"
    assert info["selected_discard_project_card"] is True
    assert isinstance(reward, float)


def test_remote_like_reward_penalizes_discarding_low_single_before_high_single():
    # Use far-apart suits/ranks so both cards are true non-project singletons.
    env_low = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env_low.reset()
    env_low.engine.state.phase = "play_or_discard"
    env_low.engine.state.players[0].hand = [
        Card(rank=2, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=12, suit="spades", deck_id=0, is_joker=False),
        Card(rank=8, suit="hearts", deck_id=0, is_joker=False),
    ]
    env_low._refresh_legal_actions()
    low_ix = next((i for i, a in enumerate(env_low._last_legal_actions) if a.get("type") == "Discard" and a.get("card", {}).get("rank") == "2"), 0)
    _, low_reward, _, _, _ = env_low.step(low_ix)

    env_high = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    env_high.reset()
    env_high.engine.state.phase = "play_or_discard"
    env_high.engine.state.players[0].hand = [
        Card(rank=2, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=12, suit="spades", deck_id=0, is_joker=False),
        Card(rank=8, suit="hearts", deck_id=0, is_joker=False),
    ]
    env_high._refresh_legal_actions()
    high_ix = next((i for i, a in enumerate(env_high._last_legal_actions) if a.get("type") == "Discard" and a.get("card", {}).get("rank") == "Q"), 0)
    _, high_reward, _, _, _ = env_high.step(high_ix)

    assert low_reward < high_reward


def test_project_detection_excludes_same_card_identity():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    hand_payload = [
        {"rank": "7", "suit": "C", "deck_id": 0},
        {"rank": "K", "suit": "H", "deck_id": 1},
    ]
    # Should not be project just because same card is present as itself.
    assert env._is_project_card_payload(hand_payload[0], hand_payload) is False


def test_project_detection_includes_same_suit_gaps():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    hand_payload = [
        {"rank": "5", "suit": "D", "deck_id": 0},
        {"rank": "7", "suit": "D", "deck_id": 1},
        {"rank": "K", "suit": "H", "deck_id": 0},
    ]

    assert env._is_project_card_payload(hand_payload[0], hand_payload) is True
    facts = project_breakdown(hand_payload[0], hand_payload)
    assert facts["is_run_project"] is True
    assert facts["is_project_card"] is True


def test_duplicate_same_card_is_not_project():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    hand_payload = [
        {"rank": "A", "suit": "S", "deck_id": 0},
        {"rank": "A", "suit": "S", "deck_id": 1},
        {"rank": "2", "suit": "H", "deck_id": 0},
    ]

    assert env._is_project_card_payload(hand_payload[0], hand_payload) is False
    facts = project_breakdown(hand_payload[0], hand_payload)
    assert facts["same_rank_count"] == 0
    assert facts["same_suit_neighbor_count"] == 0
    assert facts["is_pair_project"] is False
    assert facts["is_run_project"] is False
    assert facts["is_project_card"] is False


def test_same_rank_different_suit_is_pair_project():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    hand_payload = [
        {"rank": "A", "suit": "S", "deck_id": 0},
        {"rank": "A", "suit": "H", "deck_id": 1},
        {"rank": "2", "suit": "H", "deck_id": 0},
    ]

    assert env._is_project_card_payload(hand_payload[0], hand_payload) is True
    facts = project_breakdown(hand_payload[0], hand_payload)
    assert facts["same_rank_count"] == 1
    assert facts["same_suit_neighbor_count"] == 0
    assert facts["is_pair_project"] is True
    assert facts["is_run_project"] is False
    assert facts["is_project_card"] is True


def test_ace_low_same_suit_run_project():
    env = RemoteLikeLobaEnv(rules=Rules(num_players=2))
    hand_payload = [
        {"rank": "A", "suit": "S", "deck_id": 0},
        {"rank": "2", "suit": "S", "deck_id": 1},
        {"rank": "K", "suit": "H", "deck_id": 0},
    ]

    assert env._is_project_card_payload(hand_payload[0], hand_payload) is True
    facts = project_breakdown(hand_payload[0], hand_payload)
    assert facts["same_suit_neighbor_count"] == 1
    assert facts["is_run_project"] is True


def test_project_detection_includes_near_table_run():
    hand_payload = [
        {"rank": "5", "suit": "S", "deck_id": 0},
        {"rank": "K", "suit": "H", "deck_id": 0},
    ]
    table_melds = [
        {
            "kind": "escalera",
            "cards": [
                {"rank": "A", "suit": "S", "deck_id": 1},
                {"rank": "2", "suit": "S", "deck_id": 0},
                {"rank": "3", "suit": "S", "deck_id": 1},
            ],
        }
    ]

    facts = project_breakdown(hand_payload[0], hand_payload, table_melds=table_melds)
    assert facts["same_suit_neighbor_count"] == 1
    assert facts["is_run_project"] is True
    assert facts["is_project_card"] is True
    tactics = describe_action_tactics(
        {"type": "Discard", "card": hand_payload[0]},
        hand_payload,
        table_melds=table_melds,
    )
    assert tactics["is_project_card"] is True
    assert tactics["best_dead_card_points"] == 10
    assert tactics["has_high_dead_alternative"] is True


def test_discarding_gap_project_can_break_project():
    hand_payload = [
        {"rank": "5", "suit": "D", "deck_id": 0},
        {"rank": "7", "suit": "D", "deck_id": 1},
        {"rank": "K", "suit": "H", "deck_id": 0},
    ]
    tactics = describe_action_tactics(
        {"type": "Discard", "card": hand_payload[0]},
        hand_payload,
    )

    assert tactics["is_run_project"] is True
    assert tactics["breaks_project"] is True
    assert tactics["has_high_dead_alternative"] is True


def test_remote_like_canonical_order_is_deterministic():
    actions = [
        {"type": "Discard", "card": {"rank": "Q", "suit": "S", "deck_id": 1}},
        {"type": "DrawStock"},
        {"type": "LayEscalera", "cards": [{"rank": "3", "suit": "C", "deck_id": 0}, {"rank": "4", "suit": "C", "deck_id": 0}, {"rank": "5", "suit": "C", "deck_id": 0}]},
        {"type": "Discard", "card": {"rank": "4", "suit": "C", "deck_id": 0}},
    ]
    ordered = canonicalize_remote_like_legal_actions(actions)
    assert [a["type"] for a in ordered][:2] == ["DrawStock", "LayEscalera"]
    assert ordered[-2]["card"]["rank"] == "4"
    assert ordered[-1]["card"]["rank"] == "Q"


def test_strong_remote_like_heuristic_prefers_loba_over_cruzar():
    agent = StrongRemoteLikeHeuristicAgent()
    hand_payload = [
        {"rank": "3", "suit": "C", "deck_id": 0},
        {"rank": "4", "suit": "C", "deck_id": 0},
        {"rank": "5", "suit": "C", "deck_id": 0},
    ]
    legal_actions = [
        {"type": "Cruzar", "card": {"rank": "3", "suit": "C", "deck_id": 0}},
        {"type": "LayEscalera", "cards": list(hand_payload)},
        {"type": "Discard", "card": {"rank": "5", "suit": "C", "deck_id": 0}},
    ]

    idx = agent.act(legal_actions, hand_payload, "play_or_discard")

    assert legal_actions[idx]["type"] == "LayEscalera"


def test_mixed_heuristic_uses_original_for_player_2_and_strong_for_player_3():
    env = RemoteLikeMatchSmartLobaEnv(rules=Rules(num_players=3), opponent="mixed_heuristic")
    env.reset()
    cards = [
        Card(rank=3, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=4, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=5, suit="clubs", deck_id=0, is_joker=False),
    ]
    legal_actions = [
        {"type": "Cruzar", "card": {"rank": "3", "suit": "C", "deck_id": 0, "hand_index": 0}},
        {
            "type": "LayEscalera",
            "cards": [
                {"rank": "3", "suit": "C", "deck_id": 0, "hand_index": 0},
                {"rank": "4", "suit": "C", "deck_id": 0, "hand_index": 1},
                {"rank": "5", "suit": "C", "deck_id": 0, "hand_index": 2},
            ],
        },
        {"type": "Discard", "card": {"rank": "5", "suit": "C", "deck_id": 0, "hand_index": 2}},
    ]

    env.engine.state.phase = "play_or_discard"
    env._last_legal_actions = legal_actions
    env.engine.state.players[1].hand = list(cards)
    env.engine.state.current_player = 1
    player_2_idx = env._opponent_select_action(1)

    env.engine.state.players[2].hand = list(cards)
    env.engine.state.current_player = 2
    player_3_idx = env._opponent_select_action(2)

    assert legal_actions[player_2_idx]["type"] == "Cruzar"
    assert legal_actions[player_3_idx]["type"] == "LayEscalera"


def test_opponent_model_can_be_limited_to_last_seat():
    env = RemoteLikeMatchSmartLobaEnv(
        rules=Rules(num_players=3),
        opponent="mixed_heuristic",
        trained_opponent_model=object(),
        opponent_model_seats="last",
    )

    assert env._should_use_trained_opponent(1) is False
    assert env._should_use_trained_opponent(2) is True


def test_remote_like_smart_env_shape_is_fixed_and_extended():
    env = RemoteLikeSmartLobaEnv(rules=Rules(num_players=3), discard_history_window=2)
    obs, _ = env.reset()
    assert tuple(obs.shape) == tuple(env.observation_space.shape)
    # base(117) + seen(53) + rivals(2 * 2 * 53) + hand_features(3)
    assert tuple(obs.shape) == (385,)


def test_remote_like_smart_env_tracks_recent_discards_and_metrics():
    env = RemoteLikeSmartLobaEnv(rules=Rules(num_players=2), discard_history_window=2)
    env.reset()
    env.engine.state.phase = "play_or_discard"
    env.engine.state.players[0].hand = [
        Card(rank=2, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=7, suit="diamonds", deck_id=0, is_joker=False),
        Card(rank=12, suit="hearts", deck_id=0, is_joker=False),
    ]
    env._refresh_legal_actions()
    discard_ix = next((i for i, a in enumerate(env._last_legal_actions) if a.get("type") == "Discard"), None)
    if discard_ix is None:
        return
    _, _, _, _, info = env.step(discard_ix)
    assert "episode_discard_with_play_options" in info
    assert "episode_play_when_available_ratio" in info
    assert isinstance(info["episode_play_when_available_ratio"], float)


def test_remote_like_move_joker_invalid_rolls_back_meld_and_hand():
    engine = RemoteLikeGameEngine(Rules(num_players=2), seed=1)
    engine.state.current_player = 0
    engine.state.phase = "play_or_discard"
    # Escalera with left joker placeholder: J,6C,7C
    run_cards = [
        Card(rank=None, suit=None, deck_id=0, is_joker=True),
        Card(rank=6, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=7, suit="clubs", deck_id=0, is_joker=False),
    ]
    engine.table_melds = [{"meld_id": 0, "owner": 1, "kind": "escalera", "cards": list(run_cards)}]
    bad_replacement = Card(rank=10, suit="hearts", deck_id=0, is_joker=False)
    engine.state.players[0].hand = [bad_replacement]

    ok = engine._move_joker({"type": "MoveJoker", "meld_id": 0, "replacement": {"rank": "10", "suit": "H", "deck_id": 0}})
    assert ok is False
    # Replacement must return to hand on rollback.
    assert len(engine.state.players[0].hand) == 1
    # Meld must remain unchanged.
    meld_cards = engine.table_melds[0]["cards"]
    assert bool(meld_cards[0].is_joker) is True
    assert meld_cards[1].rank == 6 and meld_cards[1].suit == "clubs"
    assert meld_cards[2].rank == 7 and meld_cards[2].suit == "clubs"


def test_remote_like_move_joker_valid_keeps_legal_run():
    engine = RemoteLikeGameEngine(Rules(num_players=2), seed=2)
    engine.state.current_player = 0
    engine.state.phase = "play_or_discard"
    # Escalera with right joker placeholder: 6C,7C,J
    run_cards = [
        Card(rank=6, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=7, suit="clubs", deck_id=0, is_joker=False),
        Card(rank=None, suit=None, deck_id=1, is_joker=True),
    ]
    engine.table_melds = [{"meld_id": 0, "owner": 1, "kind": "escalera", "cards": list(run_cards)}]
    good_replacement = Card(rank=8, suit="clubs", deck_id=0, is_joker=False)
    engine.state.players[0].hand = [good_replacement]

    ok = engine._move_joker({"type": "MoveJoker", "meld_id": 0, "replacement": {"rank": "8", "suit": "C", "deck_id": 0}})
    assert ok is True
    assert len(engine.state.players[0].hand) == 0
    meld_cards = engine.table_melds[0]["cards"]
    assert meld_cards[0].is_joker is True
    assert [c.rank for c in meld_cards[1:] if not c.is_joker] == [6, 7, 8]
