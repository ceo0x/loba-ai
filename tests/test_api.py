from fastapi.testclient import TestClient
import json
import time
import numpy as np
from gymnasium import spaces

from loba_ai.api import app
from loba_ai.cards import Card
from loba_ai.melds import is_valid_run
from loba_ai.rules import Rules


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_match_human_vs_heuristic():
    r = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "human", "model_name": None},
                {"kind": "heuristic", "model_name": None},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "match_id" in data
    assert "state" in data
    assert len(data["state"]["players"]) == 2
    assert "discard_last_three" in data["state"]
    assert isinstance(data["state"]["discard_last_three"], list)
    assert "table_melds_grouped" in data["state"]
    assert len(data["state"]["table_melds_grouped"]) == 2


def test_delete_unknown_model_returns_404():
    r = client.delete("/models/not-found-model")
    assert r.status_code == 404


def test_quack_asset_available():
    r = client.get("/assets/quack.mp3")
    assert r.status_code == 200


def test_create_match_supports_six_players():
    r = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "human", "model_name": None},
                {"kind": "heuristic", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "heuristic", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "heuristic", "model_name": None},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["state"]["players"]) == 6


def test_create_match_rejects_more_than_six_players():
    r = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
            ]
        },
    )
    assert r.status_code == 400


def test_state_has_winner_detail_when_finished():
    create = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "random", "model_name": None},
                {"kind": "random", "model_name": None},
            ]
        },
    )
    assert create.status_code == 200
    data = create.json()
    match_id = data["match_id"]
    state = data["state"]

    for _ in range(20):
        if state.get("finished"):
            break
        r = client.post(f"/matches/{match_id}/autoplay")
        assert r.status_code == 200
        state = r.json()["state"]

    assert state["finished"] is True
    assert state["winner"] is not None
    assert state["winner_detail"] is not None
    assert state["winner_detail"]["index"] == state["winner"]
    assert "kind" in state["winner_detail"]


def test_create_match_with_autoplay_enabled_sets_flag_and_runs():
    r = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "human", "model_name": None},
                {"kind": "random", "model_name": None},
            ],
            "autoplay_enabled": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["state"]["autoplay_enabled"] is True
    assert isinstance(data["autoplay_log"], list)
    assert len(data["autoplay_log"]) > 0


def test_action_returns_400_when_model_obs_is_incompatible(monkeypatch):
    create = client.post(
        "/matches",
        json={
            "players": [
                {"kind": "human", "model_name": None},
                {"kind": "heuristic", "model_name": None},
            ]
        },
    )
    assert create.status_code == 200
    match_id = create.json()["match_id"]

    from loba_ai.api import service

    match = service.get_match(match_id)

    def _boom(*args, **kwargs):
        raise ValueError("Model observation mismatch: model expects 122, got 119.")

    monkeypatch.setattr(match, "autoplay_until_human_or_end", _boom)

    state = client.get(f"/matches/{match_id}").json()["state"]
    action_id = state["valid_actions"][0]["id"]
    r = client.post(f"/matches/{match_id}/action", json={"action": action_id})
    assert r.status_code == 400
    assert "Model observation mismatch" in r.json()["detail"]


def test_action_with_4p_model_in_2p_match_uses_adapter():
    from loba_ai.api import service
    from loba_ai.playground.service import RegisteredModel

    class _Dummy4PModel:
        def __init__(self) -> None:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(122,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            assert tuple(obs.shape) == (122,)
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["dummy4p"] = RegisteredModel(name="dummy4p", path="dummy", model=_Dummy4PModel())
    try:
        create = client.post(
            "/matches",
            json={
                "players": [
                    {"kind": "human", "model_name": None},
                    {"kind": "model", "model_name": "dummy4p"},
                ]
            },
        )
        assert create.status_code == 200
        match_id = create.json()["match_id"]
        state = client.get(f"/matches/{match_id}").json()["state"]
        action_id = state["valid_actions"][0]["id"]
        r = client.post(f"/matches/{match_id}/action", json={"action": action_id})
        assert r.status_code == 200
    finally:
        service.models.pop("dummy4p", None)


def test_arena_requires_registered_host_model():
    r = client.post("/arena/rooms", json={"host_model_name": "missing-model"})
    assert r.status_code == 400


def test_arena_state_masks_other_player_cards():
    from loba_ai.api import service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["arena-host"] = RegisteredModel(name="arena-host", path="dummy", model=_DummyModel())
    try:
        create = client.post("/arena/rooms", json={"host_model_name": "arena-host", "target_points": 100})
        assert create.status_code == 200
        room_id = create.json()["room"]["room_id"]
        host_token = create.json()["host_player_token"]

        join = client.post(f"/arena/rooms/{room_id}/join", json={"guest_name": "bro"})
        assert join.status_code == 200
        guest_token = join.json()["guest_player_token"]

        host_ready = client.post(f"/arena/rooms/{room_id}/ready", json={"player_token": host_token})
        assert host_ready.status_code == 200
        guest_ready = client.post(f"/arena/rooms/{room_id}/ready", json={"player_token": guest_token})
        assert guest_ready.status_code == 200
        assert guest_ready.json()["room"]["status"] == "in_game"
        assert guest_ready.json()["room"]["target_points"] == 100

        guest_state = client.get(f"/arena/rooms/{room_id}/state", params={"player_token": guest_token})
        assert guest_state.status_code == 200
        players = guest_state.json()["state"]["players"]
        assert len(players[0]["hand_full"]) == 0
        assert len(players[1]["hand_full"]) == players[1]["cards_in_hand"]
    finally:
        service.models.pop("arena-host", None)


def test_arena_series_finishes_when_target_points_reached():
    from loba_ai.api import service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["arena-series-host"] = RegisteredModel(name="arena-series-host", path="dummy", model=_DummyModel())
    try:
        create = client.post("/arena/rooms", json={"host_model_name": "arena-series-host", "target_points": 10})
        assert create.status_code == 200
        room_id = create.json()["room"]["room_id"]
        host_token = create.json()["host_player_token"]

        join = client.post(f"/arena/rooms/{room_id}/join", json={"guest_name": "bro"})
        assert join.status_code == 200
        guest_token = join.json()["guest_player_token"]

        assert client.post(f"/arena/rooms/{room_id}/ready", json={"player_token": host_token}).status_code == 200
        assert client.post(f"/arena/rooms/{room_id}/ready", json={"player_token": guest_token}).status_code == 200

        for _ in range(3000):
            state_r = client.get(f"/arena/rooms/{room_id}/state", params={"player_token": guest_token})
            assert state_r.status_code == 200
            payload = state_r.json()
            room = payload["room"]
            if room["match_finished"]:
                assert room["champion"] in [0, 1]
                assert max(room["scores"]) >= 10
                break

            state = payload["state"]
            valid = state["valid_actions"] if state else []
            if valid:
                action_id = valid[0]["id"]
                play_r = client.post(
                    f"/arena/rooms/{room_id}/action",
                    json={"player_token": guest_token, "action": action_id},
                )
                assert play_r.status_code == 200
        else:
            raise AssertionError("Arena series did not finish within step budget")
    finally:
        service.models.pop("arena-series-host", None)


class _FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._queue = [json.dumps(m) for m in messages]
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        if self._queue:
            return self._queue.pop(0)
        raise TimeoutError()

    async def close(self, code=1000, reason="") -> None:
        self.closed = True


class _FakeConnectCtx:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_remote_session_lifecycle_and_actions(monkeypatch):
    from loba_ai.api import remote_manager, service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(115,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["remote-host"] = RegisteredModel(name="remote-host", path="dummy", model=_DummyModel())

    fake_ws = _FakeWebSocket(
        [
            {"type": "registered", "name": "brother-bot", "seat": 2},
            {
                "type": "observation",
                "obs": {
                    "seat": 2,
                    "num_players": 3,
                    "phase": "draw",
                    "hand": [{"rank": "5", "suit": "H", "deck_id": 0}],
                    "other_hand_sizes": [9, 9, 1],
                    "stock_size": 60,
                    "discard_top": {"rank": "6", "suit": "H", "deck_id": 0},
                    "discard_size": 4,
                    "pending_discard": None,
                    "melds_on_table": [],
                    "has_laid_meld_this_round": [False, False, False],
                    "cumulative_scores": [0, 0, 0],
                    "reenganches_used": [0, 0, 0],
                    "eliminated": [False, False, False],
                    "legal_actions": [{"type": "DrawStock"}, {"type": "DrawDiscard", "play": {"type": "LayEscalera", "cards": [{"rank": "4", "suit": "H", "deck_id": 1}, {"rank": "5", "suit": "H", "deck_id": 0}, {"rank": "6", "suit": "H", "deck_id": 0}]}}],
                },
            },
            {"type": "round_end", "winner": 1, "penalties": [0, 3, 7], "cumulative": [0, 3, 7]},
            {"type": "match_end", "winner": 1},
        ]
    )

    monkeypatch.setattr("loba_ai.remote.service.websockets.connect", lambda *args, **kwargs: _FakeConnectCtx(fake_ws))

    try:
        create = client.post(
            "/remote/sessions",
            json={
                "url": "ws://example:8765/",
                "name": "brother-bot",
                "model_name": "remote-host",
            },
        )
        assert create.status_code == 200
        session_id = create.json()["session"]["session_id"]

        start = client.post(f"/remote/sessions/{session_id}/start")
        assert start.status_code == 200

        for _ in range(50):
            snapshot = client.get(f"/remote/sessions/{session_id}")
            assert snapshot.status_code == 200
            if snapshot.json()["session"]["status"] in {"finished", "stopped"}:
                break
            time.sleep(0.02)

        snapshot = client.get(f"/remote/sessions/{session_id}")
        session = snapshot.json()["session"]
        assert session["observations_seen"] >= 1
        assert session["actions_sent"] >= 1
        assert session["last_match_end"] is not None
        assert session["decision_metrics"]["model_decisions"] >= 1

        sent_types = [m.get("type") for m in fake_ws.sent]
        assert "register" in sent_types
        assert "act" in sent_types

        events = client.get(f"/remote/sessions/{session_id}/events")
        assert events.status_code == 200
        items = events.json()["items"]
        assert isinstance(items, list)
        action_events = [ev for ev in items if ev.get("type") == "action_sent"]
        assert action_events
        assert "hand" in action_events[0]["data"]
    finally:
        service.models.pop("remote-host", None)


def test_remote_create_session_rejects_unknown_model():
    r = client.post(
        "/remote/sessions",
        json={"url": "ws://example:8765/", "name": "brother-bot", "model_name": "missing"},
    )
    assert r.status_code == 400


def test_remote_3p_observation_adapts_to_4p_model_shape():
    from loba_ai.remote.adapter import choose_remote_action

    class _Dummy4PObsModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(119,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            assert tuple(obs.shape) == (119,)
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    obs = {
        "seat": 2,
        "num_players": 3,
        "phase": "draw",
        "hand": [{"rank": "5", "suit": "H", "deck_id": 0}],
        "other_hand_sizes": [9, 9, 1],
        "stock_size": 60,
        "discard_top": {"rank": "6", "suit": "H", "deck_id": 0},
        "discard_size": 4,
        "pending_discard": None,
        "melds_on_table": [],
        "has_laid_meld_this_round": [False, False, False],
        "cumulative_scores": [0, 0, 0],
        "reenganches_used": [0, 0, 0],
        "eliminated": [False, False, False],
        "legal_actions": [{"type": "DrawStock"}],
    }

    action, meta = choose_remote_action(obs, model=_Dummy4PObsModel())
    assert action["type"] == "DrawStock"
    assert meta["used_model"] is True


def test_remote_adapter_emits_go_out_metrics_when_available():
    from loba_ai.remote.adapter import choose_remote_action

    obs = {
        "seat": 0,
        "num_players": 2,
        "phase": "play_or_discard",
        "hand": [
            {"rank": "3", "suit": "C", "deck_id": 0},
            {"rank": "4", "suit": "C", "deck_id": 0},
            {"rank": "5", "suit": "C", "deck_id": 0},
        ],
        "other_hand_sizes": [3, 9],
        "stock_size": 50,
        "discard_top": {"rank": "9", "suit": "H", "deck_id": 0},
        "discard_size": 4,
        "pending_discard": None,
        "melds_on_table": [],
        "has_laid_meld_this_round": [True, False],
        "cumulative_scores": [0, 0],
        "reenganches_used": [0, 0],
        "eliminated": [False, False],
        "legal_actions": [
            {"type": "LayEscalera", "cards": [{"rank": "3", "suit": "C", "deck_id": 0}, {"rank": "4", "suit": "C", "deck_id": 0}, {"rank": "5", "suit": "C", "deck_id": 0}]},
            {"type": "Discard", "card": {"rank": "3", "suit": "C", "deck_id": 0}},
        ],
    }
    action, meta = choose_remote_action(obs, model=None)
    assert action["type"] == "LayEscalera"
    assert meta["go_out_available"] is True
    assert meta["selected_go_out_action"] is True
    assert meta["discard_while_go_out_available"] is False


def test_remote_session_exposes_decision_metrics(monkeypatch):
    from loba_ai.api import remote_manager, service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(115,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["remote-metrics"] = RegisteredModel(name="remote-metrics", path="dummy", model=_DummyModel())

    fake_ws = _FakeWebSocket(
        [
            {"type": "registered", "name": "bro", "seat": 0},
            {
                "type": "observation",
                "obs": {
                    "seat": 0,
                    "num_players": 2,
                    "phase": "play_or_discard",
                    "hand": [
                        {"rank": "3", "suit": "C", "deck_id": 0},
                        {"rank": "4", "suit": "C", "deck_id": 0},
                        {"rank": "5", "suit": "C", "deck_id": 0},
                    ],
                    "other_hand_sizes": [3, 9],
                    "stock_size": 50,
                    "discard_top": {"rank": "9", "suit": "H", "deck_id": 0},
                    "discard_size": 4,
                    "pending_discard": None,
                    "melds_on_table": [],
                    "has_laid_meld_this_round": [True, False],
                    "cumulative_scores": [0, 0],
                    "reenganches_used": [0, 0],
                    "eliminated": [False, False],
                    "legal_actions": [
                        {"type": "LayEscalera", "cards": [{"rank": "3", "suit": "C", "deck_id": 0}, {"rank": "4", "suit": "C", "deck_id": 0}, {"rank": "5", "suit": "C", "deck_id": 0}]},
                        {"type": "Discard", "card": {"rank": "3", "suit": "C", "deck_id": 0}},
                    ],
                },
            },
            {"type": "match_end", "winner": 0},
        ]
    )
    monkeypatch.setattr("loba_ai.remote.service.websockets.connect", lambda *args, **kwargs: _FakeConnectCtx(fake_ws))

    try:
        create = client.post(
            "/remote/sessions",
            json={"url": "ws://example:8765/", "name": "bro", "model_name": "remote-metrics"},
        )
        session_id = create.json()["session"]["session_id"]
        assert client.post(f"/remote/sessions/{session_id}/start").status_code == 200

        for _ in range(50):
            snapshot = client.get(f"/remote/sessions/{session_id}")
            if snapshot.json()["session"]["status"] in {"finished", "stopped"}:
                break
            time.sleep(0.02)

        session = client.get(f"/remote/sessions/{session_id}").json()["session"]
        metrics = session["decision_metrics"]
        assert metrics["model_decisions"] >= 1
        assert metrics["go_out_available_observations"] >= 1
    finally:
        service.models.pop("remote-metrics", None)


def test_remote_session_move_joker_guard_forces_discard(monkeypatch):
    from loba_ai.api import service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(117,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["remote-guard"] = RegisteredModel(name="remote-guard", path="dummy", model=_DummyModel())

    base_obs = {
        "seat": 0,
        "num_players": 3,
        "phase": "play_or_discard",
        "hand": [
            {"rank": "K", "suit": "H", "deck_id": 1},
            {"rank": "A", "suit": "H", "deck_id": 0},
            {"rank": "2", "suit": "H", "deck_id": 1},
        ],
        "other_hand_sizes": [3, 9, 9],
        "stock_size": 50,
        "discard_top": {"rank": "9", "suit": "H", "deck_id": 0},
        "discard_size": 4,
        "pending_discard": None,
        "melds_on_table": [],
        "has_laid_meld_this_round": [True, False, False],
        "cumulative_scores": [0, 0, 0],
        "reenganches_used": [0, 0, 0],
        "eliminated": [False, False, False],
        "legal_actions": [
            {"type": "MoveJoker", "meld_id": 0, "replacement": {"rank": "K", "suit": "H", "deck_id": 1, "hand_index": 0}},
            {"type": "Discard", "card": {"rank": "A", "suit": "H", "deck_id": 0, "hand_index": 1}},
        ],
    }
    fake_ws = _FakeWebSocket(
        [
            {"type": "registered", "name": "bro", "seat": 0},
            {"type": "observation", "obs": dict(base_obs)},
            {"type": "observation", "obs": dict(base_obs)},
            {"type": "observation", "obs": dict(base_obs)},
            {"type": "match_end", "winner": 0},
        ]
    )

    def _always_move_joker(obs, model=None, remote_like_policy=False, **kwargs):
        return (
            {"type": "MoveJoker", "meld_id": 0, "replacement": {"rank": "K", "suit": "H", "deck_id": 1, "hand_index": 0}},
            {
                "selected_index": 0,
                "used_model": True,
                "remote_like_policy": True,
                "predicted_local_action_id": 0,
                "fallback": False,
                "phase": "play_or_discard",
                "action_type_selected": "MoveJoker",
                "legal_action_type_counts": {"MoveJoker": 1, "Discard": 1},
            },
        )

    monkeypatch.setattr("loba_ai.remote.service.websockets.connect", lambda *args, **kwargs: _FakeConnectCtx(fake_ws))
    monkeypatch.setattr("loba_ai.remote.service.choose_remote_action", _always_move_joker)

    try:
        create = client.post(
            "/remote/sessions",
            json={"url": "ws://example:8765/", "name": "bro", "model_name": "remote-guard", "remote_like_policy": True},
        )
        assert create.status_code == 200
        session_id = create.json()["session"]["session_id"]
        assert client.post(f"/remote/sessions/{session_id}/start").status_code == 200

        for _ in range(60):
            snapshot = client.get(f"/remote/sessions/{session_id}")
            assert snapshot.status_code == 200
            if snapshot.json()["session"]["status"] in {"finished", "stopped"}:
                break
            time.sleep(0.02)

        snapshot = client.get(f"/remote/sessions/{session_id}")
        metrics = snapshot.json()["session"]["decision_metrics"]
        assert metrics["move_joker_guard_forced_discard"] >= 1

        act_messages = [m for m in fake_ws.sent if m.get("type") == "act"]
        assert len(act_messages) >= 3
        # First actions may stay MoveJoker, then guard should force Discard.
        assert any(m.get("action", {}).get("type") == "Discard" for m in act_messages)
    finally:
        service.models.pop("remote-guard", None)


def test_remote_mock_lifecycle():
    stop_before = client.post("/remote/mock/stop")
    assert stop_before.status_code == 200

    start = client.post(
        "/remote/mock/start",
        json={"host": "127.0.0.1", "port": 0, "num_players": 3, "remote_name": "brother-bot", "seed": 13},
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["mock"]["running"] is True
    assert start_payload["mock"]["num_players"] == 3
    assert start_payload["mock"]["host"] == "127.0.0.1"
    assert isinstance(start_payload["mock"]["port"], int)
    assert start_payload["mock"]["port"] > 0

    status = client.get("/remote/mock/status")
    assert status.status_code == 200
    assert status.json()["mock"]["running"] is True

    stop = client.post("/remote/mock/stop")
    assert stop.status_code == 200
    assert stop.json()["mock"]["running"] is False


def test_remote_session_can_connect_to_mock_server():
    from loba_ai.api import service
    from loba_ai.playground.service import RegisteredModel

    class _DummyModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(117,), dtype=np.float32)

        def predict(self, obs, deterministic=True, action_masks=None):
            valid = np.flatnonzero(action_masks)
            return int(valid[0]), None

    service.models["remote-mock-host"] = RegisteredModel(name="remote-mock-host", path="dummy", model=_DummyModel())
    mock_started = False
    session_id: str | None = None
    last_observation: dict | None = None

    def _card_from_payload(payload: dict) -> Card:
        if payload.get("joker"):
            return Card(rank=None, suit=None, deck_id=int(payload.get("deck_id", 0)), is_joker=True)
        rank_raw = str(payload.get("rank", "0"))
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
        if rank_raw in rank_map:
            rank = rank_map[rank_raw]
        else:
            rank = int(rank_raw)
        suit_map = {"S": "spades", "H": "hearts", "C": "clubs", "D": "diamonds"}
        suit = suit_map.get(str(payload.get("suit", "C")), "clubs")
        return Card(rank=rank, suit=suit, deck_id=int(payload.get("deck_id", 0)), is_joker=False)

    try:
        start = client.post(
            "/remote/mock/start",
            json={"host": "127.0.0.1", "port": 0, "num_players": 3, "remote_name": "brother-bot", "seed": 17},
        )
        assert start.status_code == 200
        mock_started = True
        ws_url = start.json()["ws_url"]

        create = client.post(
            "/remote/sessions",
            json={
                "url": ws_url,
                "name": "brother-bot",
                "model_name": "remote-mock-host",
                "remote_like_policy": True,
            },
        )
        assert create.status_code == 200
        session_id = create.json()["session"]["session_id"]
        assert client.post(f"/remote/sessions/{session_id}/start").status_code == 200

        saw_action = False
        for _ in range(100):
            snap = client.get(f"/remote/sessions/{session_id}")
            assert snap.status_code == 200
            session = snap.json()["session"]
            if isinstance(session.get("last_observation"), dict):
                last_observation = session["last_observation"]
            if session["actions_sent"] >= 1 and session["observations_seen"] >= 1:
                saw_action = True
                break
            time.sleep(0.05)
        assert saw_action
        if last_observation and isinstance(last_observation.get("melds_on_table"), list):
            rules = Rules(num_players=3)
            for meld in last_observation["melds_on_table"]:
                if meld.get("kind") != "escalera":
                    continue
                cards_payload = meld.get("cards", [])
                cards = [_card_from_payload(c) for c in cards_payload if isinstance(c, dict)]
                assert is_valid_run(cards, rules), f"Invalid escalera in mock observation: {meld}"
    finally:
        if session_id:
            client.post(f"/remote/sessions/{session_id}/stop")
        if mock_started:
            client.post("/remote/mock/stop")
        service.models.pop("remote-mock-host", None)
