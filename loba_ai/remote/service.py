from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

import websockets

from loba_ai.playground.service import RegisteredModel
from loba_ai.remote.adapter import choose_remote_action
from loba_ai.remote_like_obs_builder import SmartObsMemory, card_token

MOVE_JOKER_GUARD_MAX_STREAK = 2


def _rank_to_value(rank: Any) -> int:
    rank_map = {"A": 14, "J": 11, "Q": 12, "K": 13}
    if isinstance(rank, str) and rank in rank_map:
        return rank_map[rank]
    try:
        return int(rank)
    except Exception:
        return 0


def _pick_guard_discard_action(legal_actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    discards = [a for a in legal_actions if str(a.get("type")) == "Discard" and isinstance(a.get("card"), dict)]
    if not discards:
        return None

    def _discard_key(action: dict[str, Any]) -> tuple[int, int]:
        card = action.get("card", {})
        is_joker = 1 if bool(card.get("joker")) else 0
        # Prefer non-joker discard for guard; among naturals, discard higher rank first.
        return (is_joker, -_rank_to_value(card.get("rank")))

    return sorted(discards, key=_discard_key)[0]


@dataclass(slots=True)
class RemoteSessionConfig:
    url: str
    name: str
    token: str | None = None
    model_name: str | None = None
    remote_like_policy: bool = False
    action_delay_seconds: float = 0.0  # async sleep before sending each action — useful for spectating


@dataclass(slots=True)
class RemoteSessionEvent:
    ts: float
    type: str
    data: dict[str, Any]


@dataclass(slots=True)
class RemoteSession:
    session_id: str
    config: RemoteSessionConfig
    status: str = "created"
    seat: int | None = None
    connected: bool = False
    running: bool = False
    stop_requested: bool = False
    last_error: str | None = None
    last_observation: dict[str, Any] | None = None
    last_round_end: dict[str, Any] | None = None
    last_match_end: dict[str, Any] | None = None
    events: deque[RemoteSessionEvent] = field(default_factory=lambda: deque(maxlen=500))
    started_at: float | None = None
    stopped_at: float | None = None
    actions_sent: int = 0
    observations_seen: int = 0
    decision_metrics: dict[str, int] = field(
        default_factory=lambda: {
            "model_decisions": 0,
            "fallback_decisions": 0,
            "go_out_available_observations": 0,
            "go_out_selected": 0,
            "go_out_missed": 0,
            "discard_selected": 0,
            "discard_while_go_out_available": 0,
            "lay_escalera_selected": 0,
            "lay_pierna_selected": 0,
            "extend_meld_selected": 0,
            "move_joker_selected": 0,
            "cruzar_selected": 0,
            "move_joker_guard_forced_discard": 0,
        }
    )

    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _move_joker_streak: int = 0
    # Cross-step memory the deployment adapter needs to build the smart obs the
    # same way the training env does (seen cards, recent discards, own actions, match state).
    obs_memory: SmartObsMemory | None = None
    _last_top_discard_token: int | None = None

    def log(self, event_type: str, **data: Any) -> None:
        with self._lock:
            self.events.appendleft(RemoteSessionEvent(ts=time.time(), type=event_type, data=data))

    # ===== Smart-obs memory maintenance =====
    def _ensure_obs_memory(self, observation: dict[str, Any]) -> SmartObsMemory:
        """Initialize obs_memory on first observation (we only know num_players at that point)."""
        n = max(2, int(observation.get("num_players", 3)))
        if self.obs_memory is None or self.obs_memory.num_players != n:
            self.obs_memory = SmartObsMemory(
                num_players=n,
                discard_history_window=2,
                target_points=100,
                max_reenganches=2,
            )
            # Reset arrays to the right size.
            import numpy as _np
            self.obs_memory.match_scores = _np.zeros(n, dtype=_np.float32)
            self.obs_memory.reenganches_used = _np.zeros(n, dtype=_np.int32)
            self.obs_memory.eliminated = _np.zeros(n, dtype=bool)
            self.obs_memory.recent_discards_by_player = [[] for _ in range(n)]
            self._last_top_discard_token = None
        return self.obs_memory

    def update_obs_memory_from_observation(self, observation: dict[str, Any]) -> None:
        memory = self._ensure_obs_memory(observation)
        n = memory.num_players
        my_seat = int(observation.get("seat", 0))

        # Match-level fields if the server provides them.
        cumulative = observation.get("cumulative_scores")
        if isinstance(cumulative, list):
            for i in range(min(n, len(cumulative))):
                try:
                    memory.match_scores[i] = float(cumulative[i] or 0)
                except (TypeError, ValueError):
                    pass
        reeng = observation.get("reenganches_used")
        if isinstance(reeng, list):
            for i in range(min(n, len(reeng))):
                try:
                    memory.reenganches_used[i] = int(reeng[i] or 0)
                except (TypeError, ValueError):
                    pass
        elim = observation.get("eliminated")
        if isinstance(elim, list):
            for i in range(min(n, len(elim))):
                memory.eliminated[i] = bool(elim[i])

        # Mark public cards as seen (own hand, top discard, all cards in melds on table).
        for card in observation.get("hand", []) or []:
            if isinstance(card, dict):
                memory.record_seen_card(card)
        top = observation.get("discard_top")
        if isinstance(top, dict):
            memory.record_seen_card(top)
        for meld in observation.get("melds_on_table", []) or []:
            if not isinstance(meld, dict):
                continue
            for card in meld.get("cards", []) or []:
                if isinstance(card, dict):
                    memory.record_seen_card(card)

        # Detect a new public discard since the last observation.
        # Heuristic: if top changed, attribute it to the player just before me in turn order
        # (the one who discarded most recently before the bot's turn).
        new_top_token: int | None = None
        if isinstance(top, dict):
            new_top_token = int(card_token(top))
        if (
            new_top_token is not None
            and new_top_token != self._last_top_discard_token
            and isinstance(top, dict)
        ):
            previous_seat = (my_seat - 1) % n
            memory.record_discard_event(previous_seat, top)
        self._last_top_discard_token = new_top_token

    def update_obs_memory_from_round_end(self, payload: dict[str, Any]) -> None:
        memory = self.obs_memory
        if memory is None:
            return
        n = memory.num_players
        cumulative = payload.get("cumulative")
        if isinstance(cumulative, list):
            for i in range(min(n, len(cumulative))):
                try:
                    memory.match_scores[i] = float(cumulative[i] or 0)
                except (TypeError, ValueError):
                    pass
        # New round → reset per-round memory but keep match-level state.
        memory.reset_round()
        self._last_top_discard_token = None

    def record_own_action(self, action: dict[str, Any]) -> None:
        memory = self.obs_memory
        if memory is None:
            return
        action_type = str(action.get("type", ""))
        memory.record_own_action(action_type)
        # If this was a Discard or Cruzar, we know exactly which card we threw — add it
        # to our own recent_discards entry (seat 0 from our perspective).
        if action_type in {"Discard", "Cruzar"}:
            card = action.get("card")
            if isinstance(card, dict):
                my_seat = self.seat if self.seat is not None else 0
                memory.record_discard_event(int(my_seat), card)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "status": self.status,
                "connected": self.connected,
                "running": self.running,
                "seat": self.seat,
                "stop_requested": self.stop_requested,
                "last_error": self.last_error,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "actions_sent": self.actions_sent,
                "observations_seen": self.observations_seen,
                "decision_metrics": dict(self.decision_metrics),
                "last_observation": self.last_observation,
                "last_round_end": self.last_round_end,
                "last_match_end": self.last_match_end,
                "config": asdict(self.config),
            }

    def events_payload(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"ts": e.ts, "type": e.type, "data": e.data} for e in self.events]


class RemoteSessionManager:
    def __init__(self, model_registry: dict[str, RegisteredModel]) -> None:
        self.model_registry = model_registry
        self._sessions: dict[str, RemoteSession] = {}
        self._lock = threading.Lock()

    def create_session(self, config: RemoteSessionConfig) -> RemoteSession:
        if config.model_name and config.model_name not in self.model_registry:
            raise ValueError(f"Unknown model '{config.model_name}'")
        session = RemoteSession(session_id=uuid4().hex, config=config)
        with self._lock:
            self._sessions[session.session_id] = session
        session.log("created", config=asdict(config))
        return session

    def get_session(self, session_id: str) -> RemoteSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("Remote session not found")
            return self._sessions[session_id]

    def start_session(self, session_id: str) -> RemoteSession:
        session = self.get_session(session_id)
        with session._lock:
            if session.running:
                raise ValueError("Session already running")
            session.running = True
            session.stop_requested = False
            session.status = "connecting"
            session.started_at = time.time()
            session.stopped_at = None
            session.last_error = None
            session._move_joker_streak = 0

        thread = threading.Thread(
            target=self._run_session_thread,
            args=(session,),
            daemon=True,
            name=f"remote-session-{session_id[:8]}",
        )
        session._thread = thread
        thread.start()
        return session

    def stop_session(self, session_id: str) -> RemoteSession:
        session = self.get_session(session_id)
        with session._lock:
            session.stop_requested = True
            session.status = "stopping"
        session.log("stop_requested")
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [s.to_dict() for s in sessions]

    def _run_session_thread(self, session: RemoteSession) -> None:
        try:
            asyncio.run(self._run_session(session))
        except Exception as exc:  # pragma: no cover - final safety net
            with session._lock:
                session.last_error = str(exc)
                session.status = "error"
            session.log("fatal_error", message=str(exc))
        finally:
            with session._lock:
                session.running = False
                session.connected = False
                if session.status not in {"error", "finished"}:
                    session.status = "stopped" if session.stop_requested else session.status
                session.stopped_at = time.time()
            session.log("stopped", status=session.status)

    async def _run_session(self, session: RemoteSession) -> None:
        cfg = session.config
        with session._lock:
            session.status = "connecting"
        session.log("connecting", url=cfg.url)

        register_payload: dict[str, Any] = {"type": "register", "name": cfg.name}
        if cfg.token:
            register_payload["token"] = cfg.token

        model = self.model_registry.get(cfg.model_name).model if cfg.model_name else None
        ws_timeout = 10.0
        async with websockets.connect(cfg.url, open_timeout=ws_timeout, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps(register_payload))
            session.log("register_sent", name=cfg.name)

            raw_first = await asyncio.wait_for(ws.recv(), timeout=30.0)
            first = json.loads(raw_first)
            if first.get("type") != "registered":
                msg = f"Register failed: {first}"
                with session._lock:
                    session.last_error = msg
                    session.status = "error"
                session.log("register_failed", payload=first)
                return

            with session._lock:
                session.connected = True
                session.status = "running"
                session.seat = int(first.get("seat", -1))
            session.log("registered", payload=first)

            # Heartbeat / staleness diagnostics — log when waiting for the server takes too long.
            last_recv_ts = time.time()
            last_idle_log_ts = last_recv_ts
            idle_log_interval = 10.0  # seconds: log "still waiting" every N seconds of silence
            stale_warning_threshold = 30.0  # seconds: log a stronger warning after this much silence
            stale_warning_emitted = False

            while True:
                if session.stop_requested:
                    session.log("closing", reason="stop_requested")
                    await ws.close(code=1000, reason="stop_requested")
                    return
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except TimeoutError:
                    now = time.time()
                    silence = now - last_recv_ts
                    if silence >= idle_log_interval and (now - last_idle_log_ts) >= idle_log_interval:
                        session.log(
                            "idle",
                            silence_seconds=round(silence, 1),
                            ws_open=not getattr(ws, "closed", False),
                        )
                        last_idle_log_ts = now
                    if silence >= stale_warning_threshold and not stale_warning_emitted:
                        # Try a low-cost ping to surface a dead connection promptly.
                        try:
                            pong_waiter = await ws.ping()
                            await asyncio.wait_for(pong_waiter, timeout=5.0)
                            session.log("stale_warning", silence_seconds=round(silence, 1), ping="ok")
                        except TimeoutError:
                            session.log(
                                "stale_warning",
                                silence_seconds=round(silence, 1),
                                ping_error="timeout: server did not respond to ping in 5s",
                            )
                        except Exception as ping_exc:
                            err = str(ping_exc) or type(ping_exc).__name__
                            session.log("stale_warning", silence_seconds=round(silence, 1), ping_error=err)
                        stale_warning_emitted = True
                    continue
                except websockets.ConnectionClosed as conn_exc:
                    session.log(
                        "connection_closed",
                        code=getattr(conn_exc, "code", None),
                        reason=str(getattr(conn_exc, "reason", "") or ""),
                    )
                    return

                last_recv_ts = time.time()
                last_idle_log_ts = last_recv_ts
                stale_warning_emitted = False
                msg = json.loads(raw_msg)
                msg_type = msg.get("type")
                session.log("recv", payload=msg)

                if msg_type == "observation":
                    obs = msg.get("obs", {})
                    with session._lock:
                        session.last_observation = obs
                        session.observations_seen += 1
                    # Update the smart-obs memory before invoking the model so it sees
                    # consistent state with what the training env produces.
                    session.update_obs_memory_from_observation(obs)

                    action, meta = choose_remote_action(
                        obs,
                        model=model,
                        remote_like_policy=cfg.remote_like_policy,
                        obs_memory=session.obs_memory,
                    )
                    legal_actions = obs.get("legal_actions", []) if isinstance(obs.get("legal_actions"), list) else []
                    selected_type = str(meta.get("action_type_selected") or action.get("type") or "")
                    phase = str(obs.get("phase") or "")
                    move_joker_streak = int(getattr(session, "_move_joker_streak", 0))

                    if phase != "play_or_discard":
                        move_joker_streak = 0
                    elif selected_type == "MoveJoker":
                        move_joker_streak += 1
                    else:
                        move_joker_streak = 0

                    guard_applied = False
                    if (
                        phase == "play_or_discard"
                        and selected_type == "MoveJoker"
                        and move_joker_streak > MOVE_JOKER_GUARD_MAX_STREAK
                    ):
                        forced = _pick_guard_discard_action(legal_actions)
                        if forced is not None:
                            action = forced
                            selected_type = "Discard"
                            move_joker_streak = 0
                            guard_applied = True
                            meta["guard_applied"] = True
                            meta["guard_reason"] = "move_joker_loop_protection"
                            meta["action_type_selected"] = "Discard"
                        else:
                            meta["guard_applied"] = False

                    setattr(session, "_move_joker_streak", move_joker_streak)
                    # Record the action in the obs memory so the next observation's
                    # own-action-history feature is in sync.
                    session.record_own_action(action)
                    # Optional spectator delay so a watching human can follow along.
                    delay = float(getattr(cfg, "action_delay_seconds", 0.0) or 0.0)
                    if delay > 0:
                        await asyncio.sleep(min(delay, 30.0))  # clamp at 30s safety cap
                    await ws.send(json.dumps({"type": "act", "action": action}))
                    hand_cards = obs.get("hand", []) if isinstance(obs.get("hand"), list) else []
                    hand_compact = []
                    for c in hand_cards:
                        if not isinstance(c, dict):
                            continue
                        if c.get("joker"):
                            hand_compact.append(f"J*#{c.get('deck_id')}")
                        else:
                            hand_compact.append(f"{c.get('rank')}{c.get('suit')}#{c.get('deck_id')}")
                    with session._lock:
                        session.actions_sent += 1
                        if meta.get("used_model"):
                            session.decision_metrics["model_decisions"] += 1
                        if meta.get("fallback"):
                            session.decision_metrics["fallback_decisions"] += 1
                        if meta.get("go_out_available"):
                            session.decision_metrics["go_out_available_observations"] += 1
                        if meta.get("selected_go_out_action"):
                            session.decision_metrics["go_out_selected"] += 1
                        if meta.get("go_out_available") and not meta.get("selected_go_out_action"):
                            session.decision_metrics["go_out_missed"] += 1
                        if meta.get("discard_selected"):
                            session.decision_metrics["discard_selected"] += 1
                        if meta.get("discard_while_go_out_available"):
                            session.decision_metrics["discard_while_go_out_available"] += 1
                        selected_type = str(meta.get("action_type_selected") or action.get("type") or "")
                        if selected_type == "LayEscalera":
                            session.decision_metrics["lay_escalera_selected"] += 1
                        elif selected_type == "LayPierna":
                            session.decision_metrics["lay_pierna_selected"] += 1
                        elif selected_type == "ExtendMeld":
                            session.decision_metrics["extend_meld_selected"] += 1
                        elif selected_type == "MoveJoker":
                            session.decision_metrics["move_joker_selected"] += 1
                        elif selected_type == "Cruzar":
                            session.decision_metrics["cruzar_selected"] += 1
                        if guard_applied:
                            session.decision_metrics["move_joker_guard_forced_discard"] += 1
                    session.log("action_sent", action=action, meta=meta, hand=hand_compact)
                    continue

                if msg_type == "round_end":
                    with session._lock:
                        session.last_round_end = msg
                    # Update match-level obs memory with cumulative scores; reset per-round memory.
                    session.update_obs_memory_from_round_end(msg)
                    session.log("round_end", payload=msg)
                    continue

                if msg_type == "match_end":
                    with session._lock:
                        session.last_match_end = msg
                        session.status = "finished"
                    session.log("match_end", payload=msg)
                    return

                if msg_type == "error":
                    with session._lock:
                        session.last_error = msg.get("message", "remote_error")
                    session.log("remote_error", payload=msg)
                    continue

                if msg_type == "pong":
                    session.log("pong")
                    continue

                session.log("unknown_message", payload=msg)
