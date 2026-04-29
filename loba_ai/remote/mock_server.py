from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass
from typing import Any

import websockets

from loba_ai.cards import hand_points
from loba_ai.melds import is_valid_run
from loba_ai.remote_like_engine import RemoteLikeGameEngine
from loba_ai.rules import Rules


def _card_payload(card) -> dict[str, Any]:
    if card.is_joker:
        return {"joker": True, "deck_id": card.deck_id}
    rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
    suit_map = {"spades": "S", "hearts": "H", "clubs": "C", "diamonds": "D"}
    return {"rank": rank_map.get(card.rank, str(card.rank)), "suit": suit_map.get(card.suit or "clubs", "C"), "deck_id": card.deck_id}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class RemoteMockServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    num_players: int = 3
    remote_name: str = "brother-bot"
    token: str | None = None
    seed: int = 0


@dataclass(slots=True)
class RemoteMockStatus:
    running: bool
    host: str
    port: int
    num_players: int
    remote_name: str
    connected: bool
    seat: int
    turn_number: int
    phase: str
    last_error: str | None


class RemoteMockServerManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._started_event = threading.Event()
        self._config = RemoteMockServerConfig()
        self._engine = RemoteLikeGameEngine(Rules(num_players=3), seed=0)
        self._last_error: str | None = None
        self._connected = False
        self._remote_seat = 0
        self._pending_observation = False
        self._registered = False
        self._server: Any | None = None
        self._games_played = 0  # increments per client connection so seeds vary across games

    def _validate_table_integrity_or_fail(self) -> None:
        for meld in self._engine.table_melds:
            if meld.get("kind") != "escalera":
                continue
            cards = meld.get("cards", [])
            if not is_valid_run(cards, self._engine.rules):
                self._last_error = (
                    f"invalid_escalera_detected meld_id={meld.get('meld_id')} owner={meld.get('owner')}"
                )
                self._engine.state.finished = True
                self._engine.state.winner = self._remote_seat
                return

    def status(self) -> dict[str, Any]:
        with self._lock:
            cfg = self._config
            s = self._engine.state
            payload = RemoteMockStatus(
                running=self._thread is not None and self._thread.is_alive(),
                host=cfg.host,
                port=cfg.port,
                num_players=cfg.num_players,
                remote_name=cfg.remote_name,
                connected=self._connected,
                seat=self._remote_seat,
                turn_number=int(s.turn_number),
                phase=str(s.phase),
                last_error=self._last_error,
            )
        return asdict(payload)

    def full_state(self) -> dict[str, Any]:
        """Return a god-mode snapshot of the engine — all hands visible. Spectator-only."""
        with self._lock:
            s = self._engine.state
            cfg = self._config
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "current_player": int(s.current_player),
                "phase": str(s.phase),
                "turn_number": int(s.turn_number),
                "round_index": int(s.round_index),
                "finished": bool(s.finished),
                "winner": s.winner,
                "stock_size": len(s.stock_pile),
                "discard_size": len(s.discard_pile),
                "discard_top": _card_payload(s.discard_pile[-1]) if s.discard_pile else None,
                "remote_seat": int(self._remote_seat),
                "players": [
                    {
                        "seat": idx,
                        "is_remote": idx == self._remote_seat,
                        "hand_size": len(p.hand),
                        "hand": [_card_payload(c) for c in p.hand],
                        "has_opened": bool(p.has_opened),
                    }
                    for idx, p in enumerate(s.players)
                ],
                "melds_on_table": [
                    {
                        "meld_id": m.get("meld_id"),
                        "owner": m.get("owner"),
                        "kind": m.get("kind"),
                        "cards": [_card_payload(c) for c in m.get("cards", [])],
                    }
                    for m in self._engine.table_melds
                ],
                "games_played": int(self._games_played),
                "config": {
                    "num_players": cfg.num_players,
                    "remote_name": cfg.remote_name,
                },
            }

    def start(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        num_players: int = 3,
        remote_name: str = "brother-bot",
        token: str | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("Remote mock server is already running")
            cfg = RemoteMockServerConfig(
                host=host,
                port=int(port),
                num_players=max(2, min(5, int(num_players))),
                remote_name=remote_name,
                token=token,
                seed=int(seed),
            )
            self._config = cfg
            self._engine = RemoteLikeGameEngine(Rules(num_players=cfg.num_players), seed=cfg.seed)
            self._connected = False
            self._last_error = None
            self._remote_seat = 0
            self._pending_observation = False
            self._registered = False

        self._started_event.clear()
        thread = threading.Thread(target=self._thread_main, name="remote-mock-server", daemon=True)
        self._thread = thread
        thread.start()
        if not self._started_event.wait(timeout=5.0):
            raise RuntimeError("Remote mock server did not start in time")
        return self.status()

    def stop(self) -> dict[str, Any]:
        thread = None
        with self._lock:
            if self._loop and self._stop_event:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            thread = self._thread
        if thread:
            thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
            self._loop = None
            self._stop_event = None
            self._server = None
            self._connected = False
            self._registered = False
            self._pending_observation = False
        return self.status()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop_event = stop_event
        try:
            loop.run_until_complete(self._run_server(stop_event))
        except Exception as exc:  # pragma: no cover - emergency safety net
            with self._lock:
                self._last_error = str(exc)
        finally:
            try:
                loop.stop()
            finally:
                loop.close()

    async def _run_server(self, stop_event: asyncio.Event) -> None:
        cfg = self._config
        ws_server = await websockets.serve(self._handle_client, cfg.host, cfg.port, ping_interval=20, ping_timeout=20)
        self._server = ws_server
        if ws_server.sockets:
            actual_port = int(ws_server.sockets[0].getsockname()[1])
            with self._lock:
                self._config.port = actual_port
        self._started_event.set()
        await stop_event.wait()
        ws_server.close()
        await ws_server.wait_closed()

    async def _handle_client(self, websocket) -> None:
        # Auto-reset the engine for every new connection so each bot session starts a fresh game.
        # The seed advances per-connection so consecutive games aren't identical.
        with self._lock:
            cfg = self._config
            game_seed = int(cfg.seed) + int(self._games_played)
            self._games_played += 1
            self._engine = RemoteLikeGameEngine(Rules(num_players=cfg.num_players), seed=game_seed)
            self._connected = False
            self._registered = False
            self._pending_observation = False
            self._last_error = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            first = json.loads(raw)
            if first.get("type") != "register":
                await websocket.send(json.dumps({"type": "error", "code": "bad_register", "message": "expected register"}))
                await websocket.close(code=1003, reason="bad_register")
                return
            name = str(first.get("name") or "")
            token = first.get("token")
            cfg = self._config
            with self._lock:
                if name != cfg.remote_name:
                    await websocket.send(
                        json.dumps({"type": "error", "code": "unknown_slot", "message": f"no slot named '{name}'"})
                    )
                    await websocket.close(code=4404, reason="unknown_slot")
                    return
                if cfg.token and token != cfg.token:
                    await websocket.send(json.dumps({"type": "error", "code": "auth_failed", "message": "invalid token"}))
                    await websocket.close(code=4401, reason="auth_failed")
                    return
                if self._connected:
                    await websocket.send(json.dumps({"type": "error", "code": "slot_busy", "message": "slot busy"}))
                    await websocket.close(code=4409, reason="slot_busy")
                    return
                self._connected = True
                self._registered = True
                self._remote_seat = 0

            await websocket.send(json.dumps({"type": "registered", "name": name, "seat": self._remote_seat}))
            await self._autoplay_until_remote_or_done()
            await self._send_observation_if_needed(websocket)
            while True:
                raw_msg = await websocket.recv()
                msg = json.loads(raw_msg)
                msg_type = msg.get("type")
                if msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                    continue
                if msg_type != "act":
                    await websocket.send(
                        json.dumps({"type": "error", "code": "unknown_type", "message": f"unsupported type '{msg_type}'"})
                    )
                    continue
                if not self._pending_observation:
                    await websocket.send(json.dumps({"type": "error", "code": "not_your_turn", "message": "no pending observation"}))
                    continue

                action = msg.get("action")
                if not isinstance(action, dict):
                    await websocket.send(json.dumps({"type": "error", "code": "bad_action", "message": "action must be object"}))
                    continue

                legal = self._engine.legal_actions()
                legal_map = {_canonical_json(a): a for a in legal}
                incoming_key = _canonical_json(action)
                if incoming_key not in legal_map:
                    await websocket.send(json.dumps({"type": "error", "code": "illegal_action", "message": "action not legal"}))
                    continue

                self._pending_observation = False
                self._engine.step(legal_map[incoming_key], is_agent_player=True)
                self._validate_table_integrity_or_fail()
                if self._engine.state.finished:
                    await self._send_round_and_match_end(websocket)
                    return

                await self._autoplay_until_remote_or_done()
                if self._engine.state.finished:
                    await self._send_round_and_match_end(websocket)
                    return
                await self._send_observation_if_needed(websocket)
        except asyncio.TimeoutError:
            await websocket.send(json.dumps({"type": "error", "code": "register_timeout", "message": "register timeout"}))
            await websocket.close(code=4408, reason="register_timeout")
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:  # pragma: no cover - defensive safety
            with self._lock:
                self._last_error = str(exc)
            try:
                await websocket.send(json.dumps({"type": "error", "code": "server_error", "message": str(exc)}))
            except Exception:
                pass
        finally:
            with self._lock:
                self._connected = False
                self._registered = False
                self._pending_observation = False

    async def _autoplay_until_remote_or_done(self) -> None:
        # Hard cap on iterations between two remote-turns. Real games have ~50-150
        # opponent actions per gap; 1000 is generous and only fires on degenerate loops.
        max_iterations = 1000
        iterations = 0
        action_type_counts: dict[str, int] = {}
        last_seat: int | None = None
        consecutive_same_seat = 0
        while not self._engine.state.finished and self._engine.state.current_player != self._remote_seat:
            iterations += 1
            if iterations > max_iterations:
                self._last_error = (
                    f"autoplay_iteration_cap_exceeded "
                    f"iterations={iterations} action_counts={action_type_counts} "
                    f"current_player={self._engine.state.current_player} phase={self._engine.state.phase}"
                )
                # Force-finish so the bot doesn't hang. Mark the bot as winner by default;
                # this is a degraded outcome but preferable to an infinite spin.
                self._engine.state.finished = True
                self._engine.state.winner = int(self._remote_seat)
                return

            legal = self._engine.legal_actions()
            if not legal:
                self._engine.state.finished = True
                self._engine.state.winner = int(self._remote_seat)
                return

            current_seat = int(self._engine.state.current_player)
            if last_seat == current_seat:
                consecutive_same_seat += 1
            else:
                last_seat = current_seat
                consecutive_same_seat = 1

            choice = self._choose_heuristic_action(
                legal, suppress_move_joker=consecutive_same_seat >= 3
            )
            choice_type = str(choice.get("type", "?"))
            action_type_counts[choice_type] = action_type_counts.get(choice_type, 0) + 1
            self._engine.step(choice, is_agent_player=False)
            self._validate_table_integrity_or_fail()

    def _choose_heuristic_action(
        self,
        legal_actions: list[dict[str, Any]],
        suppress_move_joker: bool = False,
    ) -> dict[str, Any]:
        # MoveJoker placed LAST so the heuristic never picks it over a Discard.
        # Without this, a player with a joker on the table can MoveJoker indefinitely
        # (engine doesn't end the turn on MoveJoker), causing the autoplay loop to spin.
        preferred = [
            "LayEscalera",
            "LayPierna",
            "ExtendMeld",
            "Cruzar",
            "DrawDiscard",
            "DrawStock",
            "Discard",
            "MoveJoker",
        ]
        candidates = legal_actions
        if suppress_move_joker:
            filtered = [a for a in legal_actions if str(a.get("type")) != "MoveJoker"]
            if filtered:
                candidates = filtered
        legal_sorted = sorted(
            candidates,
            key=lambda a: (
                preferred.index(str(a.get("type"))) if str(a.get("type")) in preferred else 99,
                _canonical_json(a),
            ),
        )
        return legal_sorted[0]

    def _observation_payload(self) -> dict[str, Any]:
        self._validate_table_integrity_or_fail()
        s = self._engine.state
        seat = self._remote_seat
        return {
            "seat": seat,
            "num_players": self._config.num_players,
            "phase": s.phase,
            "hand": [_card_payload(c) for c in s.players[seat].hand],
            "other_hand_sizes": [len(p.hand) for p in s.players],
            "stock_size": len(s.stock_pile),
            "discard_top": _card_payload(s.discard_pile[-1]) if s.discard_pile else None,
            "discard_size": len(s.discard_pile),
            "pending_discard": None,
            "melds_on_table": [
                {
                    "meld_id": m["meld_id"],
                    "owner": m["owner"],
                    "kind": m["kind"],
                    "cards": [_card_payload(c) for c in m["cards"]],
                }
                for m in self._engine.table_melds
            ],
            "has_laid_meld_this_round": [bool(p.has_opened) for p in s.players],
            "cumulative_scores": [int(p.score) for p in s.players],
            "reenganches_used": [0 for _ in s.players],
            "eliminated": [False for _ in s.players],
            "legal_actions": list(self._engine.legal_actions()),
        }

    async def _send_observation_if_needed(self, websocket) -> None:
        if self._engine.state.finished:
            return
        if self._engine.state.current_player != self._remote_seat:
            return
        self._pending_observation = True
        await websocket.send(json.dumps({"type": "observation", "obs": self._observation_payload()}))

    async def _send_round_and_match_end(self, websocket) -> None:
        winner = self._engine.state.winner
        penalties = [hand_points(p.hand) for p in self._engine.state.players]
        if winner is not None and 0 <= winner < len(penalties):
            penalties[winner] = 0
        cumulative = list(penalties)
        await websocket.send(
            json.dumps(
                {
                    "type": "round_end",
                    "winner": winner,
                    "penalties": penalties,
                    "cumulative": cumulative,
                }
            )
        )
        await websocket.send(json.dumps({"type": "match_end", "winner": winner}))
