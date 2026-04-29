from __future__ import annotations

import secrets
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from loba_ai.cards import hand_points
from loba_ai.playground.service import PlayerConfig, PlaygroundService
from loba_ai.remote import RemoteMockServerManager, RemoteSessionConfig, RemoteSessionManager

app = FastAPI(title="Loba AI Playground API", version="0.1.0")
service = PlaygroundService()
remote_manager = RemoteSessionManager(service.models)
remote_mock_manager = RemoteMockServerManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterModelRequest(BaseModel):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class PlayerRequest(BaseModel):
    kind: str
    model_name: str | None = None


class CreateMatchRequest(BaseModel):
    players: list[PlayerRequest]
    seed: int | None = None
    autoplay_enabled: bool = False


class HumanActionRequest(BaseModel):
    action: int


class ArenaCreateRoomRequest(BaseModel):
    host_model_name: str = Field(min_length=1)
    room_name: str | None = None
    target_points: int = Field(default=100, ge=10, le=1000)


class ArenaJoinRoomRequest(BaseModel):
    guest_name: str = Field(min_length=1)


class ArenaReadyRequest(BaseModel):
    player_token: str = Field(min_length=1)


class ArenaActionRequest(BaseModel):
    player_token: str = Field(min_length=1)
    action: int


class RemoteCreateSessionRequest(BaseModel):
    url: str = Field(min_length=1)
    name: str = Field(min_length=1)
    token: str | None = None
    model_name: str | None = None
    remote_like_policy: bool = False
    action_delay_seconds: float = 0.0


class RemoteMockStartRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8766, ge=0, le=65535)
    num_players: int = Field(default=3, ge=2, le=5)
    remote_name: str = Field(default="brother-bot", min_length=1)
    token: str | None = None
    seed: int = 0


@dataclass(slots=True)
class ArenaRoom:
    room_id: str
    host_model_name: str
    room_name: str | None = None
    guest_name: str | None = None
    match_id: str | None = None
    target_points: int = 100
    host_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    guest_token: str | None = None
    ready_players: set[int] = field(default_factory=set)
    scores: list[int] = field(default_factory=lambda: [0, 0])
    round_number: int = 1
    match_finished: bool = False
    champion: int | None = None
    current_match_scored: bool = False
    last_round_result: dict | None = None

    def tokens(self) -> dict[str, int]:
        mapping = {self.host_token: 0}
        if self.guest_token:
            mapping[self.guest_token] = 1
        return mapping


arena_rooms: dict[str, ArenaRoom] = {}


def _arena_start_round(room: ArenaRoom) -> None:
    created = service.create_match(
        players=[
            PlayerConfig(kind="model", model_name=room.host_model_name),
            PlayerConfig(kind="human", model_name=None),
        ]
    )
    room.match_id = created["match_id"]
    room.current_match_scored = False


def _arena_advance_if_round_finished(room: ArenaRoom) -> None:
    if not room.match_id or room.match_finished:
        return
    match = service.get_match(room.match_id)
    if not match.state.finished or room.current_match_scored:
        return

    winner = match.state.winner
    if winner is None:
        return
    round_points = [
        hand_points(match.state.players[1].hand),
        hand_points(match.state.players[0].hand),
    ]
    earned = int(round_points[winner])
    room.scores[winner] += earned
    room.current_match_scored = True
    room.last_round_result = {
        "round_number": room.round_number,
        "winner": winner,
        "points_earned": earned,
        "scores": list(room.scores),
    }

    if room.scores[winner] >= room.target_points:
        room.match_finished = True
        room.champion = winner
        return

    room.round_number += 1
    _arena_start_round(room)


def _arena_room_payload(room: ArenaRoom) -> dict:
    _arena_advance_if_round_finished(room)
    return {
        "room_id": room.room_id,
        "room_name": room.room_name,
        "host_model_name": room.host_model_name,
        "guest_name": room.guest_name,
        "match_id": room.match_id,
        "target_points": room.target_points,
        "scores": room.scores,
        "round_number": room.round_number,
        "match_finished": room.match_finished,
        "champion": room.champion,
        "last_round_result": room.last_round_result,
        "status": "finished" if room.match_finished else ("in_game" if room.match_id else "waiting"),
        "ready_player_indexes": sorted(room.ready_players),
    }


def _arena_require_room(room_id: str) -> ArenaRoom:
    room = arena_rooms.get(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Arena room not found")
    return room


def _arena_resolve_player(room: ArenaRoom, token: str) -> int:
    player_ix = room.tokens().get(token)
    if player_ix is None:
        raise HTTPException(status_code=403, detail="Invalid player token")
    return player_ix


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {"items": service.list_models()}


@app.post("/models/register")
def register_model(payload: RegisterModelRequest) -> dict:
    try:
        model = service.register_model(name=payload.name, path=payload.path)
        return {"model": model}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/models/{name}")
def delete_model(name: str) -> dict:
    try:
        service.remove_model(name)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/matches")
def create_match(payload: CreateMatchRequest) -> dict:
    try:
        players = [PlayerConfig(kind=p.kind, model_name=p.model_name) for p in payload.players]
        return service.create_match(players=players, seed=payload.seed, autoplay_enabled=payload.autoplay_enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/matches/{match_id}")
def get_match(match_id: str) -> dict:
    try:
        match = service.get_match(match_id)
        return {"state": match.snapshot()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/matches/{match_id}/action")
def action(match_id: str, payload: HumanActionRequest) -> dict:
    try:
        match = service.get_match(match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if match.state.finished:
        return {"state": match.snapshot(), "autoplay_log": []}

    current = match.state.current_player
    if match.players[current].kind != "human":
        raise HTTPException(status_code=400, detail="Current player is not human; call /autoplay")

    valid_actions = [v["id"] for v in match.snapshot()["valid_actions"]]
    if payload.action not in valid_actions:
        raise HTTPException(status_code=400, detail="Invalid action")

    try:
        human = match.step(payload.action)
        auto = match.autoplay_until_human_or_end()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"state": auto["state"], "autoplay_log": auto["log"], "human_log_entry": human["event"]}


@app.post("/matches/{match_id}/autoplay")
def autoplay(match_id: str) -> dict:
    try:
        match = service.get_match(match_id)
        auto = match.autoplay_until_human_or_end()
        return {"state": auto["state"], "autoplay_log": auto["log"]}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/arena/rooms")
def arena_create_room(payload: ArenaCreateRoomRequest) -> dict:
    if payload.host_model_name not in {m["name"] for m in service.list_models()}:
        raise HTTPException(status_code=400, detail=f"Unknown model '{payload.host_model_name}'")
    room = ArenaRoom(
        room_id=secrets.token_urlsafe(8),
        host_model_name=payload.host_model_name,
        room_name=payload.room_name,
        target_points=payload.target_points,
    )
    arena_rooms[room.room_id] = room
    return {"room": _arena_room_payload(room), "host_player_token": room.host_token}


@app.get("/arena/rooms/{room_id}")
def arena_get_room(room_id: str) -> dict:
    room = _arena_require_room(room_id)
    return {"room": _arena_room_payload(room)}


@app.post("/arena/rooms/{room_id}/join")
def arena_join_room(room_id: str, payload: ArenaJoinRoomRequest) -> dict:
    room = _arena_require_room(room_id)
    if room.match_id:
        raise HTTPException(status_code=400, detail="Match already started for this room")
    if room.guest_token:
        raise HTTPException(status_code=400, detail="Room already has a guest")
    room.guest_name = payload.guest_name
    room.guest_token = secrets.token_urlsafe(24)
    return {"room": _arena_room_payload(room), "guest_player_token": room.guest_token, "player_index": 1}


@app.post("/arena/rooms/{room_id}/ready")
def arena_ready(room_id: str, payload: ArenaReadyRequest) -> dict:
    room = _arena_require_room(room_id)
    if room.match_finished:
        raise HTTPException(status_code=400, detail="Arena match is already finished")
    player_ix = _arena_resolve_player(room, payload.player_token)
    room.ready_players.add(player_ix)

    if room.match_id is None and room.guest_token and room.ready_players == {0, 1}:
        _arena_start_round(room)

    response = {"room": _arena_room_payload(room)}
    if room.match_id and not room.match_finished:
        match = service.get_match(room.match_id)
        response["state"] = match.snapshot_for_player(player_ix)
    return response


@app.get("/arena/rooms/{room_id}/state")
def arena_state(room_id: str, player_token: str) -> dict:
    room = _arena_require_room(room_id)
    _arena_advance_if_round_finished(room)
    if not room.match_id:
        raise HTTPException(status_code=400, detail="Match not started yet")
    if room.match_finished:
        return {"room": _arena_room_payload(room), "state": None}
    player_ix = _arena_resolve_player(room, player_token)
    match = service.get_match(room.match_id)
    return {"room": _arena_room_payload(room), "state": match.snapshot_for_player(player_ix), "player_index": player_ix}


@app.post("/arena/rooms/{room_id}/action")
def arena_action(room_id: str, payload: ArenaActionRequest) -> dict:
    room = _arena_require_room(room_id)
    _arena_advance_if_round_finished(room)
    if not room.match_id:
        raise HTTPException(status_code=400, detail="Match not started yet")
    if room.match_finished:
        return {"room": _arena_room_payload(room), "state": None, "autoplay_log": []}
    player_ix = _arena_resolve_player(room, payload.player_token)
    match = service.get_match(room.match_id)

    if match.state.finished:
        return {"room": _arena_room_payload(room), "state": match.snapshot_for_player(player_ix), "autoplay_log": []}

    if match.state.current_player != player_ix:
        raise HTTPException(status_code=400, detail="Not your turn")

    valid_actions = [v["id"] for v in match.snapshot_for_player(player_ix)["valid_actions"]]
    if payload.action not in valid_actions:
        raise HTTPException(status_code=400, detail="Invalid action")

    try:
        own = match.step(payload.action)
        auto = match.autoplay_until_human_or_end()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _arena_advance_if_round_finished(room)
    if room.match_finished:
        return {
            "room": _arena_room_payload(room),
            "state": None,
            "autoplay_log": auto["log"],
            "human_log_entry": own["event"],
        }

    current = service.get_match(room.match_id)
    return {
        "room": _arena_room_payload(room),
        "state": current.snapshot_for_player(player_ix),
        "autoplay_log": auto["log"],
        "human_log_entry": own["event"],
    }


@app.get("/remote/sessions")
def remote_list_sessions() -> dict:
    return {"items": remote_manager.list_sessions()}


@app.post("/remote/sessions")
def remote_create_session(payload: RemoteCreateSessionRequest) -> dict:
    try:
        session = remote_manager.create_session(
            RemoteSessionConfig(
                url=payload.url,
                name=payload.name,
                token=payload.token,
                model_name=payload.model_name,
                remote_like_policy=payload.remote_like_policy,
                action_delay_seconds=max(0.0, float(payload.action_delay_seconds)),
            )
        )
        return {"session": session.to_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/remote/sessions/{session_id}/restart")
def remote_restart_session(session_id: str) -> dict:
    """Stop the given session if needed, clone its config into a brand-new session, and start it.

    One-click "play another game" — UI flow becomes a single request.
    """
    try:
        old_session = remote_manager.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Stop the old session if it's still alive (no-op if already finished).
    try:
        remote_manager.stop_session(session_id)
    except Exception:
        pass

    cfg = old_session.config
    try:
        new_session = remote_manager.create_session(
            RemoteSessionConfig(
                url=cfg.url,
                name=cfg.name,
                token=cfg.token,
                model_name=cfg.model_name,
                remote_like_policy=cfg.remote_like_policy,
                action_delay_seconds=cfg.action_delay_seconds,
            )
        )
        started = remote_manager.start_session(new_session.session_id)
        return {"session": started.to_dict(), "previous_session_id": session_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/remote/sessions/{session_id}/start")
def remote_start_session(session_id: str) -> dict:
    try:
        session = remote_manager.start_session(session_id)
        return {"session": session.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/remote/sessions/{session_id}/stop")
def remote_stop_session(session_id: str) -> dict:
    try:
        session = remote_manager.stop_session(session_id)
        return {"session": session.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/remote/sessions/{session_id}")
def remote_get_session(session_id: str) -> dict:
    try:
        session = remote_manager.get_session(session_id)
        return {"session": session.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/remote/sessions/{session_id}/events")
def remote_get_session_events(session_id: str) -> dict:
    try:
        session = remote_manager.get_session(session_id)
        return {"items": session.events_payload()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/remote/mock/start")
def remote_mock_start(payload: RemoteMockStartRequest) -> dict:
    try:
        status = remote_mock_manager.start(
            host=payload.host,
            port=payload.port,
            num_players=payload.num_players,
            remote_name=payload.remote_name,
            token=payload.token,
            seed=payload.seed,
        )
        ws_url = f"ws://{status['host']}:{status['port']}"
        return {"mock": status, "ws_url": ws_url}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/remote/mock/stop")
def remote_mock_stop() -> dict:
    status = remote_mock_manager.stop()
    ws_url = f"ws://{status['host']}:{status['port']}"
    return {"mock": status, "ws_url": ws_url}


@app.get("/remote/mock/full-state")
def remote_mock_full_state() -> dict:
    """Spectator view: full engine state including all opponents' hands.

    Only useful when the mock server is the WebSocket counterparty (you're
    playing against in-process opponents). Don't expose this in production
    against real game servers.
    """
    return remote_mock_manager.full_state()


@app.get("/remote/mock/status")
def remote_mock_status() -> dict:
    status = remote_mock_manager.status()
    ws_url = f"ws://{status['host']}:{status['port']}"
    return {"mock": status, "ws_url": ws_url}


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
QUACK_PATH = Path(__file__).resolve().parent.parent / "quack.mp3"
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/arena")
def arena_home() -> FileResponse:
    return FileResponse(str(WEB_DIR / "arena.html"))


@app.get("/remote")
def remote_home() -> FileResponse:
    return FileResponse(str(WEB_DIR / "remote.html"))


@app.get("/assets/quack.mp3")
def quack_sound() -> FileResponse:
    if not QUACK_PATH.exists():
        raise HTTPException(status_code=404, detail="quack.mp3 not found")
    return FileResponse(str(QUACK_PATH), media_type="audio/mpeg")
