"""Shared observation builder for the remote-like Loba environment.

Used by both the training env (RemoteLikeMatchSmartLobaEnv) and the deployment
adapter (loba_ai/remote/adapter.py) so they cannot diverge.

A `payload` dict has the same shape as both:
  - the env's _build_observation_payload() output
  - the server's `observation` message

A `SmartObsMemory` carries the running cross-step state that the env or the
deployment session must maintain (seen cards, recent discards, own action
history, match scores...).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ===== Layout constants =====
SUIT_TO_OFFSET = {"C": 0, "D": 13, "H": 26, "S": 39}

PHASE_TO_INDEX = {"draw": 0, "play_or_discard": 1, "cruzar_discard": 2}

OWN_ACTION_TYPES = (
    "DrawStock",
    "DrawDiscard",
    "LayPierna",
    "LayEscalera",
    "ExtendMeld",
    "MoveJoker",
    "Cruzar",
    "Discard",
)
OWN_ACTION_HISTORY_LEN = 4   # how many of the agent's own past actions to encode
OWN_ACTION_TYPE_DIM = len(OWN_ACTION_TYPES)
OWN_ACTION_FEATURE_DIM = OWN_ACTION_HISTORY_LEN * OWN_ACTION_TYPE_DIM

# Per-slot action type features. For the first LEGAL_ACTION_PREVIEW_SLOTS slots in
# the canonicalized legal-action list, encode the action type as one-hot. The
# canonical ordering already groups by type-priority, but the model still needs to
# *see* what type lives at each slot in order to learn priorities like
# "always pick Cruzar over Discard". Without this, the model only sees the action
# index numerically and has to infer type from observation features alone.
# Empirical action-count distribution: max 31, p95=14 → 32 slots covers everything.
LEGAL_ACTION_PREVIEW_SLOTS = 32
ACTION_TYPE_TO_INDEX = {t: i for i, t in enumerate(OWN_ACTION_TYPES)}
LEGAL_ACTION_TYPES_FEATURE_DIM = LEGAL_ACTION_PREVIEW_SLOTS * OWN_ACTION_TYPE_DIM

# Tactical per-action features for the same first LEGAL_ACTION_PREVIEW_SLOTS slots.
# These make decisions like "discard the dead K, not the paired 5" visible to the
# policy at the action-index level instead of forcing it to infer everything from
# the global hand vector.
ACTION_TACTICAL_FEATURES = (
    "valid",
    "is_discard",
    "is_cruzar",
    "is_draw_stock",
    "is_draw_discard",
    "is_new_meld",
    "is_extend",
    "is_move_joker",
    "is_go_out",
    "card_points",
    "card_rank",
    "is_joker",
    "same_rank_count",
    "same_suit_neighbor_count",
    "is_pair_project",
    "is_run_project",
    "is_project_card",
    "breaks_project",
    "has_high_dead_alternative",
    "discard_value_gap_to_best_dead",
    "post_action_hand_points",
    "post_action_hand_size",
    "post_action_unsafe_excess",
    "meld_cards_count",
    "meld_points_removed",
    "keeps_or_enters_zone",
)
ACTION_TACTICAL_FEATURE_DIM = len(ACTION_TACTICAL_FEATURES)
LEGAL_ACTION_TACTICAL_FEATURE_DIM = (
    LEGAL_ACTION_PREVIEW_SLOTS * ACTION_TACTICAL_FEATURE_DIM
)

# Treat same-suit cards at distance 2 as run projects too: e.g. 5D + 7D is
# waiting for 6D. Kept inside the existing features so v3 models remain shape-
# compatible while the tactical/reward signal becomes more useful.
RUN_PROJECT_MAX_RANK_GAP = 2

TABLE_FEATURE_BASE = 53      # card-presence vector
TABLE_FEATURE_AGG = 4        # escalera/pierna/left-joker/right-joker counts

MATCH_FEATURE_DIM_PER_PLAYER = 3   # score + reenganches_used_ratio + eliminated
MATCH_FEATURE_EXTRA = 1            # gap_to_best


# ===== Card payload helpers =====

def card_token(card_payload: dict) -> int:
    """Map a card payload to a token index in [0, 52]. Joker → 52."""
    if card_payload.get("joker"):
        return 52
    suit = str(card_payload.get("suit", ""))
    rank_raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    if suit not in SUIT_TO_OFFSET:
        return 52
    try:
        rank_num = int(rank_raw)
    except ValueError:
        rank_num = rank_map.get(rank_raw, 0)
    if rank_num <= 0:
        return 52
    return SUIT_TO_OFFSET[suit] + max(0, min(12, rank_num - 1))


def payload_is_joker(card_payload: dict) -> bool:
    return bool(card_payload.get("joker"))


def payload_rank_value(card_payload: dict) -> int:
    if payload_is_joker(card_payload):
        return 99
    raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 14, "J": 11, "Q": 12, "K": 13}
    if raw in rank_map:
        return rank_map[raw]
    try:
        return int(raw)
    except ValueError:
        return 0


def payload_run_rank_value(card_payload: dict) -> int:
    if payload_is_joker(card_payload):
        return 99
    raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    if raw in rank_map:
        return rank_map[raw]
    try:
        return int(raw)
    except ValueError:
        return 0


def payload_point_value(card_payload: dict) -> int:
    if payload_is_joker(card_payload):
        return 15
    raw = str(card_payload.get("rank", "0"))
    if raw == "A":
        return 11
    if raw in {"J", "Q", "K"}:
        return 10
    try:
        return int(raw)
    except ValueError:
        return 0


def payload_suit(card_payload: dict) -> str:
    return str(card_payload.get("suit", ""))


def payload_key(card_payload: dict) -> str:
    if payload_is_joker(card_payload):
        return f"J|{card_payload.get('deck_id')}"
    return f"{card_payload.get('rank')}|{card_payload.get('suit')}|{card_payload.get('deck_id')}"


def is_project_card_payload(
    card_payload: dict,
    hand_payload: list[dict],
    table_melds: list[dict] | None = None,
) -> bool:
    if payload_is_joker(card_payload):
        return True
    rv = payload_rank_value(card_payload)
    run_rv = payload_run_rank_value(card_payload)
    if rv <= 0:
        return False
    suit = payload_suit(card_payload)
    target_key = payload_key(card_payload)
    for c in hand_payload:
        if payload_key(c) == target_key:
            continue
        if payload_is_joker(c):
            continue
        rv2 = payload_rank_value(c)
        if rv2 == rv and payload_suit(c) != suit:
            return True
        rank_gap = abs(payload_run_rank_value(c) - run_rv)
        if payload_suit(c) == suit and 1 <= rank_gap <= RUN_PROJECT_MAX_RANK_GAP:
            return True
    if table_melds and table_meld_near_run_count(card_payload, table_melds) > 0:
        return True
    return False


def project_breakdown(
    card_payload: dict,
    hand_payload: list[dict],
    table_melds: list[dict] | None = None,
) -> dict[str, Any]:
    """Return tactical project facts for one card in the current hand."""
    if not isinstance(card_payload, dict):
        return {
            "same_rank_count": 0,
            "same_suit_neighbor_count": 0,
            "is_pair_project": False,
            "is_run_project": False,
            "is_project_card": False,
        }
    if payload_is_joker(card_payload):
        return {
            "same_rank_count": 0,
            "same_suit_neighbor_count": 0,
            "is_pair_project": True,
            "is_run_project": True,
            "is_project_card": True,
        }
    rv = payload_rank_value(card_payload)
    run_rv = payload_run_rank_value(card_payload)
    suit = payload_suit(card_payload)
    target_key = payload_key(card_payload)
    same_rank = 0
    same_suit_neighbors = 0
    for c in hand_payload:
        if not isinstance(c, dict):
            continue
        if payload_key(c) == target_key:
            continue
        if payload_is_joker(c):
            continue
        rv2 = payload_rank_value(c)
        if rv2 == rv and payload_suit(c) != suit:
            same_rank += 1
        rank_gap = abs(payload_run_rank_value(c) - run_rv)
        if payload_suit(c) == suit and 1 <= rank_gap <= RUN_PROJECT_MAX_RANK_GAP:
            same_suit_neighbors += 1
    same_suit_neighbors += table_meld_near_run_count(card_payload, table_melds or [])
    return {
        "same_rank_count": same_rank,
        "same_suit_neighbor_count": same_suit_neighbors,
        "is_pair_project": same_rank >= 1,
        "is_run_project": same_suit_neighbors >= 1,
        "is_project_card": same_rank >= 1 or same_suit_neighbors >= 1,
    }


def hand_points_payload(hand_payload: list[dict]) -> int:
    return sum(payload_point_value(c) for c in hand_payload if isinstance(c, dict))


def _card_or_payload_suit(c: Any) -> str:
    if isinstance(c, dict):
        return payload_suit(c)
    suit = str(getattr(c, "suit", ""))
    suit_long_to_short = {
        "clubs": "C",
        "diamonds": "D",
        "hearts": "H",
        "spades": "S",
    }
    return suit_long_to_short.get(suit, suit)


def _card_or_payload_run_rank(c: Any) -> int:
    if isinstance(c, dict):
        return payload_run_rank_value(c)
    if bool(getattr(c, "is_joker", False)):
        return 99
    return int(getattr(c, "rank", 0) or 0)


def table_meld_near_run_count(
    card_payload: dict,
    table_melds: list[dict] | None,
    max_gap: int = RUN_PROJECT_MAX_RANK_GAP,
) -> int:
    """Count visible table runs this card is close to extending.

    Distance 1 means the card can extend now. Distance 2 means it is one missing
    rank away from extending, e.g. table A-2-3 of spades and hand 5S.
    """
    if not isinstance(card_payload, dict) or payload_is_joker(card_payload):
        return 0
    suit = payload_suit(card_payload)
    rank = payload_run_rank_value(card_payload)
    if not suit or rank <= 0 or rank > 13:
        return 0
    count = 0
    for meld in table_melds or []:
        if not isinstance(meld, dict) or str(meld.get("kind")) != "escalera":
            continue
        cards = [c for c in (meld.get("cards") or []) if not _card_or_payload_is_joker(c)]
        ranks = sorted(
            _card_or_payload_run_rank(c)
            for c in cards
            if _card_or_payload_suit(c) == suit
        )
        ranks = [r for r in ranks if 1 <= r <= 13]
        if not ranks:
            continue
        if rank in ranks:
            continue
        lo = ranks[0]
        hi = ranks[-1]
        if 1 <= (lo - rank) <= max_gap or 1 <= (rank - hi) <= max_gap:
            count += 1
    return count


# ===== Memory dataclass =====

def _card_unique_id(card_payload: dict) -> tuple:
    """Stable unique identifier (rank, suit, deck_id, is_joker) for dedup."""
    if bool(card_payload.get("joker")):
        return (0, None, int(card_payload.get("deck_id", -1)), True)
    raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    if raw in rank_map:
        rank_int = rank_map[raw]
    else:
        try:
            rank_int = int(raw)
        except ValueError:
            rank_int = 0
    suit_short_to_long = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}
    suit_raw = str(card_payload.get("suit", ""))
    suit = suit_short_to_long.get(suit_raw, suit_raw)
    return (rank_int, suit, int(card_payload.get("deck_id", -1)), False)


@dataclass
class SmartObsMemory:
    """Cross-step memory needed to build the smart obs.

    Defaults are sane for fresh sessions / round resets. Both env and adapter
    populate the same shape.
    """
    num_players: int = 3
    discard_history_window: int = 2
    target_points: int = 100
    max_reenganches: int = 2

    seen_token_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(53, dtype=np.float32)
    )
    # Set of unique card identifiers we've publicly observed this round.
    # Dedup avoids the legacy bug of saturating seen counts via repeated own-hand counting.
    seen_unique_ids: set = field(default_factory=set)
    recent_discards_by_player: list[list[int]] = field(
        default_factory=lambda: [[] for _ in range(3)]
    )
    own_action_history: deque[str] = field(
        default_factory=lambda: deque(maxlen=OWN_ACTION_HISTORY_LEN)
    )

    match_scores: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    reenganches_used: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.int32)
    )
    eliminated: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )

    def reset_round(self) -> None:
        """Reset per-round memory but keep cross-round (match) state."""
        self.seen_token_counts = np.zeros(53, dtype=np.float32)
        self.seen_unique_ids = set()
        self.recent_discards_by_player = [[] for _ in range(self.num_players)]
        self.own_action_history.clear()

    def reset_match(self) -> None:
        """Reset everything (start of a new match)."""
        self.reset_round()
        self.match_scores = np.zeros(self.num_players, dtype=np.float32)
        self.reenganches_used = np.zeros(self.num_players, dtype=np.int32)
        self.eliminated = np.zeros(self.num_players, dtype=bool)

    def record_seen_card(self, card_payload: dict | None) -> None:
        """Mark a publicly visible card as seen. Dedup by unique identity so calling
        this repeatedly for the same physical card does NOT inflate the count.
        """
        if not isinstance(card_payload, dict):
            return
        uid = _card_unique_id(card_payload)
        if uid in self.seen_unique_ids:
            return
        self.seen_unique_ids.add(uid)
        token = card_token(card_payload)
        self.seen_token_counts[token] += 1.0
        np.clip(self.seen_token_counts, 0.0, 4.0, out=self.seen_token_counts)

    def record_discard_event(self, actor_seat: int, card_payload: dict) -> None:
        if actor_seat < 0 or actor_seat >= self.num_players:
            return
        if not isinstance(card_payload, dict):
            return
        token = card_token(card_payload)
        if token < 0 or token >= 53:
            return
        self.recent_discards_by_player[actor_seat].append(int(token))
        self.recent_discards_by_player[actor_seat] = self.recent_discards_by_player[
            actor_seat
        ][-self.discard_history_window :]
        # Mark as seen (deduped).
        self.record_seen_card(card_payload)

    def record_own_action(self, action_type: str) -> None:
        if action_type in OWN_ACTION_TYPES:
            self.own_action_history.append(action_type)


# ===== Sub-vector builders =====

def phase_vector(phase: str) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    vec[PHASE_TO_INDEX.get(phase, 0)] = 1.0
    return vec


def build_base_obs(payload: dict[str, Any]) -> np.ndarray:
    """111 + 2*N base features (53 hand, 53 top_discard, N counts, N opened, 1 stock, 3 phase, 1 pad)."""
    hand = np.zeros(53, dtype=np.float32)
    for c in payload.get("hand", []) or []:
        if isinstance(c, dict):
            hand[card_token(c)] += 1.0
    hand /= 4.0

    top_discard = np.zeros(53, dtype=np.float32)
    discard = payload.get("discard_top")
    if isinstance(discard, dict):
        top_discard[card_token(discard)] += 1.0

    my_seat = int(payload.get("seat", 0))
    num_players = max(2, int(payload.get("num_players", 2)))
    other_sizes = payload.get("other_hand_sizes") or []
    laid = payload.get("has_laid_meld_this_round") or []
    seat_order = list(range(my_seat, num_players)) + list(range(0, my_seat))
    counts_values = []
    opened_values = []
    for seat in seat_order:
        size = float(other_sizes[seat]) if seat < len(other_sizes) else 0.0
        counts_values.append(size / 10.0)
        opened_values.append(1.0 if (seat < len(laid) and bool(laid[seat])) else 0.0)

    stock = float(payload.get("stock_size", 0)) / 108.0
    phase = phase_vector(str(payload.get("phase", "draw")))
    pad = np.zeros(1, dtype=np.float32)
    return np.concatenate(
        [
            hand,
            top_discard,
            np.array(counts_values, dtype=np.float32),
            np.array(opened_values, dtype=np.float32),
            np.array([stock], dtype=np.float32),
            phase,
            pad,
        ]
    ).astype(np.float32)


def build_seen_normalized(seen_token_counts: np.ndarray) -> np.ndarray:
    """53 features: per-token sighting count clamped & normalized."""
    return (np.clip(seen_token_counts, 0.0, 4.0) / 4.0).astype(np.float32)


def build_recent_discards(
    recent_discards_by_player: list[list[int]],
    viewer_seat: int,
    num_players: int,
    window: int,
) -> np.ndarray:
    """53 * window * (N-1) features: per-rival, last K discards as one-hot tokens.

    Slots are ROTATED by viewer_seat: out[0..53*window) = clockwise neighbor of
    viewer (viewer+1), out[53*window..) = next neighbor, etc. Consistent with
    the rotation used in build_base_obs and build_match_features.
    """
    num_rivals = max(0, num_players - 1)
    out = np.zeros(53 * window * num_rivals, dtype=np.float32)
    cursor = 0
    for offset in range(1, num_players):
        seat = (viewer_seat + offset) % num_players
        if seat < len(recent_discards_by_player):
            hist = list(recent_discards_by_player[seat])[-window:]
        else:
            hist = []
        # left-pad with -1 so newest discards land in the same slot regardless of count
        hist = ([-1] * (window - len(hist))) + hist
        for token in hist:
            if 0 <= token < 53:
                out[cursor + token] = 1.0
            cursor += 53
    return out


def build_hand_project_features(hand_payload: list[dict]) -> np.ndarray:
    """3 features: pair_count, near_run_count, dead_card_count (all normalized)."""
    rank_counts: dict[int, int] = {}
    suit_to_ranks: dict[str, list[int]] = {}
    for c in hand_payload:
        if not isinstance(c, dict):
            continue
        if payload_is_joker(c):
            continue
        rv = payload_rank_value(c)
        if rv <= 0:
            continue
        rank_counts[rv] = rank_counts.get(rv, 0) + 1
        suit_to_ranks.setdefault(payload_suit(c), []).append(rv)

    pair_count = sum(1 for v in rank_counts.values() if v >= 2)
    near_run_count = 0
    for ranks in suit_to_ranks.values():
        uniq = sorted(set(ranks))
        for idx in range(1, len(uniq)):
            if abs(uniq[idx] - uniq[idx - 1]) == 1:
                near_run_count += 1
    dead_card_count = sum(
        1 for c in hand_payload if isinstance(c, dict) and not is_project_card_payload(c, hand_payload)
    )

    return np.array(
        [
            min(1.0, pair_count / 6.0),
            min(1.0, near_run_count / 8.0),
            min(1.0, dead_card_count / 14.0),
        ],
        dtype=np.float32,
    )


def build_match_features(
    match_scores: np.ndarray,
    reenganches_used: np.ndarray,
    eliminated: np.ndarray,
    viewer_seat: int,
    num_players: int,
    target_points: int,
    max_reenganches: int,
) -> np.ndarray:
    """3*N + 1 features (default for 3p = 10): scores, reenganches, eliminated, gap_to_best."""
    n = num_players
    order = list(range(viewer_seat, n)) + list(range(0, viewer_seat))
    rotated_scores = np.asarray(match_scores)[order]
    rotated_reeng = np.asarray(reenganches_used)[order]
    rotated_elim = np.asarray(eliminated)[order]
    scores_norm = (rotated_scores.astype(np.float32) / float(target_points)).astype(np.float32)
    reeng_norm = (
        rotated_reeng.astype(np.float32) / max(1.0, float(max_reenganches))
    ).astype(np.float32)
    elim_norm = rotated_elim.astype(np.float32)
    my_score = float(scores_norm[0])
    active_others = [
        float(scores_norm[i]) for i in range(1, n) if not bool(rotated_elim[i])
    ]
    best_other = min(active_others) if active_others else my_score
    gap = best_other - my_score
    return np.concatenate(
        [scores_norm, reeng_norm, elim_norm, np.array([gap], dtype=np.float32)]
    ).astype(np.float32)


def build_table_features(
    melds: list[dict],
    viewer_seat: int,
    num_players: int,
) -> np.ndarray:
    """53 + 2*N + 4 features: card-presence + cards-laid + melds-by-owner + aggregates."""
    n = num_players
    card_presence = np.zeros(TABLE_FEATURE_BASE, dtype=np.float32)
    cards_laid = np.zeros(n, dtype=np.float32)
    melds_per_owner = np.zeros(n, dtype=np.float32)
    escalera_count = 0
    pierna_count = 0
    escalera_left_joker = 0
    escalera_right_joker = 0

    for meld in melds or []:
        if not isinstance(meld, dict):
            continue
        owner = int(meld.get("owner", 0))
        rotated_owner = (owner - viewer_seat) % n
        cards = meld.get("cards", []) or []
        kind = str(meld.get("kind", ""))

        for c in cards:
            # In the env, cards is list[Card]; in deployment, list[dict].
            token = _card_or_payload_token(c)
            if 0 <= token < TABLE_FEATURE_BASE:
                card_presence[token] += 1.0

        cards_laid[rotated_owner] += float(len(cards))
        melds_per_owner[rotated_owner] += 1.0

        if kind == "escalera":
            escalera_count += 1
            if cards and _card_or_payload_is_joker(cards[0]):
                escalera_left_joker += 1
            if cards and _card_or_payload_is_joker(cards[-1]):
                escalera_right_joker += 1
        elif kind == "pierna":
            pierna_count += 1

    card_presence = np.clip(card_presence / 4.0, 0.0, 1.0)
    cards_laid_norm = np.clip(cards_laid / 14.0, 0.0, 1.0)
    melds_norm = np.clip(melds_per_owner / 5.0, 0.0, 1.0)
    aggregates = np.array(
        [
            min(1.0, escalera_count / 8.0),
            min(1.0, pierna_count / 8.0),
            min(1.0, escalera_left_joker / 4.0),
            min(1.0, escalera_right_joker / 4.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [card_presence, cards_laid_norm, melds_norm, aggregates]
    ).astype(np.float32)


def build_stock_remaining(
    payload: dict[str, Any],
    melds: list[dict] | None = None,
    num_decks: int = 2,
    num_jokers: int = 4,
) -> np.ndarray:
    """53 features: lower-bound estimate of copies of each token still unseen.

    Counts ONLY currently visible state (deterministic, no double-counting risk):
    - own hand
    - top of discard pile
    - all cards in melds on table

    This under-counts (we don't see covered discards or other players' hands) but
    gives the agent a reliable feature for "what's at least known to be elsewhere".
    The legacy version used seen_token_counts which saturated at 4 due to the env
    re-counting own hand on every step — making 'remaining' return 0 even when
    multiple copies were genuinely still in the deck.
    """
    visible = np.zeros(53, dtype=np.float32)
    for c in payload.get("hand", []) or []:
        if isinstance(c, dict):
            visible[card_token(c)] += 1.0
    top = payload.get("discard_top")
    if isinstance(top, dict):
        visible[card_token(top)] += 1.0
    for meld in melds or []:
        if not isinstance(meld, dict):
            continue
        for c in meld.get("cards", []) or []:
            visible[_card_or_payload_token(c)] += 1.0

    max_per_token = np.full(53, float(num_decks), dtype=np.float32)
    max_per_token[52] = float(num_jokers)
    visible = np.minimum(visible, max_per_token)
    remaining = max_per_token - visible
    return (remaining / np.maximum(max_per_token, 1.0)).astype(np.float32)


def build_legal_action_types(
    legal_actions: list[dict] | None,
    max_slots: int = LEGAL_ACTION_PREVIEW_SLOTS,
) -> np.ndarray:
    """For each of the first `max_slots` slots in the canonicalized legal-action
    list, encode the action type as one-hot. Slots past `len(legal_actions)` are
    all-zero (effective padding).

    Layout: (max_slots * OWN_ACTION_TYPE_DIM,) float32, slot-major order.
    Slot i type vector lives at out[i*8 : (i+1)*8].
    """
    out = np.zeros((max_slots, OWN_ACTION_TYPE_DIM), dtype=np.float32)
    if not legal_actions:
        return out.flatten()
    for i, action in enumerate(legal_actions[:max_slots]):
        if not isinstance(action, dict):
            continue
        idx = ACTION_TYPE_TO_INDEX.get(str(action.get("type", "")))
        if idx is not None:
            out[i, idx] = 1.0
    return out.flatten()


def _remove_first_matching_card(hand_payload: list[dict], card_payload: dict) -> list[dict]:
    target = payload_key(card_payload)
    removed = False
    out: list[dict] = []
    for c in hand_payload:
        if not removed and isinstance(c, dict) and payload_key(c) == target:
            removed = True
            continue
        out.append(c)
    return out


def _cards_from_action(action: dict[str, Any]) -> list[dict]:
    cards = action.get("cards")
    if isinstance(cards, list):
        return [c for c in cards if isinstance(c, dict)]
    card = action.get("card")
    if isinstance(card, dict):
        return [card]
    return []


def _apply_action_to_hand_approx(hand_payload: list[dict], action: dict[str, Any]) -> list[dict]:
    """Approximate post-action hand using payload card identities.

    This is intentionally deterministic and conservative. It is used only for
    tactical features, not game state. DrawDiscard actions in remote observations
    usually already include the drawn discard in hand; if not, removing unmatched
    cards is harmless.
    """
    action_type = str(action.get("type", ""))
    if action_type == "DrawStock":
        # Unknown drawn card, approximate +1 average card by leaving points unchanged
        # and only increasing size in feature code.
        return list(hand_payload)
    if action_type == "DrawDiscard":
        nested = action.get("play")
        if isinstance(nested, dict):
            return _apply_action_to_hand_approx(hand_payload, nested)
        return list(hand_payload)

    out = list(hand_payload)
    if action_type in {"Discard", "Cruzar", "ExtendMeld", "MoveJoker"}:
        card = action.get("card")
        if isinstance(card, dict):
            return _remove_first_matching_card(out, card)
        return out
    if action_type in {"LayPierna", "LayEscalera"}:
        for c in _cards_from_action(action):
            out = _remove_first_matching_card(out, c)
        return out
    return out


def _action_consumed_cards_count(action: dict[str, Any]) -> int:
    action_type = str(action.get("type", ""))
    if action_type in {"LayPierna", "LayEscalera"}:
        return len(_cards_from_action(action))
    if action_type in {"Discard", "Cruzar", "ExtendMeld", "MoveJoker"}:
        return 1 if isinstance(action.get("card"), dict) else 0
    if action_type == "DrawDiscard":
        nested = action.get("play")
        if isinstance(nested, dict):
            return max(0, _action_consumed_cards_count(nested) - 1)
    return 0


def _action_removed_points(action: dict[str, Any]) -> int:
    action_type = str(action.get("type", ""))
    if action_type == "DrawDiscard":
        nested = action.get("play")
        if isinstance(nested, dict):
            return _action_removed_points(nested)
        return 0
    return sum(payload_point_value(c) for c in _cards_from_action(action))


def _best_high_dead_discard(
    hand_payload: list[dict],
    table_melds: list[dict] | None = None,
) -> tuple[int, bool]:
    best = 0
    any_project = False
    for c in hand_payload:
        if not isinstance(c, dict):
            continue
        facts = project_breakdown(c, hand_payload, table_melds=table_melds)
        any_project = any_project or bool(facts["is_project_card"])
        if payload_is_joker(c):
            continue
        if not bool(facts["is_project_card"]):
            best = max(best, payload_point_value(c))
    return best, any_project


def describe_action_tactics(
    action: dict[str, Any],
    hand_payload: list[dict],
    current_score: float = 0.0,
    target_points: int = 100,
    table_melds: list[dict] | None = None,
) -> dict[str, Any]:
    """Human-readable tactical facts for a single legal action."""
    action_type = str(action.get("type", ""))
    hand = [c for c in hand_payload if isinstance(c, dict)]
    hand_size = len(hand)
    hand_points_before = hand_points_payload(hand)
    post_hand = _apply_action_to_hand_approx(hand, action)
    post_hand_size = len(post_hand) + (1 if action_type == "DrawStock" else 0)
    post_points = hand_points_payload(post_hand)
    if action_type == "DrawStock":
        # Unknown card; use a neutral mid-card estimate so the feature still says
        # "drawing increases exposure" without pretending to know the card.
        post_points += 7

    card = action.get("card")
    if action_type == "DrawDiscard":
        nested = action.get("play")
        if isinstance(nested, dict):
            card = nested.get("card")
    if not isinstance(card, dict):
        cards = _cards_from_action(action)
        card = cards[0] if cards else {}

    facts = (
        project_breakdown(card, hand, table_melds=table_melds)
        if isinstance(card, dict)
        else project_breakdown({}, hand, table_melds=table_melds)
    )
    best_dead, any_project = _best_high_dead_discard(hand, table_melds=table_melds)
    card_points = payload_point_value(card) if isinstance(card, dict) else 0
    breaks_project = False
    if action_type in {"Discard", "Cruzar"} and bool(facts["is_project_card"]):
        # Discarding one of exactly two same-rank cards destroys the remaining
        # pair project. Likewise, discarding a card with only one run neighbor
        # often turns the neighbor into a singleton.
        breaks_pair = bool(facts["is_pair_project"]) and int(facts["same_rank_count"]) <= 1
        breaks_run = bool(facts["is_run_project"]) and int(facts["same_suit_neighbor_count"]) <= 1
        breaks_project = breaks_pair or breaks_run
    has_high_dead_alt = bool(best_dead > card_points and bool(facts["is_project_card"]))
    unsafe_excess = max(0.0, float(current_score) + float(post_points) - float(target_points))
    consumed = _action_consumed_cards_count(action)
    go_out = hand_size > 0 and consumed >= hand_size
    return {
        "action_type": action_type,
        "is_discard": action_type == "Discard",
        "is_cruzar": action_type == "Cruzar",
        "is_draw_stock": action_type == "DrawStock",
        "is_draw_discard": action_type == "DrawDiscard",
        "is_new_meld": action_type in {"LayPierna", "LayEscalera"},
        "is_extend": action_type == "ExtendMeld",
        "is_move_joker": action_type == "MoveJoker",
        "is_go_out": go_out,
        "card_points": card_points,
        "card_rank": payload_rank_value(card) if isinstance(card, dict) else 0,
        "is_joker": payload_is_joker(card) if isinstance(card, dict) else False,
        "same_rank_count": int(facts["same_rank_count"]),
        "same_suit_neighbor_count": int(facts["same_suit_neighbor_count"]),
        "is_pair_project": bool(facts["is_pair_project"]),
        "is_run_project": bool(facts["is_run_project"]),
        "is_project_card": bool(facts["is_project_card"]),
        "breaks_project": breaks_project,
        "has_high_dead_alternative": has_high_dead_alt,
        "best_dead_card_points": best_dead,
        "any_project_available": any_project,
        "discard_value_gap_to_best_dead": max(0, best_dead - card_points),
        "hand_points_before": hand_points_before,
        "post_action_hand_points": post_points,
        "post_action_hand_size": post_hand_size,
        "post_action_unsafe_excess": unsafe_excess,
        "meld_cards_count": consumed if action_type in {"LayPierna", "LayEscalera", "DrawDiscard"} else 0,
        "meld_points_removed": _action_removed_points(action),
        "keeps_or_enters_zone": (float(current_score) + float(post_points)) <= float(target_points),
    }


def build_legal_action_tactical_features(
    legal_actions: list[dict] | None,
    hand_payload: list[dict],
    current_score: float = 0.0,
    target_points: int = 100,
    table_melds: list[dict] | None = None,
    max_slots: int = LEGAL_ACTION_PREVIEW_SLOTS,
) -> np.ndarray:
    out = np.zeros((max_slots, ACTION_TACTICAL_FEATURE_DIM), dtype=np.float32)
    if not legal_actions:
        return out.flatten()
    for i, action in enumerate(legal_actions[:max_slots]):
        if not isinstance(action, dict):
            continue
        t = describe_action_tactics(
            action,
            hand_payload=hand_payload,
            current_score=current_score,
            target_points=target_points,
            table_melds=table_melds,
        )
        values = [
            1.0,
            float(t["is_discard"]),
            float(t["is_cruzar"]),
            float(t["is_draw_stock"]),
            float(t["is_draw_discard"]),
            float(t["is_new_meld"]),
            float(t["is_extend"]),
            float(t["is_move_joker"]),
            float(t["is_go_out"]),
            min(1.0, float(t["card_points"]) / 15.0),
            min(1.0, float(t["card_rank"]) / 14.0),
            float(t["is_joker"]),
            min(1.0, float(t["same_rank_count"]) / 3.0),
            min(1.0, float(t["same_suit_neighbor_count"]) / 4.0),
            float(t["is_pair_project"]),
            float(t["is_run_project"]),
            float(t["is_project_card"]),
            float(t["breaks_project"]),
            float(t["has_high_dead_alternative"]),
            min(1.0, float(t["discard_value_gap_to_best_dead"]) / 15.0),
            min(2.0, float(t["post_action_hand_points"]) / float(max(1, target_points))),
            min(1.0, float(t["post_action_hand_size"]) / 14.0),
            min(1.0, float(t["post_action_unsafe_excess"]) / float(max(1, target_points))),
            min(1.0, float(t["meld_cards_count"]) / 14.0),
            min(1.0, float(t["meld_points_removed"]) / float(max(1, target_points))),
            float(t["keeps_or_enters_zone"]),
        ]
        out[i, :] = np.asarray(values, dtype=np.float32)
    return out.flatten()


def build_own_action_history(action_history: deque[str]) -> np.ndarray:
    """OWN_ACTION_HISTORY_LEN * OWN_ACTION_TYPE_DIM = 4*8 = 32 features.

    Most recent action ends up in the LAST slot (right-aligned), so the
    "freshest" decision is in a stable position regardless of how many actions
    have been taken so far.
    """
    out = np.zeros((OWN_ACTION_HISTORY_LEN, OWN_ACTION_TYPE_DIM), dtype=np.float32)
    if action_history:
        recent = list(action_history)[-OWN_ACTION_HISTORY_LEN:]
        offset = OWN_ACTION_HISTORY_LEN - len(recent)  # right-align
        for i, action_type in enumerate(recent):
            try:
                idx = OWN_ACTION_TYPES.index(action_type)
            except ValueError:
                continue
            out[offset + i, idx] = 1.0
    return out.flatten()


# ===== Top-level builders =====

def build_smart_base_obs(
    payload: dict[str, Any],
    memory: SmartObsMemory,
) -> np.ndarray:
    """The 'parent' smart obs: base + seen + recent_discards + hand_features.

    For 3 players, default window=2: 117 + 53 + 212 + 3 = 385 dims.
    """
    base = build_base_obs(payload)
    seen = build_seen_normalized(memory.seen_token_counts)
    recent = build_recent_discards(
        memory.recent_discards_by_player,
        viewer_seat=int(payload.get("seat", 0)),
        num_players=memory.num_players,
        window=memory.discard_history_window,
    )
    hand_features = build_hand_project_features(payload.get("hand", []) or [])
    return np.concatenate([base, seen, recent, hand_features]).astype(np.float32)


def build_smart_match_obs(
    payload: dict[str, Any],
    memory: SmartObsMemory,
    melds: list[dict] | None = None,
    legal_actions: list[dict] | None = None,
    include_action_tactics: bool = True,
) -> np.ndarray:
    """Full smart obs for the match-aware env / deployment.

    Layout (3p, default window=2):
      385  parent base
      +10  match features (scores, reenganches, eliminated, gap)
      +63  table features (card-presence, owner aggregates, type counts)
      +53  stock-remaining
      +32  own-action history
      +256 legal-action types (32 slots × 8 types one-hot)
      +832 optional action tactics (32 slots × 26 tactical features)
      = 1631 total features with action tactics, 799 without
    """
    viewer_seat = int(payload.get("seat", 0))

    base = build_smart_base_obs(payload, memory)
    match = build_match_features(
        match_scores=memory.match_scores,
        reenganches_used=memory.reenganches_used,
        eliminated=memory.eliminated,
        viewer_seat=viewer_seat,
        num_players=memory.num_players,
        target_points=memory.target_points,
        max_reenganches=memory.max_reenganches,
    )
    effective_melds = melds if melds is not None else (payload.get("melds_on_table") or [])
    table = build_table_features(
        melds=effective_melds,
        viewer_seat=viewer_seat,
        num_players=memory.num_players,
    )
    stock = build_stock_remaining(payload, melds=effective_melds)
    own = build_own_action_history(memory.own_action_history)
    effective_legal = legal_actions if legal_actions is not None else (payload.get("legal_actions") or [])
    legal_types = build_legal_action_types(effective_legal)
    current_score = 0.0
    try:
        current_score = float(memory.match_scores[viewer_seat])
    except Exception:
        current_score = 0.0
    legal_tactics = (
        build_legal_action_tactical_features(
            effective_legal,
            hand_payload=payload.get("hand", []) or [],
            current_score=current_score,
            target_points=int(memory.target_points),
            table_melds=effective_melds,
        )
        if include_action_tactics
        else np.zeros(0, dtype=np.float32)
    )
    return np.concatenate(
        [base, match, table, stock, own, legal_types, legal_tactics]
    ).astype(np.float32)


def smart_match_obs_dim(
    num_players: int = 3,
    discard_history_window: int = 2,
    include_action_tactics: bool = True,
) -> int:
    """Compute the expected obs dim for the match-aware smart env."""
    base_dim = 111 + 2 * num_players                   # base (counts/opened scale with N)
    seen_dim = 53
    recent_dim = 53 * discard_history_window * max(0, num_players - 1)
    hand_features_dim = 3
    parent_dim = base_dim + seen_dim + recent_dim + hand_features_dim
    match_dim = num_players * MATCH_FEATURE_DIM_PER_PLAYER + MATCH_FEATURE_EXTRA
    table_dim = TABLE_FEATURE_BASE + 2 * num_players + TABLE_FEATURE_AGG
    stock_dim = 53
    own_dim = OWN_ACTION_FEATURE_DIM
    legal_types_dim = LEGAL_ACTION_TYPES_FEATURE_DIM
    legal_tactics_dim = LEGAL_ACTION_TACTICAL_FEATURE_DIM if include_action_tactics else 0
    return (
        parent_dim
        + match_dim
        + table_dim
        + stock_dim
        + own_dim
        + legal_types_dim
        + legal_tactics_dim
    )


# ===== Internal helpers =====

def _card_or_payload_token(c: Any) -> int:
    """Token from either a Card object or a payload dict."""
    if isinstance(c, dict):
        return card_token(c)
    return int(getattr(c, "token", 52))


def _card_or_payload_is_joker(c: Any) -> bool:
    if isinstance(c, dict):
        return payload_is_joker(c)
    return bool(getattr(c, "is_joker", False))
