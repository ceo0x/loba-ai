"""Typed actions for the Loba round state machine.

Every action carries the actual ``Card`` objects involved (rather than hand
indices). This keeps actions unambiguous regardless of how a player or model
chooses to represent its hand, and it makes legal-action enumeration trivial:
the engine produces a list of fully-specified ``Action`` objects, and the agent
picks one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .cards import Card


# ---------------------------------------------------------------------------
# Atomic play actions (the things you can do with cards on a turn other than draw/discard)


@dataclass(frozen=True)
class LayPierna:
    """Place a brand-new pierna on the table using these cards (3+ from hand or
    including a discard-top freshly drawn this turn)."""

    cards: tuple[Card, ...]


@dataclass(frozen=True)
class LayEscalera:
    """Place a brand-new escalera on the table using these cards."""

    cards: tuple[Card, ...]


@dataclass(frozen=True)
class ExtendMeld:
    """Add one card to an existing meld on the table.

    For escaleras this also covers extending at the low or high end. For piernas it
    adds a same-rank card in one of the suits already in use.
    """

    meld_id: int
    card: Card


@dataclass(frozen=True)
class MoveJoker:
    """Replace the joker that currently sits at one end of an escalera with the
    natural card it represented, and shift the joker to the opposite end."""

    meld_id: int
    replacement: Card


# Aggregate type for everything that can be played with the discard-top.
PlayAction = Union[LayPierna, LayEscalera, ExtendMeld, MoveJoker]


# ---------------------------------------------------------------------------
# Turn-phase actions


@dataclass(frozen=True)
class DrawStock:
    """Draw the top card of the stock pile into hand."""


@dataclass(frozen=True)
class DrawDiscard:
    """Take the top of the discard pile AND immediately play it via `play`.

    The card represented by `play` must include the current discard-top.
    """

    play: PlayAction


@dataclass(frozen=True)
class Discard:
    """Discard a card from hand. This ends the turn."""

    card: Card


@dataclass(frozen=True)
class Cruzar:
    """Discard a 'cruzar' card — a natural card whose rank matches an existing
    pierna and whose suit is the *missing 4th suit* of that pierna.

    Cruzar is purely a discard-time alternative to a normal :class:`Discard`.
    After this action is applied, the player gets *one bonus* :class:`Discard`
    (any card from hand, normal joker rule applies). Going out is allowed on
    either the cruzar discard or the bonus discard.

    Cruzar applies only to piernas; jokers cannot be cruzar cards.
    """

    card: Card


# Aggregate types
DrawAction = Union[DrawStock, DrawDiscard]
PostDrawAction = Union[LayPierna, LayEscalera, ExtendMeld, MoveJoker, Discard, Cruzar]
Action = Union[DrawAction, PostDrawAction]


# ---------------------------------------------------------------------------
# Helpers used by both engine and agents


def cards_in_play(action: PlayAction) -> tuple[Card, ...]:
    """Return every card touched by a play action."""
    if isinstance(action, (LayPierna, LayEscalera)):
        return action.cards
    if isinstance(action, ExtendMeld):
        return (action.card,)
    if isinstance(action, MoveJoker):
        return (action.replacement,)
    raise TypeError(action)