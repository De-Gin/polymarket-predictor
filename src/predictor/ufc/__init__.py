"""UFC adapter: fight winner prediction.

Public API:
    UfcEloSystem      — fighter rating system
    UfcDataLoader     — CSV-based data loader (pandas)
    FighterProfile    — a fighter's rating + attribute snapshot
    extract_fight_vectors — builds Vectors for a bout
"""

from predictor.ufc.elo import UfcEloSystem
from predictor.ufc.features import FighterProfile, FightContext, extract_fight_vectors

__all__ = [
    "UfcEloSystem",
    "FighterProfile",
    "FightContext",
    "extract_fight_vectors",
]
