"""NBA adapter: moneyline prediction for regular-season and playoff games.

Public API:
    NbaEloSystem   — maintains Elo ratings across a season
    NbaDataLoader  — fetches game logs via nba_api (cached on disk)
    extract_game_vectors — turns team state into a list of Vectors ready for VectorSpace
"""

from predictor.nba.elo import NbaEloSystem, TeamState
from predictor.nba.features import extract_game_vectors

__all__ = ["NbaEloSystem", "TeamState", "extract_game_vectors"]
