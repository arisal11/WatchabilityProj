"""Builds the watchability score (the target variable `y`) from completed games.

Only ever applied to games that have already been played — it uses outcome data
(final score, win-probability swings) that does not exist for a future game.
"""

import pandas as pd

# Rivalry was 0.15 until 2026-09-24. Division games turned out to be no more dramatic
# than other games, so the old weight handed them a ~15-point bonus on rivalry alone.
# Cut to 0.05 (a small tiebreaker) and the rest re-spread proportionally.
DEFAULT_WEIGHTS = {
    "excitement_weighted": 0.335,
    "comeback_factor": 0.17,
    "fireworks": 0.17,
    "playoff_stakes": 0.275,
    "rivalry": 0.05,
}


def completed_games(master: pd.DataFrame) -> pd.DataFrame:
    return master[master["home_score"].notna() & master["excitement_weighted"].notna()].copy()


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def to_unit_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Put every signal on a comparable 0-1 scale.

    Continuous signals use percentile rank rather than min-max so a single
    outlier game cannot compress everything else into a narrow band.
    """
    out = pd.DataFrame(index=df.index)
    out["excitement_weighted"] = df["excitement_weighted"].rank(pct=True)
    out["comeback_factor"] = df["comeback_factor"].rank(pct=True)
    out["fireworks"] = df["fireworks"].rank(pct=True)
    out["playoff_stakes"] = df["playoff_stakes"] / 4.0
    out["rivalry"] = df["rivalry"].astype(float)
    return out


def build_label(master: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    games = completed_games(master)
    scaled = to_unit_scale(games)

    score = sum(scaled[col] * w for col, w in weights.items())
    games["watchability_score"] = (score / sum(weights.values()) * 100).round(2)
    return games
