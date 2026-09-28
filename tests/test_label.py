import numpy as np
import pandas as pd

from build_label import DEFAULT_WEIGHTS, build_label


def _master(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "home_score": rng.integers(0, 40, n).astype(float),
        "excitement_weighted": rng.gamma(2, 3, n),
        "comeback_factor": 1 + rng.gamma(1, 0.5, n),
        "fireworks": rng.integers(10, 80, n),
        "playoff_stakes": rng.choice([0, 0, 0, 1, 4], n),
        "rivalry": rng.integers(0, 2, n),
    })


def test_score_is_bounded_0_to_100():
    y = build_label(_master())["watchability_score"]
    assert y.between(0, 100).all()


def test_unplayed_games_are_not_labeled():
    m = _master()
    m.loc[:4, "home_score"] = np.nan
    assert len(build_label(m)) == len(m) - 5


def test_one_extreme_game_cannot_dominate():
    """Why percentile, not z-score: an absurd outlier only moves one game to the top."""
    m = _master()
    base = build_label(m).set_index("game_id")["watchability_score"]
    m.loc[0, "excitement_weighted"] = 1e6
    out = build_label(m).set_index("game_id")["watchability_score"]
    assert (out.drop("g0") - base.drop("g0")).abs().max() < 1.0


def test_perfect_game_scores_100():
    m = _master()
    m.loc[0, ["excitement_weighted", "comeback_factor", "fireworks", "playoff_stakes", "rivalry"]] = [1e6, 1e6, 1e6, 4, 1]
    assert build_label(m).set_index("game_id").loc["g0", "watchability_score"] == 100.0
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
