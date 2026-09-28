"""Guards against the nflverse column traps that caused real bugs."""

import pandas as pd
import pytest

from drama_features import compute_drama


def _game(home_wp, final_home, final_away, wp=None):
    n = len(home_wp)
    return pd.DataFrame({
        "game_id": "g", "play_type": "pass", "home_wp": home_wp,
        "wp": wp if wp is not None else home_wp,
        "wpa": pd.Series(home_wp).diff().fillna(0).values,
        "game_seconds_remaining": [3600 - i * 3600 / (n - 1) for i in range(n)],
        # nflverse repeats the FINAL score on every row
        "home_score": final_home, "away_score": final_away,
    })


def test_comeback_uses_winner_frame():
    # Home team falls to 5% then wins -> winner's min WP is 0.05
    out = compute_drama(_game([0.5, 0.2, 0.05, 0.6, 1.0], 28, 24)).iloc[0]
    assert out.winner_min_wp == pytest.approx(0.05)


def test_away_comeback_is_measured_from_away_side():
    # Home led 97% but lost: the AWAY winner bottomed at 3%
    out = compute_drama(_game([0.5, 0.8, 0.97, 0.4, 0.0], 24, 28)).iloc[0]
    assert out.winner_min_wp == pytest.approx(0.03)


def test_possession_flipping_wp_is_ignored():
    # `wp` flips with possession; results must depend only on home_wp
    hw = [0.5, 0.2, 0.05, 0.6, 1.0]
    a = compute_drama(_game(hw, 28, 24)).iloc[0]
    b = compute_drama(_game(hw, 28, 24, wp=[0.5, 0.8, 0.05, 0.4, 1.0])).iloc[0]
    assert a.winner_min_wp == b.winner_min_wp


def test_late_swings_count_more():
    early = compute_drama(_game([0.5, 0.9, 0.9, 0.9, 1.0], 28, 24)).iloc[0]
    late = compute_drama(_game([0.5, 0.5, 0.5, 0.9, 1.0], 28, 24)).iloc[0]
    assert early.excitement_index == pytest.approx(late.excitement_index)
    assert late.excitement_weighted > early.excitement_weighted


def test_no_play_rows_are_excluded():
    g = _game([0.5, 0.2, 0.05, 0.6, 1.0], 28, 24)
    g.loc[2, "play_type"] = "no_play"
    assert compute_drama(g).iloc[0].winner_min_wp == pytest.approx(0.2)
