"""The most important property of the whole project: a game's features must only
use information available before kickoff."""

import numpy as np
import pandas as pd

from features import build_features

TARGET = "2021_04_CCC_AAA"


def _features_for(df, game_id):
    return build_features(df).set_index("game_id").loc[game_id]


def test_own_result_does_not_change_own_features(games):
    before = _features_for(games, TARGET)
    changed = games.copy()
    i = changed.game_id == TARGET
    changed.loc[i, ["home_score", "away_score", "watchability_score"]] = [99.0, 0.0, 100.0]
    pd.testing.assert_series_equal(before, _features_for(changed, TARGET))


def test_future_games_do_not_change_past_features(games):
    before = _features_for(games, TARGET)
    changed = games.copy()
    later = changed.gameday > changed.loc[changed.game_id == TARGET, "gameday"].iloc[0]
    changed.loc[later, ["home_score", "away_score", "watchability_score"]] = 0.0
    pd.testing.assert_series_equal(before, _features_for(changed, TARGET))


def test_past_games_do_change_features(games):
    """Guard against the leakage tests passing trivially (e.g. form features all NaN)."""
    before = _features_for(games, TARGET)
    changed = games.copy()
    earlier = changed.gameday < changed.loc[changed.game_id == TARGET, "gameday"].iloc[0]
    changed.loc[earlier, "home_score"] += 30
    assert not before.equals(_features_for(changed, TARGET))


def test_unplayed_games_get_form_features(games):
    f = build_features(games)
    upcoming = f[f.game_id.isin(games.loc[games.home_score.isna(), "game_id"])]
    assert len(upcoming) == 2
    assert upcoming[["sum_form_pf", "sum_form_y", "sum_season_win_pct"]].notna().all().all()


def test_season_win_pct_resets_each_season(games):
    f = build_features(games).set_index("game_id")
    week1 = f.loc[games.loc[(games.season == 2021) & (games.week == 1), "game_id"]]
    assert week1["sum_season_win_pct"].isna().all()
    assert week1["sum_form_pf"].notna().all()   # rolling form DOES carry over seasons
