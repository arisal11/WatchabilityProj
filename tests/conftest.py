import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def games():
    """Tiny synthetic league: 4 teams, 2 seasons, each team plays every week. Last week unplayed."""
    rng = np.random.default_rng(0)
    teams, rows = ["AAA", "BBB", "CCC", "DDD"], []
    day = pd.Timestamp("2020-09-10")
    for season in (2020, 2021):
        for week in range(1, 7):
            pairs = [(0, 1), (2, 3)] if week % 2 else [(0, 2), (1, 3)]
            for h, a in pairs:
                rows.append(dict(game_id=f"{season}_{week:02d}_{teams[a]}_{teams[h]}", season=season, week=week,
                                 gameday=day.strftime("%Y-%m-%d"), home_team=teams[h], away_team=teams[a],
                                 home_score=float(rng.integers(0, 40)), away_score=float(rng.integers(0, 40)),
                                 game_type="REG", div_game=int(h == 0 and a == 1), spread_line=float(rng.normal(0, 5)),
                                 total_line=44.0, gametime="13:00", weekday="Sunday", home_rest=7, away_rest=7,
                                 roof="outdoors", temp=60.0, wind=5.0))
            day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["watchability_score"] = rng.uniform(0, 100, len(df))
    last = (df.season == 2021) & (df.week == 6)
    df.loc[last, ["home_score", "away_score", "watchability_score"]] = np.nan
    return df
