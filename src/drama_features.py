"""Drama metrics computed from play-by-play win probability.

All of these use outcome data, so they are LABEL ingredients only — never
model features. A game that has not been played has none of these.

`wp` in nflfastR is the win probability of whichever team has the ball, so it
flips on every change of possession. Anything measuring leads or comebacks must
use `home_wp`, which keeps a fixed reference frame.
"""

import numpy as np
import pandas as pd


def compute_drama(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[(pbp["play_type"] != "no_play") & pbp["home_wp"].notna()].copy()

    # Late swings matter more than 1st-quarter noise: weight each play by how
    # little time is left. 1.0 at kickoff rising to 2.0 at the final whistle.
    total_secs = 3600.0
    secs_left = plays["game_seconds_remaining"].clip(lower=0, upper=total_secs)
    plays["time_weight"] = 1.0 + (1.0 - secs_left / total_secs)
    plays["weighted_wpa"] = plays["wpa"].abs() * plays["time_weight"]

    grouped = plays.groupby("game_id")
    out = pd.DataFrame({
        "excitement_index": grouped["wpa"].apply(lambda s: s.abs().sum()),
        "excitement_weighted": grouped["weighted_wpa"].sum(),
        "max_swing": grouped["wpa"].apply(lambda s: s.abs().max()),
    })


    # Comeback factor: the lowest win probability the eventual WINNER fell to.
    # A team that was left for dead and came back scores very high.
    def comeback(g):
        final = g.iloc[-1]
        home_won = final["home_score"] > final["away_score"]
        winner_wp = g["home_wp"] if home_won else 1.0 - g["home_wp"]
        return float(winner_wp.min())

    out["winner_min_wp"] = grouped[["home_wp", "home_score", "away_score"]].apply(comeback)
    # Invert into a "how improbable was this win" scale, log-damped so a 0.4%
    # comeback doesn't produce a number 250x everything else.
    out["comeback_factor"] = np.log1p(1.0 / out["winner_min_wp"].clip(lower=0.001))

    return out.reset_index()
