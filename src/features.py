"""Pre-game features: only information available before kickoff.

Team-form features are built from each team's PREVIOUS games via shift(1),
so a game's own result never leaks into its own features.
"""

import numpy as np
import pandas as pd

STAKES = {"REG": 0, "WC": 1, "DIV": 2, "CON": 3, "SB": 4}
FORM_WINDOW = 8  # last 8 games, crossing season boundaries


def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling per-team stats from the team's prior games, one row per (game, team)."""
    # Unplayed games are included (with NaN results) so they still get a form row
    # built from the games before them; NaNs are skipped by the rolling means.
    g = games
    cols = ["game_id", "season", "gameday"]
    home = g[cols].assign(team=g["home_team"], pf=g["home_score"],
                          pa=g["away_score"], y=g["watchability_score"])
    away = g[cols].assign(team=g["away_team"], pf=g["away_score"],
                          pa=g["home_score"], y=g["watchability_score"])
    long = pd.concat([home, away]).sort_values(["team", "gameday"])
    long["margin"] = long["pf"] - long["pa"]
    long["win"] = (long["margin"] > 0).astype(float).where(long["margin"].notna())

    g = long.groupby("team")
    for c in ["pf", "pa", "margin", "win", "y"]:
        long[f"form_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    # Season-to-date win % (resets each season)
    long["season_win_pct"] = long.groupby(["team", "season"])["win"].transform(
        lambda s: s.shift(1).expanding().mean())
    keep = ["game_id", "team"] + [c for c in long.columns if c.startswith(("form_", "season_win"))]
    return long[keep]


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Game-level feature table. `games` must carry `watchability_score` (NaN where unplayed)."""
    df = games.copy()
    form = team_form(df)
    for side in ["home", "away"]:
        f = form.rename(columns={c: f"{side}_{c}" for c in form.columns if c not in ("game_id", "team")})
        df = df.merge(f, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"], how="left").drop(columns="team")

    out = pd.DataFrame({"game_id": df["game_id"], "season": df["season"], "week": df["week"]})
    # Market: Vegas already forecasts closeness (spread) and scoring (total)
    out["abs_spread"] = df["spread_line"].abs()
    out["total_line"] = df["total_line"]
    # Context
    out["playoff_stakes"] = df["game_type"].map(STAKES)
    out["div_game"] = df["div_game"]
    hour = pd.to_numeric(df["gametime"].str[:2], errors="coerce")
    out["primetime"] = ((hour >= 19) | df["weekday"].isin(["Monday", "Thursday"])).astype(int)
    out["rest_min"] = df[["home_rest", "away_rest"]].min(axis=1)
    out["roof"] = df["roof"]
    out["temp"] = df["temp"]
    out["wind"] = df["wind"]
    # Team form, combined into symmetric "both teams" features
    for c in ["form_pf", "form_pa", "form_margin", "form_y", "season_win_pct"]:
        h, a = df[f"home_{c}"], df[f"away_{c}"]
        out[f"sum_{c}"] = h + a
        out[f"gap_{c}"] = (h - a).abs()
    out["min_win_pct"] = df[["home_season_win_pct", "away_season_win_pct"]].min(axis=1)
    return out


def extra_features(games: pd.DataFrame) -> pd.DataFrame:
    """Experimental batch: team drama history, QB/coach continuity, last head-to-head.

    `games` needs `excitement_weighted`/`comeback_factor` (NaN where unplayed) and
    `watchability_score`. All values come from strictly earlier games.
    """
    g = games.sort_values("gameday").copy()
    for c in ["excitement_weighted", "comeback_factor"]:
        g[c + "_pct"] = g[c].rank(pct=True)
    sides = []
    for side in ["home", "away"]:
        sides.append(g[["game_id", "gameday", "excitement_weighted_pct", "comeback_factor_pct"]].assign(
            side=side, team=g[f"{side}_team"], qb=g[f"{side}_qb_id"], coach=g[f"{side}_coach"]))
    long = pd.concat(sides).sort_values(["team", "gameday"])
    t = long.groupby("team")
    for c in ["excitement_weighted_pct", "comeback_factor_pct"]:
        long["form_" + c] = t[c].transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    long["qb_change"] = (long["qb"] != t["qb"].shift(1)).astype(int)
    long["coach_games"] = long.groupby(["team", "coach"]).cumcount()

    wide = long.pivot(index="game_id", columns="side",
                      values=["form_excitement_weighted_pct", "form_comeback_factor_pct", "qb_change", "coach_games"])
    out = pd.DataFrame(index=wide.index)
    out["sum_form_excite"] = wide["form_excitement_weighted_pct"].sum(axis=1, min_count=2)
    out["sum_form_comeback"] = wide["form_comeback_factor_pct"].sum(axis=1, min_count=2)
    out["any_qb_change"] = wide["qb_change"].max(axis=1)
    out["min_coach_games"] = np.log1p(wide["coach_games"].min(axis=1))

    # Last meeting between the same two teams (either venue)
    g["pair"] = [tuple(sorted(p)) for p in zip(g["home_team"], g["away_team"])]
    g["h2h_last_y"] = g.groupby("pair")["watchability_score"].shift(1)
    out = out.join(g.set_index("game_id")[["h2h_last_y"]])
    return out.reset_index()


QB_WINDOW = 16      # games of history per QB
QB_SHRINK_K = 150   # dropbacks of "prior" blended in
QB_PRIOR = -0.05    # unknown QBs are usually backups/rookies: slightly below average
STREAK_WINDOW, STREAK_K = 3, 60   # "last 3 games" form, lightly shrunk


def qb_quality(games: pd.DataFrame, qb_plays: pd.DataFrame,
               window: int = QB_WINDOW, k: int = QB_SHRINK_K) -> pd.DataFrame:
    """Each starting QB's EPA per dropback over his previous QB_WINDOW games, before this game.

    Shrinkage: (sum_epa + K*prior) / (dropbacks + K). With 20 dropbacks the prior
    dominates; with 600 the QB's own record does. It's the same idea as Ridge pulling
    coefficients toward 0 -- trust noisy small-sample estimates less.
    """
    per_game = (qb_plays.dropna(subset=["passer_player_id"])
                .groupby(["game_id", "passer_player_id"])
                .agg(n=("qb_epa", "size"), epa=("qb_epa", "sum")).reset_index())
    per_game = per_game.merge(games[["game_id", "gameday"]], on="game_id").sort_values("gameday")

    # Every (game, starter) pair, including unplayed games, so they get a pre-game rating too
    starts = pd.concat([games[["game_id", "gameday", f"{s}_qb_id"]].rename(columns={f"{s}_qb_id": "qb"})
                        for s in ("home", "away")]).dropna(subset=["qb"])
    hist = per_game.rename(columns={"passer_player_id": "qb"})
    long = starts.merge(hist[["game_id", "qb", "n", "epa"]], on=["game_id", "qb"], how="left")
    long = long.sort_values(["qb", "gameday"])
    grp = long.groupby("qb")
    for c in ("n", "epa"):
        long[f"prev_{c}"] = grp[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    long["qb_epa"] = (long["prev_epa"] + k * QB_PRIOR) / (long["prev_n"] + k)
    return long[["game_id", "qb", "qb_epa", "prev_n"]].drop_duplicates(["game_id", "qb"])


def add_qb_features(feats: pd.DataFrame, games: pd.DataFrame, qb_plays: pd.DataFrame) -> pd.DataFrame:
    q = qb_quality(games, qb_plays)
    out = feats.copy()
    for s in ("home", "away"):
        side = games[["game_id", f"{s}_qb_id"]].merge(q, left_on=["game_id", f"{s}_qb_id"],
                                                        right_on=["game_id", "qb"], how="left")
        out = out.merge(side[["game_id", "qb_epa"]].rename(columns={"qb_epa": f"{s}_qb_epa"}), on="game_id", how="left")
    out["sum_qb_epa"] = out["home_qb_epa"] + out["away_qb_epa"]
    out["min_qb_epa"] = out[["home_qb_epa", "away_qb_epa"]].min(axis=1)
    return out.drop(columns=["home_qb_epa", "away_qb_epa"])
