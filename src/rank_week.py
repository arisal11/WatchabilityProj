"""Rank the next unplayed week of NFL games by predicted watchability.

Usage (from the project root):
    venv/bin/python src/rank_week.py            # refresh current season, rank next week
    venv/bin/python src/rank_week.py --week 5   # rank a specific week
    venv/bin/python src/rank_week.py --no-refresh
    venv/bin/python src/rank_week.py --explain  # show model feature contributions too

Also appends the week's games to data/my_ratings.csv for you to rate 1-5 afterwards.
"""

import argparse
from pathlib import Path

import numpy as np
import nfl_data_py as nfl
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_label import build_label
from drama_features import compute_drama
from features import STAKES, STREAK_K, STREAK_WINDOW, build_features, qb_quality

DATA = Path(__file__).resolve().parent.parent / "data"
SEASON = 2026
PBP_COLS = ["game_id", "season", "wpa", "wp", "home_wp", "away_wp", "qtr", "play_type",
            "game_seconds_remaining", "home_score", "away_score",
            "total_home_score", "total_away_score"]
QB_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "passer_player_id",
           "passer_player_name", "rusher_player_id", "qb_dropback", "qb_scramble", "qb_epa", "epa", "cpoe",
           "pass_attempt", "complete_pass", "passing_yards", "pass_touchdown", "interception", "sack",
           "air_yards", "yards_after_catch"]
NON_FEATURES = ("game_id", "season", "roof", "watchability_score", "home_score")


def refresh_season(season: int = SEASON) -> None:
    """Re-pull only the current season; pull its play-by-play only if drama is missing."""
    games = pd.read_csv(DATA / "games_2000_2026.csv", low_memory=False)
    cur = nfl.import_schedules([season])
    games = pd.concat([games[games.season != season], cur], ignore_index=True)
    games.to_csv(DATA / "games_2000_2026.csv", index=False)

    drama = pd.read_csv(DATA / "game_drama_features.csv")
    played = set(games.loc[games.home_score.notna(), "game_id"])
    if played - set(drama.game_id):
        pbp_new = nfl.import_pbp_data([season], columns=PBP_COLS, include_participation=False, downcast=True)
        old = pd.read_parquet(DATA / "pbp_wp_2000_2026.parquet")
        pd.concat([old[old.season != season], pbp_new]).to_parquet(DATA / "pbp_wp_2000_2026.parquet", index=False)
        drama = pd.concat([drama[~drama.game_id.str.startswith(str(season))], compute_drama(pbp_new)])
        drama.to_csv(DATA / "game_drama_features.csv", index=False)

    qb = pd.read_parquet(DATA / "pbp_qb_2000_2026.parquet")
    if played - set(qb.game_id):
        new = nfl.import_pbp_data([season], columns=QB_COLS, include_participation=False, downcast=True)
        new = new[new.qb_dropback == 1]
        pd.concat([qb[qb.season != season], new]).to_parquet(DATA / "pbp_qb_2000_2026.parquet", index=False)


def rebuild_tables() -> pd.DataFrame:
    """Schedule + drama -> master_games.csv -> labeled_games.csv. Returns master with the label."""
    games = pd.read_csv(DATA / "games_2000_2026.csv", low_memory=False)
    drama = pd.read_csv(DATA / "game_drama_features.csv")
    master = games.merge(drama, on="game_id", how="left")
    master["playoff_stakes"] = master["game_type"].map(STAKES)
    master["rivalry"] = master["div_game"]
    master["fireworks"] = master["total"]
    master.to_csv(DATA / "master_games.csv", index=False)

    labeled = build_label(master)
    labeled["rank"] = labeled["watchability_score"].rank(ascending=False, method="min").astype(int)
    labeled.to_csv(DATA / "labeled_games.csv", index=False)
    return master.merge(labeled[["game_id", "watchability_score"]], on="game_id", how="left")


def make_model(num_cols: list) -> Pipeline:
    """Ridge on the base features; chosen in notebook 02 (ties LightGBM, simpler, exactly explainable)."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["roof"])])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=10.0))])


def explain(model: Pipeline, rows: pd.DataFrame, top: int = 3) -> pd.Series:
    """Exact per-feature contributions: a linear prediction is intercept + sum(coef * scaled feature)."""
    Xt = model.named_steps["pre"].transform(rows)
    names = [n.split("__", 1)[1] for n in model.named_steps["pre"].get_feature_names_out()]
    contrib = pd.DataFrame(Xt * model.named_steps["ridge"].coef_, columns=names, index=rows.index)
    return contrib.apply(lambda r: ", ".join(
        f"{k} {v:+.1f}" for k, v in r[r.abs().sort_values(ascending=False).index[:top]].items()), axis=1)


STREAK_FLAG = 0.12  # EPA/dropback gap worth mentioning (~top/bottom 10% of starters)


def reasons(g: pd.Series) -> str:
    """Plain-language reasons from pre-game facts. These are for the reader, not the model:
    QB quality and win% are real reasons people tune in, even where Vegas already prices them in."""
    r = []
    if g.playoff_stakes:
        r.append(["", "Playoffs", "Divisional round", "Conference title", "Super Bowl"][int(g.playoff_stakes)])
    if g.div_game:
        r.append("division rivals")
    if g.abs_spread <= 2.5:
        r.append(f"Vegas calls it a coin flip ({g.abs_spread:g}-pt spread)")
    if g.total_line >= 48:
        r.append(f"shootout expected (O/U {g.total_line:g})")
    elite = [n for n, rk in [(g.away_qb_name, g.away_qb_rank), (g.home_qb_name, g.home_qb_rank)] if rk <= 8]
    if len(elite) == 2:
        r.append(f"QB duel: {elite[0]} vs {elite[1]}")
    elif elite:
        r.append(f"top QB {elite[0]}")
    if g.week >= 4 and g.min_win_pct >= 0.6:   # too noisy before week 4
        r.append("two winning teams")
    # Hot/cold streaks: last 3 games vs his 16-game norm. Only ~9% of a streak carries into the
    # next game historically (regression to the mean), so this is a talking point, not a model input.
    for name, st in [(g.away_qb_name, g.away_qb_streak), (g.home_qb_name, g.home_qb_streak)]:
        if st <= -STREAK_FLAG:
            r.append(f"{name} in a slump (last 3 games well below his norm)")
        elif st >= STREAK_FLAG:
            r.append(f"{name} on a heater")
    if g.abs_spread >= 7:
        r.append(f"but likely lopsided ({g.abs_spread:g}-pt favorite)")
    return "; ".join(r) if r else "nothing stands out"


def update_ratings_file(board: pd.DataFrame, season: int, week: int) -> Path:
    """Append this week's games to data/my_ratings.csv (blank ratings) without touching existing rows."""
    path = DATA / "my_ratings.csv"
    rows = pd.DataFrame({"game_id": board.game_id, "season": season, "week": week,
                         "matchup": board.away_team + " @ " + board.home_team,
                         "predicted": board.pred.round(1),
                         "qb_streak": (board.get("home_qb_streak", np.nan) + board.get("away_qb_streak", np.nan)),
                         "my_rating": pd.NA, "notes": ""})
    rows["qb_streak"] = rows.qb_streak.round(3)
    if path.exists():
        old = pd.read_csv(path, dtype={"notes": "string"})
        rows = pd.concat([old, rows[~rows.game_id.isin(old.game_id)]], ignore_index=True)
    rows.to_csv(path, index=False)
    return path


def rank_week(full: pd.DataFrame, season: int = SEASON, week: int = None) -> pd.DataFrame:
    X = build_features(full).merge(full[["game_id", "watchability_score", "home_score"]], on="game_id")
    num_cols = [c for c in X.columns if c not in NON_FEATURES]
    train = X[X.watchability_score.notna()]
    model = make_model(num_cols).fit(train, train.watchability_score)

    upcoming = X[(X.season == season) & X.home_score.isna()]
    week = week or int(upcoming.week.min())
    wk = X[(X.season == season) & (X.week == week)].copy()
    wk["pred"] = model.predict(wk)
    wk["why"] = explain(model, wk)

    info = full.set_index("game_id")[["weekday", "gametime", "away_team", "home_team", "spread_line", "total_line",
                                      "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name"]]
    board = wk.set_index("game_id")[["pred", "why", "week", "playoff_stakes", "div_game", "abs_spread"]].join(info)

    # Rank this week's starters by pre-game QB EPA/dropback (1 = best)
    qb_path = DATA / "pbp_qb_2000_2026.parquet"
    if qb_path.exists():
        plays = pd.read_parquet(qb_path, columns=["game_id", "passer_player_id", "qb_epa"])
        q, q3 = (x[x.game_id.isin(board.index)].set_index(["game_id", "qb"])["qb_epa"]
                 for x in (qb_quality(full, plays), qb_quality(full, plays, STREAK_WINDOW, STREAK_K)))
        for s in ("home", "away"):
            keys = list(zip(board.index, board[f"{s}_qb_id"]))
            board[f"{s}_qb_epa"] = [q.get(k, np.nan) for k in keys]
            board[f"{s}_qb_streak"] = [q3.get(k, np.nan) - q.get(k, np.nan) for k in keys]
        ranks = pd.concat([board.home_qb_epa, board.away_qb_epa]).rank(ascending=False)
        board["home_qb_rank"], board["away_qb_rank"] = ranks.iloc[:len(board)].values, ranks.iloc[len(board):].values
    else:
        board["home_qb_rank"] = board["away_qb_rank"] = np.inf
        board["home_qb_streak"] = board["away_qb_streak"] = np.nan
    board["min_win_pct"] = wk.set_index("game_id")["min_win_pct"]
    board["reasons"] = board.apply(reasons, axis=1)

    board = board.sort_values("pred", ascending=False)
    board.insert(0, "rank", range(1, len(board) + 1))
    board.attrs["week"] = week
    return board.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int)
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--explain", action="store_true", help="also show model feature contributions")
    args = ap.parse_args()

    if not args.no_refresh:
        refresh_season(args.season)
    board = rank_week(rebuild_tables(), args.season, args.week)

    print(f"\n{args.season} Week {board.attrs['week']} — most watchable first\n")
    for r in board.itertuples():
        game = f"{r.away_team} @ {r.home_team}"
        print(f"{r.rank:>2}. {game:<11} {r.weekday[:3]} {r.gametime}  score {r.pred:4.1f}  {r.reasons}")
        if args.explain:
            print(f"{'':16}model: {r.why}")
    path = update_ratings_file(board, args.season, board.attrs["week"])
    print(f"\nRate these 1-5 after the games in {path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
