"""Render the README charts into docs/. Run from the project root:
    venv/bin/python src/make_charts.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features as F
from rank_week import DATA, rank_week, rebuild_tables

DOCS = DATA.parent / "docs"
BLUE, GRAY, INK, MUTED, SURFACE = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
})


def week_chart(full: pd.DataFrame, week: int) -> pd.DataFrame:
    board = rank_week(full, week=week)
    b = board.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(b.away_team + " @ " + b.home_team, b.pred, color=BLUE, height=0.7)
    for y, v in enumerate(b.pred):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, b.pred.max() * 1.12)
    ax.set_xlabel("Predicted watchability score")
    ax.set_title(f"2026 Week {week}: which games to watch")
    ax.grid(axis="x", color="#ecebe7")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DOCS / f"week{week}.png", dpi=150)
    return board


def predictability_chart() -> None:
    # Walk-forward 2016-2025 R² from notebook 02: how much each part of the label is knowable pre-game
    parts = pd.Series({"Late-game excitement": 0.016, "Comeback factor": 0.003,
                       "Fireworks (points)": 0.077, "Full watchability score": 0.080})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(parts.index[::-1], parts.values[::-1], color=[BLUE] + [GRAY] * 3, height=0.6)
    for y, v in enumerate(parts.values[::-1]):
        ax.text(v + 0.002, y, f"R² {v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 0.1)
    ax.set_xlabel("Walk-forward R² from pre-game features (2016-2025)")
    ax.set_title("How much is knowable before kickoff?")
    fig.tight_layout()
    fig.savefig(DOCS / "predictability.png", dpi=150)


def streak_chart(full: pd.DataFrame) -> None:
    plays = pd.read_parquet(DATA / "pbp_qb_2000_2026.parquet", columns=["game_id", "passer_player_id", "qb_epa"])
    q = F.qb_quality(full, plays).rename(columns={"qb_epa": "base"}).merge(
        F.qb_quality(full, plays, F.STREAK_WINDOW, F.STREAK_K).rename(columns={"qb_epa": "last3"})[["game_id", "qb", "last3"]],
        on=["game_id", "qb"])
    nxt = (plays.dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"]).qb_epa
           .agg(["size", "mean"]).reset_index().rename(columns={"passer_player_id": "qb"}))
    t = q.merge(nxt, on=["game_id", "qb"])
    t = t[(t["size"] >= 15) & t.game_id.str[:4].astype(int).between(2006, 2025)]
    t["streak"], t["next_vs_base"] = t.last3 - t.base, t["mean"] - t.base
    t["bin"] = pd.qcut(t.streak, 10)
    g = t.groupby("bin", observed=True)[["streak", "next_vs_base"]].mean()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    lim = np.abs(g.streak).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=2, ls="--")
    ax.plot(g.streak, g.next_vs_base, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    ax.text(lim * 0.55, lim * 0.72, "if streaks fully\ncarried over", color=MUTED, fontsize=10)
    ax.text(g.streak.iloc[-2], g.next_vs_base.iloc[-2] + 0.02, "what actually happens", color=INK, fontsize=10)
    ax.axhline(0, color="#d8d7d2", lw=1)
    ax.set_xlabel("QB's last-3-game form vs his 16-game norm (EPA/dropback)")
    ax.set_ylabel("Next game vs his norm")
    ax.set_title(f"Hot and cold QB streaks mostly regress to the mean ({len(t):,} starts)")
    fig.tight_layout()
    fig.savefig(DOCS / "qb_streaks.png", dpi=150)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    full = rebuild_tables()
    week_chart(full, 4)
    predictability_chart()
    streak_chart(full)
    print("wrote", sorted(p.name for p in DOCS.iterdir()))
