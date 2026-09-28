import pandas as pd

import rank_week


def test_ratings_file_never_overwrites_existing_ratings(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_week, "DATA", tmp_path)
    board = pd.DataFrame({"game_id": ["g1", "g2"], "away_team": ["A", "C"], "home_team": ["B", "D"], "pred": [40.0, 30.0]})
    rank_week.update_ratings_file(board, 2026, 3)

    rated = pd.read_csv(tmp_path / "my_ratings.csv", dtype={"notes": "string"})
    rated.loc[0, ["my_rating", "notes"]] = [5, "instant classic"]
    rated.to_csv(tmp_path / "my_ratings.csv", index=False)

    board.loc[0, "pred"] = 99.0   # re-running the same week must not reset anything
    rank_week.update_ratings_file(board, 2026, 3)
    out = pd.read_csv(tmp_path / "my_ratings.csv")
    assert len(out) == 2
    assert out.loc[0, "my_rating"] == 5 and out.loc[0, "notes"] == "instant classic"
    assert out.loc[0, "predicted"] == 40.0
