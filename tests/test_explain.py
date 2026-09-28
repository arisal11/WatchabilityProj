import numpy as np

from features import build_features
from rank_week import NON_FEATURES, explain, make_model


def test_contributions_sum_exactly_to_prediction(games):
    """A linear model's prediction = intercept + sum(coef * scaled feature). No approximation."""
    X = build_features(games).merge(games[["game_id", "watchability_score", "home_score"]], on="game_id")
    num = [c for c in X.columns if c not in NON_FEATURES]
    train = X[X.watchability_score.notna()]
    model = make_model(num).fit(train, train.watchability_score)

    Xt = model.named_steps["pre"].transform(X)
    manual = Xt @ model.named_steps["ridge"].coef_ + model.named_steps["ridge"].intercept_
    assert np.allclose(manual, model.predict(X))
    assert explain(model, X).str.count(",").eq(2).all()   # top 3 drivers per game
