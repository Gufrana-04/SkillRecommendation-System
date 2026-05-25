import pytest
import pandas as pd
import numpy as np
from data_loader import load_data, build_feature_matrix, get_completed_levels, get_level_pass_rates
from recommendation_engine import RecommendationEngine


@pytest.fixture(scope="module")
def sample_df():
    data = {
        "student_id": ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", "S4", "S4"],
        "level_id":   ["L1", "L2", "L3", "L1", "L2", "L4", "L1", "L3", "L4", "L2", "L3"],
        "score":      [80, 75, 90, 70, 65, 85, 60, 55, 80, 90, 88],
        "time_spent_minutes": [30, 25, 35, 40, 30, 20, 50, 45, 30, 25, 20],
        "passed":     [1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def fitted_engine(sample_df):
    engine = RecommendationEngine(top_k=3, top_n=3, success_threshold=0.5)
    engine.fit(sample_df)
    return engine


@pytest.fixture(scope="module")
def full_df():
    return load_data("../datasets/student_progress.csv")


@pytest.fixture(scope="module")
def full_engine(full_df):
    engine = RecommendationEngine()
    engine.fit(full_df)
    return engine


class TestDataLoader:

    def test_load_returns_dataframe(self):
        df = load_data("../datasets/student_progress.csv")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = load_data("../datasets/student_progress.csv")
        for col in ["student_id", "level_id", "score", "time_spent_minutes", "passed"]:
            assert col in df.columns

    def test_no_nulls_in_key_columns(self):
        df = load_data("../datasets/student_progress.csv")
        for col in ["student_id", "level_id", "score"]:
            assert df[col].isnull().sum() == 0

    def test_passed_column_is_integer(self):
        df = load_data("../datasets/student_progress.csv")
        assert df["passed"].dtype in [int, np.int64, np.int32]

    def test_load_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("student_id,level_id\nS1,L1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_data(str(bad_csv))

    def test_feature_matrix_shape(self, sample_df):
        fm = build_feature_matrix(sample_df)
        assert fm.shape == (4, 4)

    def test_feature_matrix_no_negatives(self, sample_df):
        fm = build_feature_matrix(sample_df)
        assert (fm.values >= 0).all()

    def test_feature_matrix_zero_for_unattempted(self, sample_df):
        fm = build_feature_matrix(sample_df)
        assert fm.loc["S1", "L4"] == 0

    def test_completed_levels_correct(self, sample_df):
        assert get_completed_levels(sample_df, "S1") == ["L1", "L2", "L3"]

    def test_completed_levels_sorted(self, sample_df):
        levels = get_completed_levels(sample_df, "S2")
        assert levels == sorted(levels)

    def test_pass_rates_between_0_and_1(self, sample_df):
        rates = get_level_pass_rates(sample_df)
        assert (rates >= 0).all() and (rates <= 1).all()


class TestRecommendationEngine:

    def test_fit_stores_feature_matrix(self, fitted_engine):
        assert fitted_engine.feature_matrix is not None

    def test_similarity_matrix_shape(self, fitted_engine):
        n = len(fitted_engine.feature_matrix)
        assert fitted_engine._similarity_matrix.shape == (n, n)

    def test_similarity_diagonal_is_one(self, fitted_engine):
        diag = np.diag(fitted_engine._similarity_matrix)
        assert np.allclose(diag, 1.0, atol=1e-6)

    def test_recommend_returns_list(self, fitted_engine):
        assert isinstance(fitted_engine.recommend("S1"), list)

    def test_recommend_at_most_top_n(self, fitted_engine):
        assert len(fitted_engine.recommend("S1")) <= fitted_engine.top_n

    def test_recommend_keys_present(self, fitted_engine):
        for rec in fitted_engine.recommend("S1"):
            assert "level_id" in rec and "score" in rec and "reason" in rec

    def test_recommend_no_completed_levels(self, fitted_engine, sample_df):
        completed = set(get_completed_levels(sample_df, "S1"))
        for rec in fitted_engine.recommend("S1"):
            assert rec["level_id"] not in completed

    def test_recommend_scores_above_threshold(self, fitted_engine):
        for rec in fitted_engine.recommend("S1"):
            assert rec["score"] >= fitted_engine.success_threshold

    def test_recommend_sorted_descending(self, fitted_engine):
        scores = [r["score"] for r in fitted_engine.recommend("S1")]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_raises_for_unknown_student(self, fitted_engine):
        with pytest.raises(ValueError, match="not found"):
            fitted_engine.recommend("UNKNOWN")

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            RecommendationEngine().recommend("S1")

    def test_recommend_empty_when_all_completed(self, sample_df):
        all_levels = sample_df["level_id"].unique().tolist()
        extra = pd.DataFrame({
            "student_id": ["S99"] * len(all_levels),
            "level_id": all_levels,
            "score": [80] * len(all_levels),
            "time_spent_minutes": [30] * len(all_levels),
            "passed": [1] * len(all_levels),
        })
        df2 = pd.concat([sample_df, extra], ignore_index=True)
        engine = RecommendationEngine(top_k=3, top_n=3, success_threshold=0.5)
        engine.fit(df2)
        assert engine.recommend("S99") == []

    def test_70pct_threshold_on_full_data(self, full_engine):
        rng = np.random.default_rng(42)
        ids = rng.choice(full_engine.feature_matrix.index.tolist(), size=50, replace=False)
        for sid in ids:
            for rec in full_engine.recommend(str(sid)):
                assert rec["score"] >= 0.70

    def test_evaluation_keys(self, full_engine):
        metrics = full_engine.evaluate(list(full_engine.feature_matrix.index[:20]))
        for key in ["precision", "coverage", "avg_recommendations", "students_evaluated"]:
            assert key in metrics

    def test_evaluation_precision_range(self, full_engine):
        metrics = full_engine.evaluate(list(full_engine.feature_matrix.index[:20]))
        assert 0.0 <= metrics["precision"] <= 1.0


class TestIntegration:

    def test_full_pipeline_runs(self):
        df = load_data("../datasets/student_progress.csv")
        engine = RecommendationEngine()
        engine.fit(df)
        recs = engine.recommend(str(df["student_id"].iloc[0]))
        assert isinstance(recs, list)

    def test_recommendations_are_valid_levels(self, full_engine, full_df):
        valid = set(full_df["level_id"].unique())
        for sid in full_engine.feature_matrix.index[:10]:
            for rec in full_engine.recommend(sid):
                assert rec["level_id"] in valid
