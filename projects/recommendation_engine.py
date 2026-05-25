import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data, build_feature_matrix, get_completed_levels, get_level_pass_rates

TOP_K_NEIGHBOURS = 50
TOP_N_RECOMMENDATIONS = 3
SUCCESS_THRESHOLD = 0.70


class RecommendationEngine:

    def __init__(self, top_k=TOP_K_NEIGHBOURS, top_n=TOP_N_RECOMMENDATIONS, success_threshold=SUCCESS_THRESHOLD):
        self.top_k = top_k
        self.top_n = top_n
        self.success_threshold = success_threshold
        self.df = None
        self.feature_matrix = None
        self.pass_rates = None
        self._similarity_matrix = None

    def fit(self, df: pd.DataFrame):
        self.df = df
        self.feature_matrix = build_feature_matrix(df)
        self.pass_rates = get_level_pass_rates(df)
        matrix = self.feature_matrix.values.astype(float)
        self._similarity_matrix = cosine_similarity(matrix)
        return self

    def recommend(self, student_id: str) -> list:
        if self.feature_matrix is None:
            raise RuntimeError("Call fit() before recommend().")
        if student_id not in self.feature_matrix.index:
            raise ValueError(f"Student {student_id} not found in training data.")

        completed = set(get_completed_levels(self.df, student_id))
        all_levels = set(self.feature_matrix.columns.tolist())
        candidate_levels = all_levels - completed

        if not candidate_levels:
            return []

        student_idx = self.feature_matrix.index.get_loc(student_id)
        sim_scores = self._similarity_matrix[student_idx]
        sim_series = pd.Series(sim_scores, index=self.feature_matrix.index)
        sim_series = sim_series.drop(index=student_id)
        neighbours = sim_series.nlargest(self.top_k)

        neighbour_ids = neighbours.index.tolist()
        neighbour_df = self.df[self.df["student_id"].isin(neighbour_ids)].copy()
        neighbour_df = neighbour_df.merge(
            neighbours.rename("similarity"),
            left_on="student_id",
            right_index=True,
        )
        neighbour_df = neighbour_df[neighbour_df["level_id"].isin(candidate_levels)]

        if neighbour_df.empty:
            return []

        def weighted_pass_rate(group):
            total_weight = group["similarity"].sum()
            if total_weight == 0:
                return 0.0
            return (group["similarity"] * group["passed"]).sum() / total_weight

        level_scores = (
            neighbour_df.groupby("level_id")
            .apply(weighted_pass_rate, include_groups=False)
            .rename("weighted_success")
        )

        filtered = level_scores[level_scores >= self.success_threshold]
        top_levels = filtered.nlargest(self.top_n)

        recommendations = []
        for level_id, score in top_levels.items():
            neighbour_count = neighbour_df[neighbour_df["level_id"] == level_id]["student_id"].nunique()
            recommendations.append({
                "level_id": str(level_id),
                "score": round(float(score), 4),
                "reason": (
                    f"{neighbour_count} similar students attempted '{level_id}' "
                    f"with a {score*100:.1f}% weighted success rate."
                ),
            })
        return recommendations

    def evaluate(self, test_student_ids=None) -> dict:
        if self.df is None:
            raise RuntimeError("Call fit() first.")

        all_ids = self.feature_matrix.index.tolist()
        if test_student_ids is None:
            rng = np.random.default_rng(0)
            test_student_ids = rng.choice(all_ids, size=max(1, len(all_ids) // 10), replace=False).tolist()

        hits, total_recs, students_with_recs = 0, 0, 0
        for sid in test_student_ids:
            recs = self.recommend(sid)
            if not recs:
                continue
            students_with_recs += 1
            passed_levels = set(self.df[(self.df["student_id"] == sid) & (self.df["passed"] == 1)]["level_id"])
            for r in recs:
                total_recs += 1
                if r["level_id"] in passed_levels:
                    hits += 1

        return {
            "precision": round(hits / total_recs if total_recs > 0 else 0.0, 4),
            "coverage": round(students_with_recs / len(test_student_ids) if test_student_ids else 0.0, 4),
            "avg_recommendations": round(total_recs / max(students_with_recs, 1), 2),
            "students_evaluated": len(test_student_ids),
        }
