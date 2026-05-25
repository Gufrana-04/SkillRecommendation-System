import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    required_columns = {"student_id", "level_id", "score", "time_spent_minutes", "passed"}
    df = pd.read_csv(filepath)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["student_id"] = df["student_id"].astype(str)
    df["level_id"] = df["level_id"].astype(str)
    df["score"] = df["score"].astype(float)
    df["time_spent_minutes"] = df["time_spent_minutes"].astype(float)
    df["passed"] = df["passed"].astype(bool).astype(int)
    df = df.dropna(subset=["student_id", "level_id", "score"]).reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["student_id", "level_id"])["score"]
        .mean()
        .unstack(fill_value=0)
    )
    all_levels = sorted(df["level_id"].unique())
    pivot = pivot.reindex(columns=all_levels, fill_value=0)
    return pivot


def get_completed_levels(df: pd.DataFrame, student_id: str) -> list:
    student_rows = df[df["student_id"] == student_id]
    return sorted(student_rows["level_id"].unique().tolist())


def get_level_pass_rates(df: pd.DataFrame) -> pd.Series:
    return df.groupby("level_id")["passed"].mean()
