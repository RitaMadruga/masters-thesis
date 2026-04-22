import json
from pathlib import Path

import mofax as mfx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


METRICS = ["balanced_accuracy", "precision", "recall", "f1", "pr_auc"]
CLASSIFIER_LABELS = {
    "random_forest": "Random Forest",
    "linear_svm": "Linear SVM",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_mofa_models(root_dir: Path) -> dict[str, Path]:
    latent_dir = root_dir / "data" / "latent"
    return {
        "mofa_trained_lg2": latent_dir / "mofa_trained_lg2.hdf5",
        "mofa_trained_vsn": latent_dir / "mofa_trained_vsn.hdf5",
        "mofa_trained_lg2_fs": latent_dir / "mofa_trained_lg2_fs.hdf5",
        "mofa_trained_vsn_fs": latent_dir / "mofa_trained_vsn_fs.hdf5",
    }


def load_clinical_labels(root_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    clinical_path = root_dir / "data" / "cleaned_data" / "clinical_cleaned.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    y = clinical_data["tumor_subtype"].map({"other_subtype": 1, "ductal_type": 0})

    if y.isna().any():
        missing_labels = clinical_data.loc[y.isna(), "tumor_subtype"].unique().tolist()
        raise ValueError(f"Unexpected tumor_subtype values found: {missing_labels}")

    return clinical_data, y.astype(int)


def load_mofa_factors(model_path: Path, clinical_index: pd.Index) -> pd.DataFrame:
    model = mfx.mofa_model(str(model_path))
    factors = model.get_factors(df=True)
    missing_samples = clinical_index.difference(factors.index)
    if not missing_samples.empty:
        raise ValueError(
            f"MOFA factors are missing {len(missing_samples)} clinical samples for {model_path.name}"
        )
    return factors.loc[clinical_index]


def build_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=1)


def build_linear_svm(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    dual="auto",
                    random_state=seed,
                    max_iter=10000,
                ),
            ),
        ]
    )


def random_forest_param_grid() -> dict[str, list]:
    return {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [5, 10, 15, 20],
        "class_weight": ["balanced", "balanced_subsample"],
    }


def linear_svm_param_grid() -> dict[str, list]:
    return {"svm__C": [0.01, 0.1, 1, 10, 100]}


def rf_scoring() -> dict[str, object]:
    return {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc_score": make_scorer(average_precision_score, response_method="predict_proba"),
    }


def svm_scoring() -> dict[str, object]:
    return {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc_score": make_scorer(average_precision_score, response_method="decision_function"),
    }


def classifier_configs() -> dict[str, dict[str, object]]:
    return {
        "random_forest": {
            "builder": build_random_forest,
            "param_grid": random_forest_param_grid(),
            "scoring": rf_scoring(),
            "score_method": "predict_proba",
        },
        "linear_svm": {
            "builder": build_linear_svm,
            "param_grid": linear_svm_param_grid(),
            "scoring": svm_scoring(),
            "score_method": "decision_function",
        },
    }


def compute_metrics(y_true: pd.Series, y_pred, y_score) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def get_score_values(model, X, score_method: str):
    if score_method == "predict_proba":
        return model.predict_proba(X)[:, 1]
    if score_method == "decision_function":
        return model.decision_function(X)
    raise ValueError(f"Unsupported score method: {score_method}")


def metric_to_cv_name(metric: str) -> str:
    return "pr_auc_score" if metric == "pr_auc" else metric


def summarize_results(results_df: pd.DataFrame, scopes: list[str]) -> pd.DataFrame:
    rows = []
    for scope in scopes:
        for metric in METRICS:
            column = f"{scope}_{metric}"
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "mean": float(results_df[column].mean()),
                    "std": float(results_df[column].std(ddof=1)) if len(results_df) > 1 else 0.0,
                    "min": float(results_df[column].min()),
                    "max": float(results_df[column].max()),
                    "median": float(results_df[column].median()),
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_strategy_comparison(
    strategy_name: str,
    strategy_dir: Path,
    scope_name: str,
) -> pd.DataFrame:
    rows = []
    for summary_path in strategy_dir.glob("*/*/summary_across_seeds.csv"):
        classifier_name = summary_path.parents[1].name
        model_name = summary_path.parents[0].name
        summary_df = pd.read_csv(summary_path)
        scoped_df = summary_df.loc[summary_df["scope"] == scope_name].copy()
        scoped_df.insert(0, "model_name", model_name)
        scoped_df.insert(0, "classifier", classifier_name)
        scoped_df.insert(0, "evaluation_strategy", strategy_name)
        rows.append(scoped_df)

    comparison_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not comparison_df.empty:
        write_csv(comparison_df, strategy_dir / "comparison_table.csv")
    return comparison_df


def write_final_comparison(base_output_dir: Path) -> pd.DataFrame:
    frames = []
    for strategy_name in ["train_test_split", "nested_cv"]:
        comparison_path = base_output_dir / strategy_name / "comparison_table.csv"
        if comparison_path.exists():
            frames.append(pd.read_csv(comparison_path))

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not final_df.empty:
        write_csv(final_df, base_output_dir / "final_comparison_table.csv")
    return final_df


def serialize_params(params) -> str:
    return json.dumps(params, sort_keys=True)
