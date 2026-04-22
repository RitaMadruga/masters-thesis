import argparse
from pathlib import Path

import mofax as mfx
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from subtype_classification_common import (
    classifier_configs,
    default_mofa_models,
    load_clinical_labels,
    load_mofa_factors,
    project_root,
    serialize_params,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run biomarker discovery for linear SVM + train/test split across MOFA models."
    )
    parser.add_argument("--n-seeds", type=int, default=30, help="Number of seeds to run.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test fraction.")
    parser.add_argument("--inner-splits", type=int, default=5, help="Inner CV folds.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional MOFA model names to run. Defaults to all models.",
    )
    parser.add_argument(
        "--top-factors",
        type=int,
        default=5,
        help="Maximum number of robust factors to keep per model.",
    )
    parser.add_argument(
        "--min-stability",
        type=float,
        default=0.70,
        help="Minimum sign stability required for a factor to be kept.",
    )
    parser.add_argument(
        "--top-features-per-sign",
        type=int,
        default=15,
        help="Top positive and top negative MOFA loadings to keep per factor and view.",
    )
    return parser.parse_args()


def source_view_paths(root_dir: Path, model_name: str) -> dict[str, Path]:
    feature_selection_dir = root_dir / "data" / "feature_selection"
    transformed_dir = root_dir / "data" / "transformed_data"

    if model_name == "mofa_trained_lg2_fs":
        return {
            "mRNA": feature_selection_dir / "selected_features_mrna_data_lg2.csv",
            "miRNA": feature_selection_dir / "selected_features_mirna_data_lg2.csv",
            "Methylation": feature_selection_dir / "selected_features_meth_data.csv",
        }
    if model_name == "mofa_trained_vsn_fs":
        return {
            "mRNA": feature_selection_dir / "selected_features_mrna_data_vsn.csv",
            "miRNA": feature_selection_dir / "selected_features_mirna_data_vsn.csv",
            "Methylation": feature_selection_dir / "selected_features_meth_data.csv",
        }
    if model_name == "mofa_trained_lg2":
        return {
            "mRNA": transformed_dir / "mrna_data_lg2.csv",
            "miRNA": transformed_dir / "mirna_data_lg2.csv",
            "Methylation": transformed_dir / "meth_data_m_values.csv",
        }
    if model_name == "mofa_trained_vsn":
        return {
            "mRNA": transformed_dir / "mrna_data_vsn.csv",
            "miRNA": transformed_dir / "mirna_data_vsn.csv",
            "Methylation": transformed_dir / "meth_data_m_values.csv",
        }
    raise ValueError(f"Unsupported model name: {model_name}")


def feature_to_view_map(root_dir: Path, model_name: str) -> dict[str, str]:
    mapping = {}
    for view_name, path in source_view_paths(root_dir, model_name).items():
        view_df = pd.read_csv(path, index_col=0)
        for feature_name in view_df.columns:
            mapping[feature_name] = view_name
    return mapping


def run_linear_svm_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    seeds: list[int],
    test_size: float,
    inner_splits: int,
    n_jobs: int,
) -> pd.DataFrame:
    config = classifier_configs()["linear_svm"]
    coefficient_rows = []

    for seed in seeds:
        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        _ = X_test

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        model = config["builder"](seed)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=config["param_grid"],
            scoring=config["scoring"],
            refit="pr_auc_score",
            cv=inner_cv,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        coefficients = best_model.named_steps["svm"].coef_[0]

        for factor, coefficient in zip(X.columns, coefficients):
            coefficient_rows.append(
                {
                    "seed": seed,
                    "factor": factor,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                    "direction": (
                        "other_subtype"
                        if coefficient > 0
                        else "ductal_type"
                        if coefficient < 0
                        else "neutral"
                    ),
                    "best_params": serialize_params(grid_search.best_params_),
                }
            )

    return pd.DataFrame(coefficient_rows)


def summarize_factors(coefficients_df: pd.DataFrame) -> pd.DataFrame:
    factor_summary = (
        coefficients_df.groupby("factor", as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            median_abs_coefficient=("abs_coefficient", "median"),
            std_abs_coefficient=("abs_coefficient", "std"),
            times_positive=("coefficient", lambda s: int((s > 0).sum())),
            times_negative=("coefficient", lambda s: int((s < 0).sum())),
            times_zero=("coefficient", lambda s: int((s == 0).sum())),
            n_models=("coefficient", "size"),
        )
        .sort_values(["mean_abs_coefficient", "median_abs_coefficient"], ascending=False)
        .reset_index(drop=True)
    )

    factor_summary["stability"] = (
        factor_summary[["times_positive", "times_negative"]].max(axis=1)
        / factor_summary["n_models"]
    )
    factor_summary["dominant_direction"] = np.where(
        factor_summary["times_positive"] > factor_summary["times_negative"],
        "other_subtype",
        np.where(
            factor_summary["times_negative"] > factor_summary["times_positive"],
            "ductal_type",
            "neutral",
        ),
    )
    return factor_summary


def load_weights_long(model_path: Path, root_dir: Path, model_name: str) -> pd.DataFrame:
    model = mfx.mofa_model(str(model_path))
    weights_df = model.get_weights(df=True)
    weights_long = (
        weights_df.reset_index()
        .melt(id_vars=["index"], var_name="factor", value_name="weight")
        .rename(columns={"index": "feature"})
    )
    weights_long["abs_weight"] = weights_long["weight"].abs()

    feature_view_mapping = feature_to_view_map(root_dir, model_name)
    weights_long["view"] = weights_long["feature"].map(feature_view_mapping).fillna("unknown")
    return weights_long


def build_candidate_biomarkers(
    weights_long: pd.DataFrame,
    robust_factors: pd.DataFrame,
    top_features_per_sign: int,
) -> pd.DataFrame:
    candidate_rows = []

    for _, factor_row in robust_factors.iterrows():
        factor = factor_row["factor"]
        factor_direction = factor_row["dominant_direction"]
        factor_weights = weights_long.loc[weights_long["factor"] == factor].copy()

        for view_name, view_df in factor_weights.groupby("view"):
            top_positive = view_df.sort_values("weight", ascending=False).head(top_features_per_sign)
            top_negative = view_df.sort_values("weight", ascending=True).head(top_features_per_sign)
            selected_df = pd.concat([top_positive, top_negative], ignore_index=True).drop_duplicates()

            for _, feature_row in selected_df.iterrows():
                if factor_direction == "other_subtype":
                    feature_direction = (
                        "other_subtype" if feature_row["weight"] > 0 else "ductal_type"
                    )
                elif factor_direction == "ductal_type":
                    feature_direction = (
                        "ductal_type" if feature_row["weight"] > 0 else "other_subtype"
                    )
                else:
                    feature_direction = "neutral"

                candidate_rows.append(
                    {
                        "factor": factor,
                        "factor_direction": factor_direction,
                        "factor_mean_abs_coefficient": float(factor_row["mean_abs_coefficient"]),
                        "factor_stability": float(factor_row["stability"]),
                        "view": view_name,
                        "feature": feature_row["feature"],
                        "weight": float(feature_row["weight"]),
                        "abs_weight": float(feature_row["abs_weight"]),
                        "feature_direction": feature_direction,
                    }
                )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        return candidate_df

    candidate_df = candidate_df.sort_values(
        ["factor_mean_abs_coefficient", "factor", "view", "abs_weight"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    candidate_df["rank_within_factor_view"] = (
        candidate_df.groupby(["factor", "view"])["abs_weight"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return candidate_df


def run_for_model(
    root_dir: Path,
    model_name: str,
    model_path: Path,
    clinical_index: pd.Index,
    y: pd.Series,
    seeds: list[int],
    test_size: float,
    inner_splits: int,
    n_jobs: int,
    top_factors: int,
    min_stability: float,
    top_features_per_sign: int,
    base_output_dir: Path,
) -> dict[str, Path]:
    X = load_mofa_factors(model_path, clinical_index)
    coefficients_df = run_linear_svm_train_test_split(
        X=X,
        y=y,
        seeds=seeds,
        test_size=test_size,
        inner_splits=inner_splits,
        n_jobs=n_jobs,
    )
    factor_summary = summarize_factors(coefficients_df)
    robust_factors = (
        factor_summary.loc[factor_summary["stability"] >= min_stability]
        .head(top_factors)
        .reset_index(drop=True)
    )

    weights_long = load_weights_long(model_path=model_path, root_dir=root_dir, model_name=model_name)
    candidate_biomarkers = build_candidate_biomarkers(
        weights_long=weights_long,
        robust_factors=robust_factors,
        top_features_per_sign=top_features_per_sign,
    )

    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_summary_path = output_dir / "factor_summary_global.csv"
    robust_factors_path = output_dir / "robust_factors.csv"
    candidate_biomarkers_path = output_dir / "candidate_biomarkers_final.csv"
    metadata_path = output_dir / "run_metadata.json"

    write_csv(factor_summary, factor_summary_path)
    write_csv(robust_factors, robust_factors_path)
    write_csv(candidate_biomarkers, candidate_biomarkers_path)
    write_json(
        {
            "evaluation_strategy": "train_test_split",
            "classifier": "linear_svm",
            "model_name": model_name,
            "model_path": str(model_path),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "test_size": test_size,
            "inner_splits": inner_splits,
            "top_factors": top_factors,
            "min_stability": min_stability,
            "top_features_per_sign": top_features_per_sign,
        },
        metadata_path,
    )

    return {
        "factor_summary_global": factor_summary_path,
        "robust_factors": robust_factors_path,
        "candidate_biomarkers_final": candidate_biomarkers_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()
    root_dir = project_root()
    clinical_data, y = load_clinical_labels(root_dir)
    model_map = default_mofa_models(root_dir)
    selected_models = args.models or list(model_map.keys())
    seeds = list(range(args.n_seeds))

    base_output_dir = (
        root_dir / "data" / "biomarker_discovery" / "train_test_split" / "linear_svm"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for model_name in selected_models:
        saved = run_for_model(
            root_dir=root_dir,
            model_name=model_name,
            model_path=model_map[model_name],
            clinical_index=clinical_data.index,
            y=y,
            seeds=seeds,
            test_size=args.test_size,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
            top_factors=args.top_factors,
            min_stability=args.min_stability,
            top_features_per_sign=args.top_features_per_sign,
            base_output_dir=base_output_dir,
        )
        saved_paths.append(saved)

    print("Saved biomarker discovery outputs:")
    for saved in saved_paths:
        print(saved["factor_summary_global"])
        print(saved["robust_factors"])
        print(saved["candidate_biomarkers_final"])


if __name__ == "__main__":
    main()
