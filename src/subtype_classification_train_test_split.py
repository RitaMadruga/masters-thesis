import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from subtype_classification_common import (
    CLASSIFIER_LABELS,
    METRICS,
    build_strategy_comparison,
    classifier_configs,
    compute_metrics,
    default_mofa_models,
    get_score_values,
    load_clinical_labels,
    load_mofa_factors,
    metric_to_cv_name,
    project_root,
    serialize_params,
    summarize_results,
    write_csv,
    write_final_comparison,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run subtype classification with a train/test split across seeds and MOFA models."
    )
    parser.add_argument("--n-seeds", type=int, default=30, help="Number of seeds to run.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test fraction.")
    parser.add_argument("--inner-splits", type=int, default=5, help="Inner CV folds for tuning.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional MOFA model names to run. Defaults to all models.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=None,
        choices=list(classifier_configs().keys()),
        help="Optional classifiers to run. Defaults to all supported classifiers.",
    )
    return parser.parse_args()


def run_experiment_for_model(
    classifier_name: str,
    model_name: str,
    model_path: Path,
    clinical_index,
    y,
    seeds: list[int],
    test_size: float,
    inner_splits: int,
    n_jobs: int,
    strategy_dir: Path,
) -> dict[str, Path]:
    config = classifier_configs()[classifier_name]
    X = load_mofa_factors(model_path, clinical_index)

    output_dir = strategy_dir / classifier_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        model = config["builder"](seed)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=config["param_grid"],
            scoring=config["scoring"],
            refit="pr_auc_score",
            cv=inner_cv,
            n_jobs=n_jobs,
            return_train_score=True,
        )
        grid_search.fit(X_train, y_train)

        best_index = grid_search.best_index_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_score = get_score_values(best_model, X_test, config["score_method"])
        test_metrics = compute_metrics(y_test, y_pred, y_score)

        row = {
            "seed": seed,
            "best_params": serialize_params(grid_search.best_params_),
        }

        for metric in METRICS:
            cv_metric = metric_to_cv_name(metric)
            row[f"train_{metric}"] = float(grid_search.cv_results_[f"mean_train_{cv_metric}"][best_index])
            row[f"validation_{metric}"] = float(grid_search.cv_results_[f"mean_test_{cv_metric}"][best_index])
            row[f"test_{metric}"] = float(test_metrics[metric])

        seed_rows.append(row)

    per_seed_df = pd.DataFrame(seed_rows)
    summary_df = summarize_results(per_seed_df, scopes=["train", "validation", "test"])

    metadata = {
        "evaluation_strategy": "train_test_split",
        "classifier": classifier_name,
        "classifier_label": CLASSIFIER_LABELS[classifier_name],
        "model_name": model_name,
        "model_path": str(model_path),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "test_size": test_size,
        "inner_splits": inner_splits,
        "selection_metric": "pr_auc_score",
    }

    per_seed_path = output_dir / "per_seed_results.csv"
    summary_path = output_dir / "summary_across_seeds.csv"
    metadata_path = output_dir / "run_metadata.json"

    write_csv(per_seed_df, per_seed_path)
    write_csv(summary_df, summary_path)
    write_json(metadata, metadata_path)

    return {
        "per_seed": per_seed_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    args = parse_args()

    root_dir = project_root()
    clinical_data, y = load_clinical_labels(root_dir)
    model_map = default_mofa_models(root_dir)
    selected_models = args.models or list(model_map.keys())
    selected_classifiers = args.classifiers or list(classifier_configs().keys())
    seeds = list(range(args.n_seeds))

    strategy_dir = root_dir / "data" / "classification_results" / "train_test_split"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for classifier_name in selected_classifiers:
        for model_name in selected_models:
            saved = run_experiment_for_model(
                classifier_name=classifier_name,
                model_name=model_name,
                model_path=model_map[model_name],
                clinical_index=clinical_data.index,
                y=y,
                seeds=seeds,
                test_size=args.test_size,
                inner_splits=args.inner_splits,
                n_jobs=args.n_jobs,
                strategy_dir=strategy_dir,
            )
            saved_paths.append(saved)

    comparison_df = build_strategy_comparison(
        strategy_name="train_test_split",
        strategy_dir=strategy_dir,
        scope_name="test",
    )
    final_df = write_final_comparison(strategy_dir.parent)

    print("Saved train/test split experiment outputs:")
    for saved in saved_paths:
        print(saved["summary"])
    if not comparison_df.empty:
        print(strategy_dir / "comparison_table.csv")
    if not final_df.empty:
        print(strategy_dir.parent / "final_comparison_table.csv")


if __name__ == "__main__":
    main()
