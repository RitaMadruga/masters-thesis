from pathlib import Path

import mofax as mfx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


def load_mofa_factors(model_path, clinical_index):
    model = mfx.mofa_model(str(model_path))
    factors = model.get_factors(df=True)
    factors = factors.loc[clinical_index]
    return factors


def build_rf(seed):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def metric_dict(y_true, y_pred, y_prob):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def section_lines(title, values):
    lines = [title]
    for metric_name, metric_value in values.items():
        lines.append(f"  {metric_name}: {metric_value:.6f}")
    return lines


def run_for_model(model_name, model_path, clinical_index, y, seeds, scoring, base_output_dir):
    X = load_mofa_factors(model_path, clinical_index)

    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in seeds:
        np.random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rf_model = build_rf(seed)

        cv_scores = cross_validate(
            rf_model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train)
        y_test_pred = rf_model.predict(X_test)
        y_test_prob = rf_model.predict_proba(X_test)[:, 1]
        test_metrics = metric_dict(y_test, y_test_pred, y_test_prob)

        record = {"seed": seed}
        for metric_name in scoring.keys():
            record[f"cv_train_{metric_name}"] = float(np.mean(cv_scores[f"train_{metric_name}"]))
            record[f"cv_validation_{metric_name}"] = float(np.mean(cv_scores[f"test_{metric_name}"]))
            record[f"test_{metric_name}"] = float(test_metrics[metric_name])
        rows.append(record)

        seed_lines = [
            "Random Forest + MOFA Per-Seed Report",
            f"model_name: {model_name}",
            f"model_path: {model_path}",
            f"seed: {seed}",
            "",
        ]
        seed_lines.extend(
            section_lines(
                "CV Train (5-fold mean):",
                {k: record[f"cv_train_{k}"] for k in scoring.keys()},
            )
        )
        seed_lines.append("")
        seed_lines.extend(
            section_lines(
                "CV Validation (5-fold mean):",
                {k: record[f"cv_validation_{k}"] for k in scoring.keys()},
            )
        )
        seed_lines.append("")
        seed_lines.extend(
            section_lines(
                "Final Test:",
                {k: record[f"test_{k}"] for k in scoring.keys()},
            )
        )
        seed_lines.append("")

        seed_report_path = output_dir / f"seed_{seed:02d}.txt"
        seed_report_path.write_text("\n".join(seed_lines), encoding="utf-8")

    results_df = pd.DataFrame(rows)

    aggregate_rows = []
    scopes = ["cv_train", "cv_validation", "test"]
    for scope in scopes:
        for metric_name in scoring.keys():
            column_name = f"{scope}_{metric_name}"
            aggregate_rows.append(
                {
                    "scope": scope,
                    "metric": metric_name,
                    "min": float(results_df[column_name].min()),
                    "max": float(results_df[column_name].max()),
                    "mean": float(results_df[column_name].mean()),
                    "median": float(results_df[column_name].median()),
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows)

    summary_lines = [
        "Random Forest + MOFA Aggregate Report",
        f"model_name: {model_name}",
        f"model_path: {model_path}",
        f"n_seeds: {len(seeds)}",
        f"seeds: {seeds}",
        "",
        "min/max/mean/median across all seeds:",
        aggregate_df.to_string(index=False),
        "",
    ]
    summary_report_path = output_dir / "summary_all_seeds.txt"
    summary_report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved aggregate summary to: {summary_report_path}")
    return summary_report_path


def main():
    root_dir = Path(__file__).resolve().parents[1]

    mofa_models = {
        "mofa_trained_lg2": root_dir / "data" / "latent" / "mofa_trained_lg2.hdf5",
        "mofa_trained_vsn": root_dir / "data" / "latent" / "mofa_trained_vsn.hdf5",
        "mofa_trained_lg2_fs": root_dir / "data" / "latent" / "mofa_trained_lg2_fs.hdf5",
        "mofa_trained_vsn_fs": root_dir / "data" / "latent" / "mofa_trained_vsn_fs.hdf5",
    }

    clinical_path = root_dir / "data" / "cleaned_data" / "clinical_cleaned.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    y = clinical_data["tumor_subtype"].map({"other_subtype": 1, "ductal_type": 0})

    base_output_dir = root_dir / "data" / "classification_results"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(30))
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "pr_auc": "average_precision",
    }

    summary_paths = []
    for model_name, model_path in mofa_models.items():
        summary_path = run_for_model(
            model_name=model_name,
            model_path=model_path,
            clinical_index=clinical_data.index,
            y=y,
            seeds=seeds,
            scoring=scoring,
            base_output_dir=base_output_dir,
        )
        summary_paths.append(summary_path)

    print("Completed all models. Summary files:")
    for path in summary_paths:
        print(path)


if __name__ == "__main__":
    main()