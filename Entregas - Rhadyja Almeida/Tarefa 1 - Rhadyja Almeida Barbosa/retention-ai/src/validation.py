"""Validação metodológica do RetentionAI.

Este módulo centraliza o fluxo de avaliação sem usar o conjunto final de teste
para escolher modelo, estratégia de desbalanceamento, hiperparâmetros,
calibração ou threshold.

Todos os números retornados são calculados a partir dos dados recebidos.
Nenhuma métrica ou resultado de modelo é hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42
MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
)
IMBALANCE_STRATEGIES = ("none", "class_weight", "smotenc")


@dataclass(frozen=True)
class Candidate:
    model_name: str
    strategy: str
    estimator: Any


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Retorna StratifiedKFold reproduzível."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def class_distribution_table(
    y_full: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Distribuição de classes para base completa, treino e teste."""
    rows: list[dict[str, Any]] = []
    for set_name, series in (
        ("Dataset completo", y_full),
        ("Treino", y_train),
        ("Teste", y_test),
    ):
        s = pd.Series(series).dropna()
        counts = s.value_counts().sort_index()
        for cls, count in counts.items():
            rows.append(
                {
                    "conjunto": set_name,
                    "classe": int(cls) if isinstance(cls, (int, np.integer, bool, np.bool_)) else cls,
                    "quantidade": int(count),
                    "percentual": float(count / len(s) * 100) if len(s) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def split_integrity_checks(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str = "churn",
) -> dict[str, Any]:
    """Executa testes de sanidade do split sem consultar resultados de modelo."""
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    full_idx = set(X.index)

    full_prev = float(y.mean())
    train_prev = float(y_train.mean())
    test_prev = float(y_test.mean())
    # Tolerância de arredondamento equivalente a no máximo uma observação na menor partição.
    rounding_tolerance = max(1 / max(len(y_train), 1), 1 / max(len(y_test), 1))
    stratification_preserved = max(
        abs(train_prev - full_prev),
        abs(test_prev - full_prev),
    ) <= rounding_tolerance

    return {
        "train_test_indices_disjoint": train_idx.isdisjoint(test_idx),
        "split_covers_original_rows": train_idx | test_idx == full_idx,
        "target_not_in_X": target_name not in X.columns,
        "train_xy_aligned": X_train.index.equals(y_train.index),
        "test_xy_aligned": X_test.index.equals(y_test.index),
        "stratification_preserved_with_rounding": bool(stratification_preserved),
        "stratification_difference_pp": float(abs(train_prev - test_prev) * 100),
        "train_prevalence": train_prev,
        "test_prevalence": test_prev,
        "full_prevalence": full_prev,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "full_size": int(len(y)),
    }


def _numeric_pipeline() -> SkPipeline:
    return SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> SkPipeline:
    return SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_standard_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Pré-processador padrão usado nos cenários sem resampling."""
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(), numeric_features),
            ("cat", _categorical_pipeline(), categorical_features),
        ],
        remainder="drop",
    )


def _smotenc_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Imputa numéricas e codifica categóricas ordinalmente antes do SMOTENC."""
    num = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=float,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num, numeric_features),
            ("cat", cat, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _smotenc_postprocessor(n_numeric: int, n_categorical: int) -> ColumnTransformer:
    """Escala numéricas e one-hot-encoda categóricas após o SMOTENC."""
    numeric_idx = list(range(n_numeric))
    categorical_idx = list(range(n_numeric, n_numeric + n_categorical))
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_idx),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_idx),
        ],
        remainder="drop",
    )


def _class_ratio(y_train: pd.Series) -> float:
    counts = pd.Series(y_train).value_counts()
    positives = float(counts.get(1, 0))
    negatives = float(counts.get(0, 0))
    if positives <= 0 or negatives <= 0:
        return 1.0
    return negatives / positives


def build_base_model(
    model_name: str,
    strategy: str,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Cria estimador base sem declarar qualquer configuração como ótima."""
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if strategy == "class_weight" else None,
            random_state=random_state,
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            class_weight="balanced" if strategy == "class_weight" else None,
            random_state=random_state,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced" if strategy == "class_weight" else None,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "XGBoost":
        params = {
            "n_estimators": 200,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": random_state,
        }
        if strategy == "class_weight":
            params["scale_pos_weight"] = _class_ratio(y_train)
        return XGBClassifier(**params)
    raise KeyError(f"Modelo não suportado: {model_name}")


def build_candidate_pipeline(
    model_name: str,
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = RANDOM_STATE,
) -> Any:
    """Monta pipeline anti-leakage para uma combinação modelo/estratégia."""
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    model = build_base_model(model_name, strategy, y_train, random_state=random_state)

    if strategy in {"none", "class_weight"}:
        return ImbPipeline(
            steps=[
                ("preprocessor", build_standard_preprocessor(numeric_features, categorical_features)),
                ("classifier", model),
            ]
        )

    if not categorical_features:
        raise ValueError("SMOTENC exige pelo menos uma variável categórica.")

    pre = _smotenc_preprocessor(numeric_features, categorical_features)
    cat_idx = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )
    sampler = SMOTENC(
        categorical_features=cat_idx,
        random_state=random_state,
    )
    post = _smotenc_postprocessor(len(numeric_features), len(categorical_features))
    return ImbPipeline(
        steps=[
            ("pre_smotenc", pre),
            ("smotenc", sampler),
            ("post_smotenc", post),
            ("classifier", model),
        ]
    )


def scoring_dict() -> dict[str, Any]:
    """Scorers usados de forma consistente em todos os candidatos."""
    return {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef),
    }


def evaluate_candidate_cv(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    model_name: str,
    strategy: str,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Avalia um candidato somente via cross-validation no conjunto de treino."""
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        scoring=scoring_dict(),
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    row: dict[str, Any] = {"model": model_name, "strategy": strategy}
    for metric in scoring_dict().keys():
        test_values = np.asarray(scores[f"test_{metric}"], dtype=float)
        train_values = np.asarray(scores[f"train_{metric}"], dtype=float)
        row[f"cv_{metric}_mean"] = float(np.mean(test_values))
        row[f"cv_{metric}_std"] = float(np.std(test_values, ddof=1)) if len(test_values) > 1 else 0.0
        row[f"train_{metric}_mean"] = float(np.mean(train_values))
    row["n_folds"] = int(cv.get_n_splits())
    row["pr_auc_generalization_gap"] = row["train_pr_auc_mean"] - row["cv_pr_auc_mean"]
    return row


def compare_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    cv: StratifiedKFold,
    model_names: Iterable[str] = MODEL_NAMES,
    strategies: Iterable[str] = IMBALANCE_STRATEGIES,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any]]:
    """Compara modelos × estratégias sem acessar o conjunto de teste."""
    rows: list[dict[str, Any]] = []
    estimators: dict[tuple[str, str], Any] = {}

    for strategy in strategies:
        for model_name in model_names:
            estimator = build_candidate_pipeline(
                model_name,
                strategy,
                X_train,
                y_train,
                numeric_features,
                categorical_features,
            )
            row = evaluate_candidate_cv(
                estimator,
                X_train,
                y_train,
                cv=cv,
                model_name=model_name,
                strategy=strategy,
            )
            rows.append(row)
            estimators[(model_name, strategy)] = estimator

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["cv_pr_auc_mean", "cv_pr_auc_std", "cv_recall_mean", "cv_mcc_mean"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return df, estimators


def evaluate_dummy_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Avalia DummyClassifier e registra prevalência como referência de AP."""
    preprocessor = build_standard_preprocessor(numeric_features, categorical_features)
    dummy = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
        ]
    )
    row = evaluate_candidate_cv(
        dummy,
        X_train,
        y_train,
        cv=cv,
        model_name="DummyClassifier",
        strategy="prior",
    )
    row["positive_prevalence"] = float(pd.Series(y_train).mean())
    return pd.DataFrame([row])


def one_standard_error_selection(cv_results: pd.DataFrame) -> dict[str, Any]:
    """Seleciona candidato com regra de 1 erro-padrão e desempates transparentes.

    1. Localiza o maior PR-AUC médio.
    2. Calcula o erro-padrão desse melhor candidato.
    3. Considera elegíveis candidatos dentro de 1 erro-padrão do melhor.
    4. Entre os elegíveis, prioriza menor variabilidade; depois maior Recall,
       MCC e Precision.

    Isso evita declarar um grande vencedor quando diferenças são menores que a
    variabilidade da validação cruzada.
    """
    if cv_results.empty:
        raise ValueError("Resultados de cross-validation vazios.")

    ranked = cv_results.sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    n_folds = int(best.get("n_folds", 5))
    se = float(best["cv_pr_auc_std"]) / np.sqrt(max(n_folds, 1))
    cutoff = float(best["cv_pr_auc_mean"]) - se
    eligible = ranked[ranked["cv_pr_auc_mean"] >= cutoff].copy()
    selected = eligible.sort_values(
        ["cv_pr_auc_std", "cv_recall_mean", "cv_mcc_mean", "cv_precision_mean"],
        ascending=[True, False, False, False],
    ).iloc[0]

    return {
        "selected_model": selected["model"],
        "selected_strategy": selected["strategy"],
        "best_pr_auc_mean": float(best["cv_pr_auc_mean"]),
        "best_pr_auc_std": float(best["cv_pr_auc_std"]),
        "one_se_cutoff": float(cutoff),
        "eligible_candidates": int(len(eligible)),
        "selected_pr_auc_mean": float(selected["cv_pr_auc_mean"]),
        "selected_pr_auc_std": float(selected["cv_pr_auc_std"]),
        "selection_rule": (
            "Regra de 1 erro-padrão no PR-AUC; entre candidatos elegíveis, "
            "menor variabilidade, maior Recall, MCC e Precision."
        ),
    }


def _parameter_space(
    model_name: str,
    strategy: str,
    y_train: pd.Series,
) -> dict[str, list[Any]]:
    """Espaço moderado de busca; nenhum valor é tratado previamente como vencedor."""
    if model_name == "Logistic Regression":
        params: dict[str, list[Any]] = {
            "classifier__C": np.logspace(-2, 2, 9).tolist(),
            "classifier__solver": ["liblinear", "lbfgs"],
        }
        return params

    if model_name == "Decision Tree":
        return {
            "classifier__max_depth": [None, 3, 5, 7, 9, 12, 16],
            "classifier__min_samples_split": [2, 5, 10, 20, 40],
            "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
        }

    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": [150, 250, 400, 600],
            "classifier__max_depth": [None, 5, 8, 12, 16, 24],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.5],
            "classifier__bootstrap": [True, False],
        }

    if model_name == "XGBoost":
        params = {
            "classifier__n_estimators": [150, 250, 400, 600],
            "classifier__max_depth": [2, 3, 4, 5, 6, 8],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 10],
            "classifier__gamma": [0, 0.1, 0.3, 0.7],
            "classifier__reg_alpha": [0, 0.01, 0.1, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
        if strategy == "class_weight":
            ratio = _class_ratio(y_train)
            params["classifier__scale_pos_weight"] = [ratio * 0.75, ratio, ratio * 1.25]
        return params

    raise KeyError(f"Modelo não suportado: {model_name}")


def tune_candidate(
    estimator: Any,
    model_name: str,
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_iter: int = 16,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Tuning usando somente treino e PR-AUC/Average Precision."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=_parameter_space(model_name, strategy, y_train),
        n_iter=n_iter,
        scoring="average_precision",
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def summarize_tuning(
    search: RandomizedSearchCV,
    model_name: str,
    strategy: str,
) -> pd.DataFrame:
    """Resume a melhor configuração realmente encontrada pelo RandomizedSearchCV."""
    idx = int(search.best_index_)
    cv = search.cv_results_
    row = {
        "model": model_name,
        "strategy": strategy,
        "best_cv_pr_auc": float(search.best_score_),
        "best_cv_pr_auc_std": float(cv["std_test_score"][idx]),
        "best_params": search.best_params_,
        "n_candidates_tested": int(len(cv["params"])),
    }
    return pd.DataFrame([row])


def oof_probabilities(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int = -1,
) -> np.ndarray:
    """Probabilidades out-of-fold para decisões de calibração/threshold."""
    probs = cross_val_predict(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    return np.asarray(probs)[:, 1]


def probability_quality_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def compare_calibration_methods(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    outer_cv: StratifiedKFold,
    methods: Iterable[str] = ("sigmoid", "isotonic"),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compara probabilidades OOF originais e calibradas sem tocar no teste final."""
    rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}

    base_probs = oof_probabilities(estimator, X_train, y_train, outer_cv, n_jobs=-1)
    base_metrics = probability_quality_metrics(y_train, base_probs)
    rows.append({"method": "uncalibrated", **base_metrics})
    probabilities["uncalibrated"] = base_probs

    for method in methods:
        calibrated = CalibratedClassifierCV(
            estimator=clone(estimator),
            method=method,
            cv=3,
            n_jobs=-1,
        )
        # outer CV mantém a comparação fora da amostra; n_jobs=1 evita nested parallelism excessivo.
        probs = cross_val_predict(
            calibrated,
            X_train,
            y_train,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        metrics = probability_quality_metrics(y_train, probs)
        rows.append({"method": method, **metrics})
        probabilities[method] = np.asarray(probs)

    results = pd.DataFrame(rows).sort_values(
        ["brier_score", "pr_auc"], ascending=[True, False]
    ).reset_index(drop=True)
    return results, probabilities


def choose_calibration_method(calibration_results: pd.DataFrame) -> str:
    """Escolhe menor Brier Score, com PR-AUC como desempate transparente."""
    if calibration_results.empty:
        raise ValueError("Resultados de calibração vazios.")
    row = calibration_results.sort_values(
        ["brier_score", "pr_auc"], ascending=[True, False]
    ).iloc[0]
    return str(row["method"])


def wrap_calibrated_estimator(estimator: Any, method: str) -> Any:
    if method == "uncalibrated":
        return clone(estimator)
    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method=method,
        cv=3,
        n_jobs=-1,
    )


def threshold_metrics_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calcula Precision/Recall/F1/MCC e contagens para vários thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    y_arr = np.asarray(y_true)
    for threshold in thresholds:
        pred = (np.asarray(y_proba) >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_arr, pred, zero_division=0)),
                "recall": float(recall_score(y_arr, pred, zero_division=0)),
                "f1": float(f1_score(y_arr, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_arr, pred)),
                "predicted_positive": int(pred.sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def select_thresholds(
    threshold_df: pd.DataFrame,
    recall_target: float | None = None,
) -> dict[str, Any]:
    """Seleciona thresholds usando somente a tabela de validação OOF."""
    if threshold_df.empty:
        raise ValueError("Tabela de thresholds vazia.")

    f1_row = threshold_df.sort_values(
        ["f1", "precision", "mcc"], ascending=[False, False, False]
    ).iloc[0]

    default_row = threshold_df.iloc[(threshold_df["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    out: dict[str, Any] = {
        "default_0_5": float(default_row["threshold"]),
        "f1_optimal": float(f1_row["threshold"]),
        "f1_at_default": float(default_row["f1"]),
        "f1_at_optimal": float(f1_row["f1"]),
        "selected_threshold": float(f1_row["threshold"]),
        "selection_reason": (
            "Sem custos empresariais parametrizados, o threshold final é o que maximiza F1 "
            "nas previsões out-of-fold do conjunto de treino."
        ),
    }
    if recall_target is not None:
        eligible = threshold_df[threshold_df["recall"] >= recall_target]
        if not eligible.empty:
            row = eligible.sort_values(
                ["precision", "f1", "threshold"], ascending=[False, False, False]
            ).iloc[0]
            out["recall_target"] = float(recall_target)
            out["recall_oriented_threshold"] = float(row["threshold"])
            out["recall_at_recall_oriented_threshold"] = float(row["recall"])
            out["precision_at_recall_oriented_threshold"] = float(row["precision"])
    return out


def calibration_curve_table(
    y_true: pd.Series,
    probabilities: dict[str, np.ndarray],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for method, probs in probabilities.items():
        fraction_pos, mean_pred = calibration_curve(
            y_true,
            probs,
            n_bins=n_bins,
            strategy="quantile",
        )
        for observed, predicted in zip(fraction_pos, mean_pred):
            rows.append(
                {
                    "method": method,
                    "mean_predicted_probability": float(predicted),
                    "observed_fraction_positive": float(observed),
                }
            )
    return pd.DataFrame(rows)


def fit_final_estimator(
    estimator: Any,
    calibration_method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Ajusta o estimador final somente após todas as decisões de validação."""
    final_estimator = wrap_calibrated_estimator(estimator, calibration_method)
    final_estimator.fit(X_train, y_train)
    return final_estimator


def final_test_evaluation(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Avaliação única no test set após modelo/calibração/threshold estarem definidos."""
    probs = np.asarray(estimator.predict_proba(X_test))[:, 1]
    pred = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, probs)),
        "threshold": float(threshold),
    }
    return metrics, pred, probs


def top_k_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Recall@K, Precision@K, Lift@K e churn rate no topo do ranking."""
    data = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    data = data.sort_values("p", ascending=False).reset_index(drop=True)
    total_positives = int(data["y"].sum())
    prevalence = float(data["y"].mean()) if len(data) else np.nan
    rows = []
    for fraction in fractions:
        k = max(1, int(np.ceil(len(data) * fraction)))
        top = data.head(k)
        hits = int(top["y"].sum())
        recall_at_k = hits / total_positives if total_positives else np.nan
        precision_at_k = hits / k if k else np.nan
        lift_at_k = precision_at_k / prevalence if prevalence and prevalence > 0 else np.nan
        rows.append(
            {
                "fraction": float(fraction),
                "top_percent": float(fraction * 100),
                "n_selected": int(k),
                "positives_captured": hits,
                "recall_at_k": float(recall_at_k),
                "precision_at_k": float(precision_at_k),
                "lift_at_k": float(lift_at_k),
                "churn_rate_top_k": float(precision_at_k),
                "baseline_prevalence": float(prevalence),
            }
        )
    return pd.DataFrame(rows)


def risk_decile_table(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Taxa de churn por decil de risco previsto."""
    data = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    if data["p"].nunique() < 2:
        raise ValueError("Probabilidades insuficientemente variadas para criar decis.")
    q = min(n_bins, int(data["p"].nunique()))
    data["risk_decile"] = pd.qcut(
        data["p"].rank(method="first"), q=q, labels=False, duplicates="drop"
    )
    # qcut cria 0=menor risco; inverter para 1=maior risco.
    max_decile = int(data["risk_decile"].max())
    data["risk_decile"] = max_decile - data["risk_decile"] + 1
    out = (
        data.groupby("risk_decile", as_index=False)
        .agg(
            n=("y", "size"),
            churners=("y", "sum"),
            churn_rate=("y", "mean"),
            min_probability=("p", "min"),
            max_probability=("p", "max"),
            mean_probability=("p", "mean"),
        )
        .sort_values("risk_decile")
    )
    return out


def build_validation_audit(
    split_checks: dict[str, Any],
    *,
    cross_validation_completed: bool,
    imbalance_strategies_compared: bool,
    hyperparameter_tuning_completed: bool,
    leakage_audit_completed: bool,
    calibration_evaluated: bool,
    threshold_selected_without_test: bool,
    overfitting_checked: bool,
    final_test_evaluated: bool,
) -> dict[str, Any]:
    """Status objetivo da metodologia. Flags devem refletir etapas realmente executadas."""
    return {
        "test_set_isolated": bool(
            split_checks.get("train_test_indices_disjoint", False)
            and split_checks.get("target_not_in_X", False)
        ),
        "stratified_split": bool(
            split_checks.get("stratification_preserved_with_rounding", False)
        ),
        "cross_validation_completed": bool(cross_validation_completed),
        "imbalance_strategies_compared": bool(imbalance_strategies_compared),
        "hyperparameter_tuning_completed": bool(hyperparameter_tuning_completed),
        "leakage_audit_completed": bool(leakage_audit_completed),
        "calibration_evaluated": bool(calibration_evaluated),
        "threshold_selected_without_test": bool(threshold_selected_without_test),
        "overfitting_checked": bool(overfitting_checked),
        "final_test_evaluated": bool(final_test_evaluated),
    }


def confusion_matrix_table(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Real: Permaneceu", "Real: Churn"],
        columns=["Previsto: Permaneceu", "Previsto: Churn"],
    )
