"""
Funções utilitárias para avaliação de modelos de classificação de churn.

Estas funções são wrappers simples sobre o scikit-learn para centralizar
o cálculo das métricas usadas de forma consistente entre os modelos
comparados no notebook (Logistic Regression, Decision Tree, Random Forest,
XGBoost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Calcula as principais métricas de avaliação para um problema de churn
    (classificação binária, tipicamente desbalanceada).

    Parameters
    ----------
    y_true : array-like
        Rótulos verdadeiros (0 = permaneceu, 1 = churn).
    y_pred : array-like
        Rótulos previstos pelo modelo (classe discreta, 0 ou 1).
    y_proba : array-like
        Probabilidade prevista para a classe positiva (churn).

    Returns
    -------
    dict
        Dicionário com precision, recall, f1, roc_auc, pr_auc e mcc.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def metrics_to_dataframe(model_metrics: dict[str, dict]) -> pd.DataFrame:
    """
    Converte um dicionário {nome_do_modelo: métricas} em um DataFrame
    pronto para exibição/exportação (ex.: reports/results/).

    Parameters
    ----------
    model_metrics : dict
        Ex.: {"Random Forest": {"precision": 0.8, ...}, "XGBoost": {...}}

    Returns
    -------
    pd.DataFrame
        Uma linha por modelo, uma coluna por métrica.
    """
    df = pd.DataFrame(model_metrics).T
    df.index.name = "model"
    return df.sort_values("pr_auc", ascending=False)


def confusion_matrix_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Retorna a matriz de confusão como DataFrame rotulado e legível."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Real: Permaneceu", "Real: Churn"],
        columns=["Previsto: Permaneceu", "Previsto: Churn"],
    )
