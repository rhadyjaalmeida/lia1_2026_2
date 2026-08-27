"""Descoberta de padrões relevantes de churn usando somente dados observados.

As funções deste módulo não contêm métricas, percentuais, conclusões ou
insights hardcoded. Todos os resultados são calculados a partir do DataFrame
recebido. Quando uma análise não é metodologicamente possível, uma exceção
clara é emitida para que a camada de apresentação possa informar a limitação.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_MIN_GROUP_SIZE = 30


def _validate_binary_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise KeyError(f"Coluna-alvo '{target}' não encontrada no dataset.")
    values = pd.Series(df[target]).dropna().unique()
    if len(values) != 2:
        raise ValueError(
            f"A análise exige alvo binário; '{target}' possui {len(values)} valores distintos."
        )
    if not set(values).issubset({0, 1, False, True}):
        raise ValueError(
            f"A coluna '{target}' deve usar 0/1 (ou booleanos) para calcular taxa de churn."
        )


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Ajusta p-values pelo procedimento Benjamini-Hochberg (FDR)."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    valid = p[valid_mask]
    if valid.size == 0:
        return adjusted

    order = np.argsort(valid)
    ranked = valid[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    restored = np.empty_like(q)
    restored[order] = q
    adjusted[valid_mask] = restored
    return adjusted


def categorical_churn_summary(
    df: pd.DataFrame,
    column: str,
    target: str = "churn",
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Resume tamanho, churners, taxa e desvio da taxa geral por categoria."""
    _validate_binary_target(df, target)
    if column not in df.columns:
        raise KeyError(f"Coluna '{column}' não encontrada no dataset.")

    data = df[[column, target]].dropna(subset=[target]).copy()
    data[column] = data[column].astype("object").where(data[column].notna(), "Não informado")
    overall = float(data[target].mean())

    out = (
        data.groupby(column, dropna=False)[target]
        .agg(total_clientes="size", churners="sum", taxa_churn="mean")
        .reset_index()
    )
    out["taxa_geral"] = overall
    out["desvio_pp"] = (out["taxa_churn"] - overall) * 100
    out["taxa_churn_pct"] = out["taxa_churn"] * 100
    out["taxa_geral_pct"] = overall * 100
    out["amostra_reduzida"] = out["total_clientes"] < int(min_group_size)
    return out.sort_values("desvio_pp", ascending=False).reset_index(drop=True)


def cramers_v_association(
    df: pd.DataFrame,
    column: str,
    target: str = "churn",
) -> dict:
    """Calcula qui-quadrado e Cramér's V entre uma categórica e o churn."""
    _validate_binary_target(df, target)
    if column not in df.columns:
        raise KeyError(f"Coluna '{column}' não encontrada no dataset.")

    data = df[[column, target]].dropna().copy()
    contingency = pd.crosstab(data[column], data[target])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "variavel": column,
            "n": len(data),
            "n_categorias": contingency.shape[0],
            "chi2": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "valido": False,
            "motivo": "Quantidade insuficiente de categorias/classes para o teste.",
        }

    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
    return {
        "variavel": column,
        "n": int(n),
        "n_categorias": int(contingency.shape[0]),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": float(v),
        "valido": True,
        "motivo": "",
    }


def numeric_churn_association(
    df: pd.DataFrame,
    column: str,
    target: str = "churn",
) -> dict:
    """Compara uma variável numérica entre churn e não churn.

    Usa Mann-Whitney U como teste robusto e rank-biserial como tamanho de
    efeito. Também retorna médias/medianas observadas, sem inferir causalidade.
    """
    _validate_binary_target(df, target)
    if column not in df.columns:
        raise KeyError(f"Coluna '{column}' não encontrada no dataset.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"A coluna '{column}' não é numérica.")

    data = df[[column, target]].dropna()
    churn = data.loc[data[target] == 1, column].astype(float)
    stayed = data.loc[data[target] == 0, column].astype(float)
    if len(churn) < 2 or len(stayed) < 2:
        return {
            "variavel": column,
            "n": len(data),
            "n_churn": len(churn),
            "n_permaneceu": len(stayed),
            "p_value": np.nan,
            "rank_biserial": np.nan,
            "valido": False,
            "motivo": "Quantidade insuficiente de observações em uma das classes.",
        }

    u_stat, p_value = stats.mannwhitneyu(churn, stayed, alternative="two-sided")
    rank_biserial = 2 * float(u_stat) / (len(churn) * len(stayed)) - 1

    return {
        "variavel": column,
        "n": int(len(data)),
        "n_churn": int(len(churn)),
        "n_permaneceu": int(len(stayed)),
        "media_churn": float(churn.mean()),
        "media_permaneceu": float(stayed.mean()),
        "mediana_churn": float(churn.median()),
        "mediana_permaneceu": float(stayed.median()),
        "diferenca_media": float(churn.mean() - stayed.mean()),
        "diferenca_mediana": float(churn.median() - stayed.median()),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial": float(rank_biserial),
        "valido": True,
        "motivo": "",
    }


def build_association_rankings(
    df: pd.DataFrame,
    target: str = "churn",
    exclude_columns: Iterable[str] | None = None,
    max_categories: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria rankings separados para numéricas e categóricas.

    As escalas NÃO são misturadas: numéricas usam |rank-biserial| e
    categóricas usam Cramér's V. P-values são ajustados por FDR dentro de
    cada família de testes.
    """
    _validate_binary_target(df, target)
    exclude = set(exclude_columns or []) | {target}

    numeric_rows = []
    categorical_rows = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # IDs quase únicos não são tratados como variáveis explicativas.
            nunique = df[col].nunique(dropna=True)
            if nunique > 0 and nunique / max(len(df), 1) > 0.95:
                continue
            numeric_rows.append(numeric_churn_association(df, col, target))
        else:
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= max_categories:
                categorical_rows.append(cramers_v_association(df, col, target))

    num_df = pd.DataFrame(numeric_rows)
    cat_df = pd.DataFrame(categorical_rows)

    if not num_df.empty:
        num_df["p_value_fdr"] = benjamini_hochberg(num_df["p_value"])
        num_df["magnitude_efeito"] = num_df["rank_biserial"].abs()
        num_df = num_df.sort_values("magnitude_efeito", ascending=False).reset_index(drop=True)

    if not cat_df.empty:
        cat_df["p_value_fdr"] = benjamini_hochberg(cat_df["p_value"])
        cat_df = cat_df.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    return num_df, cat_df


def top_categorical_segments(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    target: str = "churn",
    min_group_size: int | None = None,
    min_group_fraction: float = 0.01,
) -> pd.DataFrame:
    """Une segmentos categóricos e ordena pelo desvio absoluto da taxa geral."""
    _validate_binary_target(df, target)
    threshold = max(
        int(min_group_size or DEFAULT_MIN_GROUP_SIZE),
        int(np.ceil(len(df) * float(min_group_fraction))),
    )
    rows = []
    for col in categorical_columns:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) < 2:
            continue
        summary = categorical_churn_summary(df, col, target, threshold)
        summary = summary.rename(columns={col: "categoria"})
        summary.insert(0, "variavel", col)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["elegivel_amostra"] = out["total_clientes"] >= threshold
    out["magnitude_desvio_pp"] = out["desvio_pp"].abs()
    return out.sort_values(
        ["elegivel_amostra", "magnitude_desvio_pp", "total_clientes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def numeric_quantile_churn_summary(
    df: pd.DataFrame,
    column: str,
    target: str = "churn",
    q: int = 10,
    min_unique: int = 4,
) -> pd.DataFrame:
    """Calcula taxa de churn por quantis de uma variável numérica."""
    _validate_binary_target(df, target)
    if column not in df.columns:
        raise KeyError(f"Coluna '{column}' não encontrada no dataset.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"A coluna '{column}' não é numérica.")

    data = df[[column, target]].dropna().copy()
    if data[column].nunique() < min_unique:
        raise ValueError(
            f"'{column}' não possui valores distintos suficientes para análise por quantis."
        )

    effective_q = min(q, int(data[column].nunique()))
    bins = pd.qcut(data[column], q=effective_q, duplicates="drop")
    data["faixa"] = bins
    out = (
        data.groupby("faixa", observed=True)[target]
        .agg(total_clientes="size", churners="sum", taxa_churn="mean")
        .reset_index()
    )
    out["ordem"] = np.arange(1, len(out) + 1)
    out["taxa_churn_pct"] = out["taxa_churn"] * 100
    out["limite_inferior"] = out["faixa"].map(lambda x: float(x.left))
    out["limite_superior"] = out["faixa"].map(lambda x: float(x.right))
    out["faixa_label"] = out["faixa"].astype(str)
    return out


def combination_churn_summary(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    target: str = "churn",
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Taxa de churn para combinações de duas variáveis categóricas."""
    _validate_binary_target(df, target)
    for col in (col_a, col_b):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada no dataset.")

    data = df[[col_a, col_b, target]].dropna(subset=[target]).copy()
    data[col_a] = data[col_a].astype("object").where(data[col_a].notna(), "Não informado")
    data[col_b] = data[col_b].astype("object").where(data[col_b].notna(), "Não informado")
    overall = float(data[target].mean())

    out = (
        data.groupby([col_a, col_b], dropna=False)[target]
        .agg(total_clientes="size", churners="sum", taxa_churn="mean")
        .reset_index()
    )
    out["taxa_churn_pct"] = out["taxa_churn"] * 100
    out["taxa_geral_pct"] = overall * 100
    out["desvio_pp"] = (out["taxa_churn"] - overall) * 100
    out["amostra_reduzida"] = out["total_clientes"] < int(min_group_size)
    return out.sort_values("desvio_pp", ascending=False).reset_index(drop=True)


def candidate_categorical_pairs(
    df: pd.DataFrame,
    columns: Iterable[str],
    max_categories_each: int = 10,
) -> list[tuple[str, str]]:
    """Retorna pares categóricos com cardinalidade administrável para heatmaps."""
    valid = [
        c for c in columns
        if c in df.columns and 2 <= df[c].nunique(dropna=True) <= max_categories_each
    ]
    return list(combinations(valid, 2))


def build_dynamic_discoveries(
    numeric_ranking: pd.DataFrame,
    categorical_ranking: pd.DataFrame,
    segments: pd.DataFrame,
    alpha: float = 0.05,
    min_numeric_effect: float = 0.10,
    min_categorical_effect: float = 0.10,
    min_segment_delta_pp: float = 2.0,
    max_items_per_type: int = 5,
) -> pd.DataFrame:
    """Gera registros de descoberta somente quando critérios explícitos são atendidos.

    Não usa score composto. Cada tipo de evidência mantém sua própria métrica:
    rank-biserial, Cramér's V ou desvio em pontos percentuais.
    """
    discoveries: list[dict] = []

    if not numeric_ranking.empty:
        subset = numeric_ranking[
            (numeric_ranking["valido"])
            & (numeric_ranking["p_value_fdr"] <= alpha)
            & (numeric_ranking["magnitude_efeito"] >= min_numeric_effect)
        ].head(max_items_per_type)
        for _, r in subset.iterrows():
            direction = "maior" if r["rank_biserial"] > 0 else "menor"
            discoveries.append({
                "tipo": "Numérica",
                "variavel": r["variavel"],
                "categoria": "",
                "metrica": "rank-biserial",
                "magnitude": float(r["rank_biserial"]),
                "p_value_fdr": float(r["p_value_fdr"]),
                "n": int(r["n"]),
                "resultado": (
                    f"Valores de {r['variavel']} tendem a ser {direction} no grupo Churn "
                    f"nesta amostra; efeito rank-biserial={r['rank_biserial']:.3f}."
                ),
                "limitacao": "Associação observacional; não implica causalidade.",
            })

    if not categorical_ranking.empty:
        subset = categorical_ranking[
            (categorical_ranking["valido"])
            & (categorical_ranking["p_value_fdr"] <= alpha)
            & (categorical_ranking["cramers_v"] >= min_categorical_effect)
        ].head(max_items_per_type)
        for _, r in subset.iterrows():
            discoveries.append({
                "tipo": "Categórica",
                "variavel": r["variavel"],
                "categoria": "",
                "metrica": "Cramér's V",
                "magnitude": float(r["cramers_v"]),
                "p_value_fdr": float(r["p_value_fdr"]),
                "n": int(r["n"]),
                "resultado": (
                    f"{r['variavel']} apresenta associação com churn nesta amostra; "
                    f"Cramér's V={r['cramers_v']:.3f}."
                ),
                "limitacao": "Associação observacional; categorias raras exigem cautela.",
            })

    if not segments.empty:
        subset = segments[
            (segments["elegivel_amostra"])
            & (segments["magnitude_desvio_pp"] >= min_segment_delta_pp)
        ].head(max_items_per_type)
        for _, r in subset.iterrows():
            discoveries.append({
                "tipo": "Segmento",
                "variavel": r["variavel"],
                "categoria": str(r["categoria"]),
                "metrica": "desvio da taxa geral (p.p.)",
                "magnitude": float(r["desvio_pp"]),
                "p_value_fdr": np.nan,
                "n": int(r["total_clientes"]),
                "resultado": (
                    f"O segmento {r['variavel']}={r['categoria']} possui taxa de churn "
                    f"{r['taxa_churn_pct']:.2f}% vs. {r['taxa_geral_pct']:.2f}% na base "
                    f"(diferença de {r['desvio_pp']:+.2f} p.p.)."
                ),
                "limitacao": "Comparação descritiva; não implica efeito causal do segmento.",
            })

    return pd.DataFrame(discoveries)
