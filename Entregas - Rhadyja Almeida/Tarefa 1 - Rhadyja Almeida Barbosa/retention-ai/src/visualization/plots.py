"""
Funções utilitárias de visualização (Plotly) para o projeto RetentionAI.

Todas as funções retornam um objeto `plotly.graph_objects.Figure`, que pode
ser:
    - exibido diretamente no notebook com `fig.show()`;
    - salvo como imagem estática em reports/figures/ com
      `fig.write_image("reports/figures/nome.png")` (requer o pacote
      `kaleido` para exportação estática).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CHURN_COLOR_MAP = {
    "Churn": "#EF553B",
    "Permaneceu": "#00CC96",
}


def plot_churn_distribution(df: pd.DataFrame, churn_col: str = "churn_label") -> go.Figure:
    """
    Gráfico de barras interativo com a distribuição de clientes entre
    "Churn" e "Permaneceu", incluindo contagem absoluta e percentual.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna de rótulo de churn já traduzida para
        texto ("Churn" / "Permaneceu").
    churn_col : str
        Nome da coluna categórica de churn.
    """
    counts = df[churn_col].value_counts().reset_index()
    counts.columns = [churn_col, "quantidade"]
    counts["percentual"] = (counts["quantidade"] / counts["quantidade"].sum() * 100).round(1)

    fig = px.bar(
        counts,
        x=churn_col,
        y="quantidade",
        color=churn_col,
        color_discrete_map=CHURN_COLOR_MAP,
        text=counts.apply(lambda r: f"{r['quantidade']} ({r['percentual']}%)", axis=1),
        title="Distribuição de Clientes: Churn vs. Permaneceu",
        labels={churn_col: "Status do Cliente", "quantidade": "Número de Clientes"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Número de Clientes")
    return fig


def plot_roc_curves(roc_data: dict[str, tuple]) -> go.Figure:
    """
    Sobrepõe curvas ROC de múltiplos modelos em um único gráfico interativo.

    Parameters
    ----------
    roc_data : dict
        Ex.: {"Random Forest": (fpr_array, tpr_array, auc_value), ...}
    """
    fig = go.Figure()
    for model_name, (fpr, tpr, auc_value) in roc_data.items():
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc_value:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório", line=dict(dash="dash"))
    )
    fig.update_layout(
        title="Curva ROC — Comparação de Modelos",
        xaxis_title="Taxa de Falsos Positivos (1 - Especificidade)",
        yaxis_title="Taxa de Verdadeiros Positivos (Recall)",
    )
    return fig


def plot_precision_recall_curves(pr_data: dict[str, tuple]) -> go.Figure:
    """
    Sobrepõe curvas Precision-Recall de múltiplos modelos.

    Parameters
    ----------
    pr_data : dict
        Ex.: {"XGBoost": (recall_array, precision_array, pr_auc_value), ...}
    """
    fig = go.Figure()
    for model_name, (recall, precision, pr_auc_value) in pr_data.items():
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{model_name} (PR-AUC = {pr_auc_value:.3f})",
            )
        )
    fig.update_layout(
        title="Curva Precision-Recall — Comparação de Modelos",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de barras agrupado comparando as métricas de cada modelo.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Índice = nome do modelo; colunas = métricas (ex.: saída de
        `src.evaluation.metrics.metrics_to_dataframe`).
    """
    df_melt = metrics_df.reset_index().melt(id_vars="model", var_name="métrica", value_name="valor")
    fig = px.bar(
        df_melt,
        x="métrica",
        y="valor",
        color="model",
        barmode="group",
        title="Comparação de Métricas entre Modelos",
        labels={"valor": "Valor da Métrica", "métrica": "Métrica", "model": "Modelo"},
    )
    return fig


def plot_numeric_association_ranking(ranking_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Ranking de variáveis numéricas por |rank-biserial|, sem misturar escalas."""
    required = {"variavel", "rank_biserial", "magnitude_efeito", "p_value_fdr", "n"}
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = ranking_df.dropna(subset=["magnitude_efeito"]).head(top_n).copy()
    if data.empty:
        raise ValueError("Não há associações numéricas válidas para visualizar.")
    data = data.sort_values("magnitude_efeito", ascending=True)
    fig = px.bar(
        data,
        x="magnitude_efeito",
        y="variavel",
        orientation="h",
        custom_data=["rank_biserial", "p_value_fdr", "n"],
        title="Quais variáveis numéricas mais diferenciam Churn e Permaneceu?",
        labels={"magnitude_efeito": "|Efeito rank-biserial|", "variavel": "Variável"},
    )
    fig.update_traces(
        hovertemplate=(
            "Variável: %{y}<br>|Efeito|: %{x:.3f}<br>"
            "Efeito com direção: %{customdata[0]:.3f}<br>"
            "p ajustado (FDR): %{customdata[1]:.4g}<br>"
            "n: %{customdata[2]}<extra></extra>"
        )
    )
    fig.update_layout(yaxis_title="", xaxis_title="Magnitude do efeito (|rank-biserial|)")
    return fig


def plot_categorical_association_ranking(ranking_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Ranking de variáveis categóricas por Cramér's V."""
    required = {"variavel", "cramers_v", "p_value_fdr", "n", "n_categorias"}
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = ranking_df.dropna(subset=["cramers_v"]).head(top_n).copy()
    if data.empty:
        raise ValueError("Não há associações categóricas válidas para visualizar.")
    data = data.sort_values("cramers_v", ascending=True)
    fig = px.bar(
        data,
        x="cramers_v",
        y="variavel",
        orientation="h",
        custom_data=["p_value_fdr", "n", "n_categorias"],
        title="Quais variáveis categóricas apresentam maior associação com churn?",
        labels={"cramers_v": "Cramér's V", "variavel": "Variável"},
    )
    fig.update_traces(
        hovertemplate=(
            "Variável: %{y}<br>Cramér's V: %{x:.3f}<br>"
            "p ajustado (FDR): %{customdata[0]:.4g}<br>"
            "n: %{customdata[1]}<br>Categorias: %{customdata[2]}<extra></extra>"
        )
    )
    fig.update_layout(yaxis_title="", xaxis_title="Cramér's V")
    return fig


def plot_churn_rate_deviation(
    segment_df: pd.DataFrame,
    top_n: int = 15,
    only_eligible: bool = True,
) -> go.Figure:
    """Mostra segmentos com maior desvio da taxa geral de churn em pontos percentuais."""
    required = {
        "variavel", "categoria", "total_clientes", "churners", "taxa_churn_pct",
        "taxa_geral_pct", "desvio_pp", "elegivel_amostra"
    }
    missing = required - set(segment_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = segment_df.copy()
    if only_eligible:
        data = data[data["elegivel_amostra"]]
    data = data.nlargest(top_n, "magnitude_desvio_pp").copy()
    if data.empty:
        raise ValueError("Não há segmentos com amostra suficiente para visualizar.")
    data["segmento"] = data["variavel"].astype(str) + " = " + data["categoria"].astype(str)
    data = data.sort_values("desvio_pp")
    fig = px.bar(
        data,
        x="desvio_pp",
        y="segmento",
        orientation="h",
        custom_data=["total_clientes", "churners", "taxa_churn_pct", "taxa_geral_pct"],
        title="Quais segmentos mais se desviam da taxa média de churn?",
        labels={"desvio_pp": "Diferença para a taxa geral (p.p.)", "segmento": "Segmento"},
    )
    fig.add_vline(x=0, line_dash="dash")
    fig.update_traces(
        hovertemplate=(
            "%{y}<br>Diferença: %{x:+.2f} p.p.<br>"
            "Clientes: %{customdata[0]}<br>Churners: %{customdata[1]}<br>"
            "Taxa do segmento: %{customdata[2]:.2f}%<br>"
            "Taxa geral: %{customdata[3]:.2f}%<extra></extra>"
        )
    )
    fig.update_layout(yaxis_title="", xaxis_title="Diferença para a taxa geral de churn (p.p.)")
    return fig


def plot_quantile_churn(summary_df: pd.DataFrame, variable_name: str) -> go.Figure:
    """Taxa de churn ao longo dos quantis/decis de uma variável numérica."""
    required = {"ordem", "faixa_label", "total_clientes", "churners", "taxa_churn_pct"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = summary_df.copy().sort_values("ordem")
    fig = px.line(
        data,
        x="ordem",
        y="taxa_churn_pct",
        markers=True,
        custom_data=["faixa_label", "total_clientes", "churners"],
        title=f"Como a taxa de churn varia ao longo da distribuição de {variable_name}?",
        labels={"ordem": "Quantil (ordem crescente)", "taxa_churn_pct": "Taxa de churn (%)"},
    )
    fig.update_traces(
        hovertemplate=(
            "Quantil: %{x}<br>Faixa: %{customdata[0]}<br>"
            "Clientes: %{customdata[1]}<br>Churners: %{customdata[2]}<br>"
            "Taxa de churn: %{y:.2f}%<extra></extra>"
        )
    )
    return fig


def plot_combination_heatmap(
    combination_df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> go.Figure:
    """Heatmap de taxa de churn para combinação de duas variáveis categóricas."""
    required = {col_a, col_b, "taxa_churn_pct", "total_clientes", "churners", "amostra_reduzida"}
    missing = required - set(combination_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = combination_df.copy()
    pivot_rate = data.pivot(index=col_a, columns=col_b, values="taxa_churn_pct")
    pivot_n = data.pivot(index=col_a, columns=col_b, values="total_clientes").reindex(
        index=pivot_rate.index, columns=pivot_rate.columns
    )
    pivot_churn = data.pivot(index=col_a, columns=col_b, values="churners").reindex(
        index=pivot_rate.index, columns=pivot_rate.columns
    )
    pivot_small = data.pivot(index=col_a, columns=col_b, values="amostra_reduzida").reindex(
        index=pivot_rate.index, columns=pivot_rate.columns
    )
    custom = np.stack(
        [pivot_n.to_numpy(), pivot_churn.to_numpy(), pivot_small.to_numpy()], axis=-1
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_rate.to_numpy(),
            x=[str(v) for v in pivot_rate.columns],
            y=[str(v) for v in pivot_rate.index],
            customdata=custom,
            colorbar=dict(title="Churn (%)"),
            hovertemplate=(
                f"{col_a}: %{{y}}<br>{col_b}: %{{x}}<br>"
                "Taxa de churn: %{z:.2f}%<br>Clientes: %{customdata[0]}<br>"
                "Churners: %{customdata[1]}<br>"
                "Amostra reduzida: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Taxa de churn: {col_a} × {col_b}",
        xaxis_title=col_b,
        yaxis_title=col_a,
    )
    return fig


def plot_risk_value_scatter(
    scored_df: pd.DataFrame,
    value_col: str,
    risk_col: str = "probabilidade_churn",
    customer_id_col: str | None = "customer_id",
) -> go.Figure:
    """Dispersão valor do cliente × probabilidade prevista de churn."""
    required = {value_col, risk_col}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    hover = [c for c in [customer_id_col, "churn"] if c and c in scored_df.columns]
    fig = px.scatter(
        scored_df,
        x=value_col,
        y=risk_col,
        hover_data=hover,
        title="Quais clientes combinam maior risco previsto e maior valor?",
        labels={value_col: value_col, risk_col: "Probabilidade prevista de churn"},
        opacity=0.65,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_error_profile(error_df: pd.DataFrame, variable: str) -> go.Figure:
    """Compara distribuição de variável numérica entre TP/FP/FN/TN."""
    required = {"tipo_resultado", variable}
    missing = required - set(error_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = px.box(
        error_df,
        x="tipo_resultado",
        y=variable,
        points=False,
        title=f"Onde o modelo erra? Perfil de {variable} por tipo de resultado",
        labels={"tipo_resultado": "Resultado da classificação", variable: variable},
    )
    return fig


def plot_split_distribution(distribution_df: pd.DataFrame) -> go.Figure:
    """Compara a distribuição percentual das classes em dataset/treino/teste."""
    required = {"conjunto", "classe", "quantidade", "percentual"}
    missing = required - set(distribution_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = distribution_df.copy()
    data["classe_label"] = data["classe"].map({0: "Permaneceu", 1: "Churn"}).fillna(
        data["classe"].astype(str)
    )
    fig = px.bar(
        data,
        x="conjunto",
        y="percentual",
        color="classe_label",
        barmode="group",
        text=data["percentual"].map(lambda v: f"{v:.2f}%"),
        custom_data=["quantidade"],
        title="A estratificação preservou a proporção das classes?",
        labels={
            "conjunto": "Conjunto",
            "percentual": "Percentual da classe (%)",
            "classe_label": "Classe",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "Conjunto: %{x}<br>Percentual: %{y:.2f}%<br>"
            "Quantidade: %{customdata[0]}<extra></extra>"
        )
    )
    return fig


def plot_cv_pr_auc(cv_results: pd.DataFrame) -> go.Figure:
    """PR-AUC médio da cross-validation com barra de erro por candidato."""
    required = {"model", "strategy", "cv_pr_auc_mean", "cv_pr_auc_std"}
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = cv_results.copy()
    data["candidato"] = data["model"] + " · " + data["strategy"]
    data = data.sort_values("cv_pr_auc_mean", ascending=True)
    fig = px.scatter(
        data,
        x="cv_pr_auc_mean",
        y="candidato",
        error_x="cv_pr_auc_std",
        custom_data=["cv_recall_mean", "cv_precision_mean", "cv_f1_mean", "cv_mcc_mean"],
        title="Cross-validation: PR-AUC médio ± desvio padrão",
        labels={"cv_pr_auc_mean": "PR-AUC médio", "candidato": "Modelo · estratégia"},
    )
    fig.update_traces(
        marker=dict(size=11),
        hovertemplate=(
            "Candidato: %{y}<br>PR-AUC médio: %{x:.4f}<br>"
            "Recall médio: %{customdata[0]:.4f}<br>"
            "Precision média: %{customdata[1]:.4f}<br>"
            "F1 médio: %{customdata[2]:.4f}<br>"
            "MCC médio: %{customdata[3]:.4f}<extra></extra>"
        ),
    )
    return fig


def plot_imbalance_strategy_comparison(cv_results: pd.DataFrame) -> go.Figure:
    """Compara PR-AUC por modelo e estratégia de desbalanceamento."""
    required = {"model", "strategy", "cv_pr_auc_mean", "cv_pr_auc_std"}
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = px.bar(
        cv_results,
        x="model",
        y="cv_pr_auc_mean",
        color="strategy",
        barmode="group",
        error_y="cv_pr_auc_std",
        custom_data=["cv_recall_mean", "cv_precision_mean", "cv_f1_mean", "cv_mcc_mean"],
        title="Qual estratégia de tratamento do desbalanceamento funciona melhor?",
        labels={
            "model": "Modelo",
            "cv_pr_auc_mean": "PR-AUC médio (CV)",
            "strategy": "Estratégia",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "Modelo: %{x}<br>PR-AUC médio: %{y:.4f}<br>"
            "Recall: %{customdata[0]:.4f}<br>Precision: %{customdata[1]:.4f}<br>"
            "F1: %{customdata[2]:.4f}<br>MCC: %{customdata[3]:.4f}<extra></extra>"
        )
    )
    return fig


def plot_stability(cv_results: pd.DataFrame) -> go.Figure:
    """Mostra desempenho médio versus variabilidade entre folds."""
    required = {"model", "strategy", "cv_pr_auc_mean", "cv_pr_auc_std"}
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = px.scatter(
        cv_results,
        x="cv_pr_auc_mean",
        y="cv_pr_auc_std",
        color="strategy",
        symbol="model",
        hover_data=["cv_recall_mean", "cv_precision_mean", "cv_mcc_mean"],
        title="Desempenho médio × estabilidade entre folds",
        labels={
            "cv_pr_auc_mean": "PR-AUC médio (maior é melhor)",
            "cv_pr_auc_std": "Desvio padrão do PR-AUC (menor é mais estável)",
            "strategy": "Estratégia",
            "model": "Modelo",
        },
    )
    return fig


def plot_train_cv_gap(cv_results: pd.DataFrame) -> go.Figure:
    """Compara PR-AUC de treino e validação cruzada para investigar overfitting."""
    required = {"model", "strategy", "train_pr_auc_mean", "cv_pr_auc_mean"}
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = cv_results.copy()
    data["candidato"] = data["model"] + " · " + data["strategy"]
    melted = data.melt(
        id_vars=["candidato"],
        value_vars=["train_pr_auc_mean", "cv_pr_auc_mean"],
        var_name="origem",
        value_name="pr_auc",
    )
    melted["origem"] = melted["origem"].map(
        {"train_pr_auc_mean": "Treino", "cv_pr_auc_mean": "Cross-validation"}
    )
    fig = px.bar(
        melted,
        x="candidato",
        y="pr_auc",
        color="origem",
        barmode="group",
        title="PR-AUC: treino × cross-validation",
        labels={"candidato": "Modelo · estratégia", "pr_auc": "PR-AUC", "origem": "Origem"},
    )
    return fig


def plot_calibration_curves(curve_df: pd.DataFrame) -> go.Figure:
    """Reliability diagram para probabilidades OOF."""
    required = {"method", "mean_predicted_probability", "observed_fraction_positive"}
    missing = required - set(curve_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Calibração perfeita",
            line=dict(dash="dash"),
        )
    )
    for method, group in curve_df.groupby("method"):
        fig.add_trace(
            go.Scatter(
                x=group["mean_predicted_probability"],
                y=group["observed_fraction_positive"],
                mode="lines+markers",
                name=str(method),
            )
        )
    fig.update_layout(
        title="As probabilidades previstas correspondem ao risco observado?",
        xaxis_title="Probabilidade média prevista",
        yaxis_title="Frequência observada de churn",
    )
    return fig


def plot_threshold_tradeoff(threshold_df: pd.DataFrame) -> go.Figure:
    """Precision, Recall e F1 ao longo do threshold."""
    required = {"threshold", "precision", "recall", "f1"}
    missing = required - set(threshold_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = go.Figure()
    for metric, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        fig.add_trace(
            go.Scatter(
                x=threshold_df["threshold"],
                y=threshold_df[metric],
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title="Como o threshold altera Precision, Recall e F1?",
        xaxis_title="Threshold",
        yaxis_title="Métrica",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def plot_lift_top_k(top_k_df: pd.DataFrame) -> go.Figure:
    """Lift@K calculado sobre probabilidades reais do conjunto final de teste."""
    required = {"top_percent", "lift_at_k", "recall_at_k", "precision_at_k"}
    missing = required - set(top_k_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    fig = px.bar(
        top_k_df,
        x="top_percent",
        y="lift_at_k",
        custom_data=["recall_at_k", "precision_at_k", "n_selected", "positives_captured"],
        title="Lift nos clientes de maior risco previsto",
        labels={"top_percent": "Topo da base (%)", "lift_at_k": "Lift@K"},
    )
    fig.update_traces(
        hovertemplate=(
            "Top %{x:.0f}%<br>Lift: %{y:.3f}<br>Recall@K: %{customdata[0]:.3f}<br>"
            "Precision@K: %{customdata[1]:.3f}<br>Selecionados: %{customdata[2]}<br>"
            "Churners capturados: %{customdata[3]}<extra></extra>"
        )
    )
    return fig


def plot_risk_deciles(decile_df: pd.DataFrame) -> go.Figure:
    """Taxa de churn observada por decil de risco previsto."""
    required = {"risk_decile", "churn_rate", "n", "churners"}
    missing = required - set(decile_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para o gráfico: {sorted(missing)}")
    data = decile_df.copy()
    data["churn_rate_pct"] = data["churn_rate"] * 100
    fig = px.bar(
        data,
        x="risk_decile",
        y="churn_rate_pct",
        custom_data=["n", "churners", "min_probability", "max_probability"],
        title="Taxa de churn por decil de risco previsto",
        labels={"risk_decile": "Decil de risco (1 = maior risco)", "churn_rate_pct": "Taxa de churn (%)"},
    )
    fig.update_traces(
        hovertemplate=(
            "Decil: %{x}<br>Taxa de churn: %{y:.2f}%<br>Clientes: %{customdata[0]}<br>"
            "Churners: %{customdata[1]}<br>Probabilidade mín.: %{customdata[2]:.3f}<br>"
            "Probabilidade máx.: %{customdata[3]:.3f}<extra></extra>"
        )
    )
    return fig
