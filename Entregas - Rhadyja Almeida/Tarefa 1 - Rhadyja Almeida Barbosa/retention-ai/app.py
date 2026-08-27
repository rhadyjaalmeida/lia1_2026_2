"""RetentionAI — exploração de dados e painel de validação metodológica.

Nenhum KPI, percentual, métrica de modelo, ranking ou insight é hardcoded.
Resultados preditivos só aparecem quando os artefatos reais produzidos pelo
notebook validado existem em reports/results/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.analysis.discovery import (
    build_association_rankings,
    build_dynamic_discoveries,
    candidate_categorical_pairs,
    categorical_churn_summary,
    combination_churn_summary,
    numeric_quantile_churn_summary,
    top_categorical_segments,
)
from src.data.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from src.visualization.plots import (
    plot_categorical_association_ranking,
    plot_churn_distribution,
    plot_churn_rate_deviation,
    plot_combination_heatmap,
    plot_numeric_association_ranking,
    plot_quantile_churn,
    plot_cv_pr_auc,
    plot_imbalance_strategy_comparison,
    plot_stability,
    plot_calibration_curves,
    plot_threshold_tradeoff,
    plot_lift_top_k,
    plot_risk_deciles,
)

RAW_FILENAME = "Sales_-_Marketing_customer_dataset.csv"


def clean_for_exploration(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica somente limpeza estrutural; não aprende imputações globais."""
    data = df.copy()
    if "age" in data.columns:
        invalid_age = (data["age"] < 0) | (data["age"] > 120)
        data.loc[invalid_age, "age"] = np.nan

    for col in ["signup_date", "last_purchase_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    if "churn" in data.columns:
        data["churn_label"] = data["churn"].map({0: "Permaneceu", 1: "Churn"})
    return data


@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame | None, str]:
    processed = PROCESSED_DATA_DIR / "churn_clean.csv"
    raw = RAW_DATA_DIR / RAW_FILENAME

    if processed.exists():
        df = clean_for_exploration(pd.read_csv(processed))
        return df, f"Dataset processado: {processed.relative_to(PROJECT_ROOT)}"

    if raw.exists():
        return clean_for_exploration(pd.read_csv(raw)), (
            f"Dataset bruto com limpeza estrutural em memória: {raw.relative_to(PROJECT_ROOT)}"
        )

    return None, "Dado ainda não disponível. Coloque o CSV em data/raw/ ou execute o notebook."


def segment_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    blocked = {"churn", "churn_label", "customer_id", "signup_date", "last_purchase_date"}
    for col in df.columns:
        if col in blocked:
            continue
        nunique = df[col].nunique(dropna=True)
        is_category = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
            or (pd.api.types.is_numeric_dtype(df[col]) and 2 <= nunique <= 10)
        )
        if is_category and 2 <= nunique <= 30:
            cols.append(col)
    return cols


def read_csv_result(filename: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json_result(filename: str) -> dict | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pending(message: str = "Resultado pendente de execução. Execute o notebook validado.") -> None:
    st.info(message)


st.set_page_config(page_title="RetentionAI", page_icon="🔎", layout="wide")
st.title("🔎 RetentionAI")
st.caption(
    "Exploração, descobertas e validação metodológica. Nenhum resultado preditivo é inventado pelo front-end."
)

df, source_message = load_dataset()
st.info(source_message)

if df is None or "churn" not in df.columns:
    st.error("Esta informação não pode ser calculada com os dados disponíveis.")
    st.stop()

# KPIs exclusivamente derivados da base carregada.
total_clientes = int(len(df))
churners = int(df["churn"].sum())
taxa_churn = float(df["churn"].mean() * 100)

k1, k2, k3 = st.columns(3)
k1.metric("Clientes", f"{total_clientes:,}".replace(",", "."))
k2.metric("Clientes com churn", f"{churners:,}".replace(",", "."))
k3.metric("Taxa de churn", f"{taxa_churn:.2f}%")

with st.expander("Distribuição geral do churn", expanded=True):
    if "churn_label" in df.columns:
        st.plotly_chart(plot_churn_distribution(df, "churn_label"), use_container_width=True)
    else:
        st.warning("Dado ainda não disponível para esta visualização.")

exclude = ["customer_id", "signup_date", "last_purchase_date", "churn_label"]
try:
    numeric_rank, categorical_rank = build_association_rankings(
        df, target="churn", exclude_columns=exclude, max_categories=30
    )
except Exception as exc:
    st.error(f"Não foi possível calcular os rankings de descoberta: {exc}")
    st.stop()

seg_cols = segment_columns(df)
segments = top_categorical_segments(
    df,
    seg_cols,
    target="churn",
    min_group_size=30,
    min_group_fraction=0.01,
)

discoveries = build_dynamic_discoveries(
    numeric_rank,
    categorical_rank,
    segments,
    alpha=0.05,
    min_numeric_effect=0.10,
    min_categorical_effect=0.10,
    min_segment_delta_pp=2.0,
    max_items_per_type=5,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⭐ Principais descobertas",
        "📊 Associações",
        "🎯 Segmentos",
        "🔬 Explorar",
        "🧪 Validação do Modelo",
    ]
)

with tab1:
    st.subheader("Principais descobertas encontradas nos dados")
    if discoveries.empty:
        st.info("Não foi identificado um padrão suficientemente relevante nesta análise.")
    else:
        for i, row in discoveries.reset_index(drop=True).iterrows():
            with st.container(border=True):
                st.markdown(f"**{i + 1}. {row['resultado']}**")
                evidence = f"{row['metrica']}: {row['magnitude']:.3f} · n={int(row['n'])}"
                if pd.notna(row["p_value_fdr"]):
                    evidence += f" · p ajustado (FDR)={row['p_value_fdr']:.4g}"
                st.caption(evidence)
                st.caption(f"Limitação: {row['limitacao']}")

with tab2:
    st.subheader("Quais variáveis mais diferenciam Churn e Permaneceu?")
    st.markdown(
        "As escalas são mostradas separadamente: **rank-biserial** para variáveis numéricas "
        "e **Cramér's V** para categóricas."
    )
    if numeric_rank.empty:
        st.info("Dado ainda não disponível para o ranking numérico.")
    else:
        st.plotly_chart(plot_numeric_association_ranking(numeric_rank), use_container_width=True)
        st.dataframe(
            numeric_rank[["variavel", "n", "rank_biserial", "p_value", "p_value_fdr"]],
            use_container_width=True,
            hide_index=True,
        )

    if categorical_rank.empty:
        st.info("Dado ainda não disponível para o ranking categórico.")
    else:
        st.plotly_chart(
            plot_categorical_association_ranking(categorical_rank),
            use_container_width=True,
        )
        st.dataframe(
            categorical_rank[
                ["variavel", "n", "n_categorias", "cramers_v", "p_value", "p_value_fdr"]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader("Quais segmentos mais se desviam da taxa média de churn?")
    if segments.empty:
        st.info("Esta visualização não foi criada porque o dataset não possui segmentos adequados.")
    else:
        st.plotly_chart(plot_churn_rate_deviation(segments), use_container_width=True)
        st.dataframe(
            segments[
                [
                    "variavel", "categoria", "total_clientes", "churners",
                    "taxa_churn_pct", "taxa_geral_pct", "desvio_pp", "elegivel_amostra",
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.subheader("Explorar descobertas")
    mode = st.radio(
        "Tipo de exploração",
        ["Taxa por categoria", "Decis de variável numérica", "Combinação de segmentos"],
        horizontal=True,
    )

    if mode == "Taxa por categoria":
        if not seg_cols:
            st.info("Dado ainda não disponível para esta análise.")
        else:
            selected = st.selectbox("Variável", seg_cols)
            summary = categorical_churn_summary(
                df,
                selected,
                target="churn",
                min_group_size=max(30, int(np.ceil(len(df) * 0.01))),
            )
            st.dataframe(
                summary[
                    [
                        selected, "total_clientes", "churners", "taxa_churn_pct",
                        "taxa_geral_pct", "desvio_pp", "amostra_reduzida",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    elif mode == "Decis de variável numérica":
        num_options = [
            c for c in numeric_rank["variavel"].tolist()
            if c in df.columns and df[c].nunique(dropna=True) >= 10
        ]
        if not num_options:
            st.info("Esta visualização não foi criada porque não há variável adequada para decis.")
        else:
            selected = st.selectbox("Variável numérica", num_options)
            q_summary = numeric_quantile_churn_summary(df, selected, target="churn", q=10)
            st.plotly_chart(plot_quantile_churn(q_summary, selected), use_container_width=True)
            st.dataframe(
                q_summary[["ordem", "faixa_label", "total_clientes", "churners", "taxa_churn_pct"]],
                use_container_width=True,
                hide_index=True,
            )

    else:
        pairs = candidate_categorical_pairs(df, seg_cols, max_categories_each=10)
        if not pairs:
            st.info("Esta visualização não foi criada porque não há combinações adequadas.")
        else:
            pair_labels = {f"{a} × {b}": (a, b) for a, b in pairs}
            selected_label = st.selectbox("Combinação", list(pair_labels))
            col_a, col_b = pair_labels[selected_label]
            combo = combination_churn_summary(
                df,
                col_a,
                col_b,
                target="churn",
                min_group_size=max(30, int(np.ceil(len(df) * 0.005))),
            )
            st.plotly_chart(plot_combination_heatmap(combo, col_a, col_b), use_container_width=True)
            st.caption("Células com amostra reduzida devem ser interpretadas com cautela.")

with tab5:
    st.subheader("🧪 Validação do Modelo")
    st.markdown(
        "Esta área **não treina nem inventa resultados**. Ela lê apenas os artefatos produzidos "
        "pela execução validada do notebook. Se uma etapa ainda não foi executada, o status permanece pendente."
    )

    audit = read_json_result("validation_audit.json")
    selection = read_json_result("validation_model_selection.json")
    threshold_selection = read_json_result("validation_threshold_selection.json")
    cv_results = read_csv_result("validation_cv_results.csv")
    calibration_results = read_csv_result("validation_calibration_results.csv")
    calibration_curve = read_csv_result("validation_calibration_curve.csv")
    threshold_results = read_csv_result("validation_threshold_results.csv")
    final_metrics = read_csv_result("validation_final_test_metrics.csv")
    top_k = read_csv_result("validation_top_k.csv")
    risk_deciles = read_csv_result("validation_risk_deciles.csv")

    st.markdown("### Status de qualidade metodológica")
    if audit is None:
        pending()
    else:
        audit_rows = []
        for name, value in audit.items():
            audit_rows.append({"verificação": name, "status": "✅" if bool(value) else "⚠️"})
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    st.markdown("### Seleção do modelo")
    if selection is None:
        pending("Resultado pendente de execução: modelo e hiperparâmetros ainda não foram validados pelo novo pipeline.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Modelo selecionado", str(selection.get("model", "Dado ainda não disponível")))
        c2.metric("Estratégia", str(selection.get("strategy", "Dado ainda não disponível")))
        st.caption(str(selection.get("selection_rule", "")))
        st.json(selection.get("best_params", {}))

    st.markdown("### Cross-validation e desbalanceamento")
    if cv_results is None or cv_results.empty:
        pending()
    else:
        st.plotly_chart(plot_imbalance_strategy_comparison(cv_results), use_container_width=True)
        st.plotly_chart(plot_cv_pr_auc(cv_results), use_container_width=True)
        st.plotly_chart(plot_stability(cv_results), use_container_width=True)
        st.dataframe(cv_results, use_container_width=True, hide_index=True)

    st.markdown("### Calibração")
    if calibration_results is None or calibration_results.empty:
        pending()
    else:
        st.dataframe(calibration_results, use_container_width=True, hide_index=True)
        if calibration_curve is not None and not calibration_curve.empty:
            st.plotly_chart(plot_calibration_curves(calibration_curve), use_container_width=True)

    st.markdown("### Threshold")
    if threshold_results is None or threshold_results.empty:
        pending()
    else:
        if threshold_selection:
            st.write(threshold_selection)
        st.plotly_chart(plot_threshold_tradeoff(threshold_results), use_container_width=True)

    st.markdown("### Avaliação final no test set")
    if final_metrics is None or final_metrics.empty:
        pending("Resultado pendente de execução. O test set ainda não possui uma avaliação final válida nesta versão.")
    else:
        st.dataframe(final_metrics, use_container_width=True, hide_index=True)

    st.markdown("### Lift, Top-K e decis de risco")
    if top_k is None or top_k.empty:
        pending()
    else:
        st.plotly_chart(plot_lift_top_k(top_k), use_container_width=True)
        st.dataframe(top_k, use_container_width=True, hide_index=True)

    if risk_deciles is not None and not risk_deciles.empty:
        st.plotly_chart(plot_risk_deciles(risk_deciles), use_container_width=True)
