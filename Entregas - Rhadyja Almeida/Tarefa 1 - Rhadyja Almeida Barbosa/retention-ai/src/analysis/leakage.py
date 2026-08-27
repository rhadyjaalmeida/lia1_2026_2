"""Auditoria conservadora de risco de data leakage.

A auditoria não afirma conhecer o momento operacional real de cada variável.
Ela identifica riscos técnicos óbvios e marca variáveis temporais para validação
de domínio quando a disponibilidade no instante da previsão não está documentada.
"""

from __future__ import annotations

import pandas as pd


def build_leakage_audit(
    columns: list[str],
    target: str = "churn",
    identifier_columns: tuple[str, ...] = ("customer_id",),
    direct_target_derivatives: tuple[str, ...] = ("churn_label",),
    raw_date_columns: tuple[str, ...] = ("signup_date", "last_purchase_date"),
    temporal_derived_columns: tuple[str, ...] = ("tenure_days", "recency_days"),
) -> pd.DataFrame:
    """Cria tabela de auditoria com decisão conservadora por coluna."""
    rows = []
    for col in columns:
        if col == target:
            risk = "Crítico"
            reason = "É a própria variável-alvo."
            decision = "Excluir das features"
        elif col in direct_target_derivatives:
            risk = "Crítico"
            reason = "É derivada diretamente da variável-alvo."
            decision = "Excluir das features"
        elif col in identifier_columns:
            risk = "Alto"
            reason = "Identificador; pode favorecer memorização e não representa comportamento generalizável."
            decision = "Excluir das features"
        elif col in raw_date_columns:
            risk = "A validar"
            reason = (
                "A utilidade temporal depende de a informação existir no instante real da previsão; "
                "a documentação disponível não estabelece esse instante."
            )
            decision = "Não usar a data bruta; validar qualquer feature derivada"
        elif col in temporal_derived_columns:
            risk = "A validar"
            reason = (
                "Feature temporal depende da definição do instante de previsão e da disponibilidade "
                "de last_purchase_date. Sem essa definição, não é seguro assumir ausência de leakage."
            )
            decision = "Excluir do modelo final até validar temporalidade"
        else:
            risk = "Sem sinal automático"
            reason = (
                "Nenhum vazamento direto foi identificado por regras técnicas. Ainda requer validação "
                "de domínio sobre disponibilidade no momento da previsão."
            )
            decision = "Candidata, sujeita à validação de domínio"
        rows.append(
            {
                "feature": col,
                "risco_leakage": risk,
                "motivo": reason,
                "decisao": decision,
            }
        )
    return pd.DataFrame(rows)
