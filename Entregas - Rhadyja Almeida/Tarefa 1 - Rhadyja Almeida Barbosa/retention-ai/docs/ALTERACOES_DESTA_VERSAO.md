# Alterações desta versão

Esta versão combina as alterações anteriores de **descobertas e insights** com a nova camada de **validação metodológica**.

## Mantido da versão anterior

- exploração de descobertas emergentes;
- rankings separados para numéricas e categóricas;
- FDR Benjamini-Hochberg;
- segmentos por desvio da taxa geral;
- decis/quantis;
- heatmaps de combinações;
- dashboard Streamlit de descobertas;
- risco × valor e análise dos erros do modelo;
- regra absoluta de não inventar dados/resultados.

## Adicionado agora

- split estratificado auditado;
- test set isolado até o final;
- imputação aprendida apenas dentro do pipeline;
- auditoria de leakage;
- DummyClassifier;
- StratifiedKFold 5 folds;
- comparação `none × class_weight × SMOTENC`;
- SMOTENC antes do one-hot final;
- PR-AUC média e desvio padrão;
- análise de estabilidade e gap treino/CV;
- pré-seleção por regra de 1 erro-padrão;
- RandomizedSearchCV;
- parâmetros finais vindos de `best_params_`;
- calibração OOF (`uncalibrated`, `sigmoid`, `isotonic`);
- Brier Score;
- threshold escolhido em previsões out-of-fold;
- avaliação final única no test set;
- CV × test final;
- Lift@K, Recall@K, Precision@K;
- decis de risco;
- painel `🧪 Validação do Modelo` no Streamlit;
- `validation_audit.json`;
- resultados anteriores movidos para `reports/legacy_previous_run/` e removidos da fonte atual de verdade.

## Correções de integridade

- removidas conclusões numéricas hardcoded do README;
- outputs antigos do notebook foram limpos;
- `coupon_code` ausente não é mais interpretado automaticamente como “não usou cupom”;
- `coupon_code_reported` indica somente se o código está informado;
- features temporais de disponibilidade incerta são excluídas conservadoramente da modelagem por padrão;
- o processed CSV preserva missing para que as imputações sejam aprendidas dentro do pipeline.
