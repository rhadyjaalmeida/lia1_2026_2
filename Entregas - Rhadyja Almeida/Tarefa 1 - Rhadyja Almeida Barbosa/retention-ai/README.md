# RetentionAI — Customer Churn Prediction

Projeto acadêmico de Data Science e Machine Learning para investigação e previsão de churn de clientes, com foco em **rigor metodológico, explicabilidade e rastreabilidade dos resultados**.

## Princípio central

O projeto segue uma regra absoluta:

> **Nenhum dado, métrica, insight, modelo vencedor, threshold, score ou recomendação é inventado ou preenchido manualmente.**

Quando um resultado ainda não foi calculado, notebook e dashboard devem tratá-lo como **resultado pendente de execução**.

## Dataset

- **Fonte:** Kaggle
- **Dataset:** Sales and Marketing Dataset
- **Página:** https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset
- **Target esperado:** `churn`

A estrutura real do arquivo é auditada na execução. Quantidades de linhas, colunas, missing, proporções e demais números não são assumidos no README.

## O que mudou na validação metodológica

A versão atual não escolhe o melhor modelo olhando o conjunto final de teste.

O fluxo passou a ser:

```text
Dataset
  ↓
Auditoria e limpeza estrutural
  ↓
Feature Engineering conservador
  ↓
Train/Test split estratificado
  ↓
TEST SET ISOLADO
  ↓
Leakage Audit
  ↓
Dummy baseline
  ↓
Stratified K-Fold Cross-Validation
  ↓
Comparação de modelos
  ×
Comparação de desbalanceamento
  ↓
Hyperparameter tuning no treino
  ↓
Análise de estabilidade / overfitting
  ↓
Calibração com previsões fora da amostra
  ↓
Threshold usando previsões out-of-fold
  ↓
Modelo final
  ↓
Avaliação única no Test Set
  ↓
Lift / Top-K / decis de risco
  ↓
SHAP / análise de erros
```

## Tratamento do desbalanceamento

O projeto **não assume que SMOTE é melhor** e não tenta tornar o test set artificialmente 50/50.

São comparadas, dentro da cross-validation:

- distribuição natural (`none`);
- pesos de classe (`class_weight`) ou `scale_pos_weight` para XGBoost;
- `SMOTENC` quando há variáveis numéricas e categóricas.

O `SMOTENC` é aplicado antes do one-hot final, preservando a natureza categórica durante o resampling.

## Modelos candidatos

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Nenhum modelo é declarado vencedor previamente.

## Métricas

A seleção não utiliza Accuracy como critério principal em um problema desbalanceado.

São calculadas, conforme a etapa:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC / Average Precision
- Balanced Accuracy
- MCC
- Brier Score
- matriz de confusão
- Recall@K
- Precision@K
- Lift@K

Cross-validation registra **média e desvio padrão**.

## Seleção de modelo

A pré-seleção usa uma regra de **1 erro-padrão** em PR-AUC para evitar declarar um grande vencedor quando as diferenças entre candidatos estão dentro da variabilidade observada.

Depois, candidatos promissores passam por `RandomizedSearchCV` usando apenas o conjunto de treino.

Os hiperparâmetros finais são obtidos programaticamente de `best_params_` e salvos nos artefatos de validação.

## Calibração

O projeto compara probabilidades:

- não calibradas;
- `sigmoid`;
- `isotonic`.

A comparação usa previsões fora da amostra dentro do conjunto de treino e inclui Brier Score.

## Threshold

O threshold 0,50 é apenas referência.

Sem custos empresariais reais parametrizados, a implementação seleciona de forma explícita o threshold que maximiza F1 nas previsões out-of-fold do treino. Se houver uma matriz real de custos ou uma meta de Recall, essa regra pode ser substituída sem consultar o test set.

## Data leakage

A versão atual também corrige um risco importante: **imputações estatísticas não são mais aprendidas sobre o dataset inteiro antes do split**.

- valores estruturalmente inválidos podem ser convertidos para missing;
- datas são convertidas de forma determinística;
- mediana/moda/scaling/encoding são aprendidos dentro do pipeline;
- identificadores, target e derivadas do target são excluídos;
- features temporais cujo instante de disponibilidade não está documentado são excluídas conservadoramente do modelo final por padrão.

A tabela completa é gerada em `reports/results/validation_leakage_audit.csv` após a execução.

## Descobertas e insights

A área de descoberta adicionada anteriormente continua presente.

Ela inclui, quando os dados permitem:

- ranking de associações numéricas por efeito rank-biserial;
- ranking categórico por Cramér's V;
- correção de múltiplos testes por Benjamini-Hochberg (FDR);
- desvios da taxa média de churn por segmento;
- decis/quantis;
- combinações de segmentos e heatmaps;
- descobertas emergentes;
- análise risco × valor;
- análise de True Positive, False Positive, False Negative e True Negative.

Os rankings numéricos e categóricos permanecem separados para não misturar métricas incompatíveis.

## Dashboard Streamlit

O `app.py` contém:

- ⭐ Principais descobertas
- 📊 Associações
- 🎯 Segmentos
- 🔬 Explorar Descobertas
- 🧪 Validação do Modelo

A página **Validação do Modelo** não executa números fictícios. Ela apenas lê artefatos reais produzidos pelo notebook. Quando eles ainda não existem, exibe resultado pendente.

Execute com:

```bash
streamlit run app.py
```

## Artefatos de validação

Após uma execução completa do notebook, a pasta `reports/results/` pode conter:

```text
validation_split_distribution.csv
validation_split_checks.json
validation_leakage_audit.csv
validation_dummy_baseline.csv
validation_cv_results.csv
validation_preselection.json
validation_tuning_results.csv
validation_model_selection.json
validation_selected_model_cv.csv
validation_calibration_results.csv
validation_calibration_curve.csv
validation_threshold_results.csv
validation_threshold_selection.json
validation_final_test_metrics.csv
validation_confusion_matrix.csv
validation_cv_vs_test.csv
validation_top_k.csv
validation_risk_deciles.csv
validation_error_counts.csv
validation_audit.json
shap_feature_importance.csv
```

Esses arquivos são a **fonte de verdade** utilizada pelo front-end para resultados preditivos.

## Resultados

**Resultado pendente de execução da nova metodologia.**

As métricas da versão anterior foram movidas para:

```text
reports/legacy_previous_run/
```

Elas foram preservadas somente para rastreabilidade e **não devem ser tratadas como resultados finais desta versão**.

## Estrutura

```text
retention-ai/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── legacy_previous_run/
│
├── notebooks/
│   └── 01_churn_analysis.ipynb
│
├── src/
│   ├── analysis/
│   │   ├── discovery.py
│   │   └── leakage.py
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── validation.py
│
├── models/
│
├── reports/
│   ├── figures/
│   ├── results/
│   └── legacy_previous_run/
│
└── docs/
    ├── REGRAS_DESCOBERTAS.md
    └── REGRAS_VALIDACAO.md
```

## Como executar

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Dataset

Coloque o CSV original em:

```text
data/raw/
```

Veja `data/README.md`.

### 4. Executar o notebook

Abra:

```text
notebooks/01_churn_analysis.ipynb
```

Use **Restart & Run All**.

O tuning e a calibração podem exigir mais processamento do que a versão anterior, pois executam múltiplos folds e configurações para evitar conclusões baseadas em uma única divisão.

### 5. Abrir o dashboard

Depois da execução do notebook:

```bash
streamlit run app.py
```

## Auditoria final

A execução gera `validation_audit.json` com verificações como:

- test set isolado;
- split estratificado;
- cross-validation concluída;
- estratégias de desbalanceamento comparadas;
- tuning concluído;
- leakage audit concluído;
- calibração avaliada;
- threshold selecionado sem test set;
- overfitting analisado;
- avaliação final concluída.

Os valores não são preenchidos manualmente.

## Limitações

- dataset público não garante representatividade de uma empresa específica;
- associação, SHAP e feature importance não demonstram causalidade;
- features temporais exigem definição clara do instante real de previsão antes de uso comercial;
- métricas acadêmicas não substituem validação externa;
- deployment real exigiria monitoramento de drift, segurança, privacidade, governança e validação contínua.

## Trabalho futuro

- API de inferência;
- integração com CRM/banco de dados;
- monitoramento de drift;
- políticas empresariais reais de custo para threshold;
- validação externa em dados temporais posteriores;
- agentes especializados sobre resultados autorizados e validados.
