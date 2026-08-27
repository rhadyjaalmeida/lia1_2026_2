# Regras de Validação Metodológica — RetentionAI

## Regra absoluta

Nenhum modelo, parâmetro, estratégia de balanceamento, threshold ou calibração pode ser escolhido usando o conjunto final de teste.

## Fluxo obrigatório

1. auditar o dataset;
2. limpar apenas inconsistências estruturais;
3. preservar missing para imputação dentro do pipeline;
4. separar treino/teste com estratificação quando adequado;
5. isolar o test set;
6. executar auditoria de leakage;
7. criar DummyClassifier;
8. usar StratifiedKFold;
9. comparar sem balanceamento, pesos de classe e SMOTENC;
10. selecionar candidatos usando resultados de cross-validation;
11. executar RandomizedSearchCV apenas no treino;
12. analisar média, desvio e gap treino/CV;
13. avaliar calibração com previsões fora da amostra;
14. escolher threshold com previsões out-of-fold;
15. ajustar o modelo final no treino;
16. usar o test set uma única vez para avaliação final;
17. calcular Lift/Top-K somente depois da seleção final;
18. gerar SHAP e análise de erros sobre o modelo escolhido;
19. salvar todos os resultados em `reports/results/`;
20. refletir o estado real no dashboard.

## Desbalanceamento

O test set mantém a distribuição original. Não existe obrigação de transformar treino ou teste em 50/50.

A estratégia vencedora deve ser resultado da cross-validation, não uma preferência prévia.

### SMOTENC

Variáveis categóricas são ordinalmente codificadas antes do sampler; o SMOTENC recebe índices categóricos; somente depois do resampling é aplicado one-hot para os modelos. Isso evita interpolar diretamente vetores one-hot como variáveis contínuas.

## Métrica de seleção

PR-AUC/Average Precision é priorizada por ser adequada a classe positiva minoritária. Recall, Precision, F1, MCC, estabilidade e calibração também são considerados.

A pré-seleção implementa regra de 1 erro-padrão para evitar declarar diferenças pequenas como vitórias claras.

## Tuning

Hiperparâmetros hardcoded são apenas configurações iniciais. O vencedor deve vir de `best_params_`.

## Calibração

Comparar sem calibração, sigmoid e isotonic por Brier Score e PR-AUC usando previsões fora da amostra.

## Threshold

0,50 é referência. Sem custo empresarial real, a implementação usa o threshold de maior F1 nas previsões OOF do treino. Uma política real deve substituir essa regra quando existir.

## Test set

É proibido usar `X_test`/`y_test` para:

- selecionar modelo;
- selecionar estratégia de balanceamento;
- tuning;
- feature selection aprendida;
- calibração;
- threshold.

## Integridade

Se um artefato não foi calculado, mostrar:

**Resultado pendente de execução.**

Nunca preencher dashboard, README, tabela ou tooltip com resultado fictício.
