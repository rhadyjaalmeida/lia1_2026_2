# Dados do Projeto

## Fonte

Sales and Marketing Dataset — Kaggle:

https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset

Consulte a licença vigente na página oficial antes de redistribuir ou utilizar fora do contexto permitido.

## Estrutura

```text
data/
├── raw/                  # arquivo original obtido da fonte
├── processed/            # limpeza estrutural, sem imputações estatísticas globais
└── legacy_previous_run/  # cópia da versão processada pela metodologia anterior
```

## Regra anti-leakage

A versão atual **não grava em `processed/` medianas/modas aprendidas sobre o dataset inteiro** para depois treinar o modelo.

A limpeza estrutural pode:

- converter valores fisicamente inválidos para missing;
- converter datas com `errors="coerce"`;
- preservar valores ausentes.

Imputação estatística, scaling e encoding ficam dentro dos pipelines de Machine Learning e são aprendidos apenas nos folds de treinamento.

## Arquivo esperado

O notebook procura inicialmente:

```text
data/raw/Sales_-_Marketing_customer_dataset.csv
```

Se o arquivo obtido na fonte tiver outro nome, ajuste `RAW_FILENAME` no notebook.
