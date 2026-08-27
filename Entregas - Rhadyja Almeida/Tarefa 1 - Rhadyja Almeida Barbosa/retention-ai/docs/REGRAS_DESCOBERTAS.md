# ATUALIZAÇÃO OBRIGATÓRIA — GRÁFICOS DE DESCOBERTA E INSIGHTS RELEVANTES

Além de todos os gráficos, análises, perguntas e visualizações já definidos anteriormente, criar uma seção específica chamada:

# 🔎 DESCOBERTAS E INSIGHTS RELEVANTES DOS DADOS

O objetivo desta seção é permitir que a análise descubra **padrões relevantes que não necessariamente foram previstos inicialmente nas perguntas de pesquisa**.

Não limitar a EDA apenas às hipóteses previamente definidas.

Depois da auditoria, limpeza e compreensão real do dataset, investigar sistematicamente os dados para descobrir:

- relações inesperadas;
- segmentos com comportamento diferente;
- concentrações de churn;
- combinações de características relevantes;
- distribuições incomuns;
- diferenças importantes entre churn e não churn;
- grupos aparentemente mais vulneráveis;
- variáveis com forte poder discriminativo;
- padrões temporais;
- mudanças comportamentais;
- possíveis interações entre variáveis;
- anomalias;
- grupos pequenos com comportamento muito diferente da média;
- fatores que mereçam investigação adicional.

## REGRA FUNDAMENTAL

**Não procurar apenas resultados que confirmem as hipóteses iniciais.**

A análise deve estar aberta a encontrar resultados:

- esperados;
- inesperados;
- contrários às hipóteses;
- inconclusivos;
- estatisticamente irrelevantes.

Se nenhuma descoberta importante existir em determinada análise, não criar artificialmente um insight.

Informar claramente:

**“Não foi identificado um padrão suficientemente relevante nesta análise.”**

---

# 1. SISTEMA DE DESCOBERTA VISUAL

Depois da EDA principal, executar uma etapa exploratória adicional para identificar automaticamente relações potencialmente relevantes.

A lógica deve ser:

```text
Auditar variáveis disponíveis
↓
Selecionar combinações metodologicamente válidas
↓
Calcular estatísticas reais
↓
Identificar diferenças ou associações relevantes
↓
Ordenar descobertas
↓
Criar os gráficos mais informativos
↓
Validar estatisticamente quando apropriado
↓
Interpretar
↓
Gerar insight somente se houver evidência

```

Não criar dezenas de gráficos aleatórios.

A análise deve selecionar visualizações que apresentem **potencial real de descoberta**.

---

# 2. GRÁFICOS DE DESCOBERTA AUTOMÁTICA

Criar, quando os dados permitirem, gráficos adicionais para investigar relações que possam ter passado despercebidas.

## 2.1 Ranking das variáveis mais associadas ao churn

Criar uma visualização chamada:

# 📊 “Quais variáveis mais diferenciam clientes que permanecem e clientes que abandonam?”

Construir um ranking utilizando medidas apropriadas ao tipo de variável.

Dependendo dos dados, considerar:

### Variáveis numéricas

- diferença de média;
- diferença de mediana;
- tamanho de efeito;
- Mann–Whitney;
- teste t quando metodologicamente válido;
- correlação apropriada;
- mutual information quando justificável.

### Variáveis categóricas

- taxa de churn por categoria;
- diferença absoluta em relação à taxa geral;
- qui-quadrado;
- Cramér's V;
- mutual information quando apropriado.

Não misturar métricas incompatíveis como se representassem exatamente a mesma coisa.

Explicar claramente qual medida está sendo utilizada.

---

# 3. GRÁFICO DE DESVIO DA TAXA MÉDIA DE CHURN

Criar uma visualização para responder:

### ❓ Pergunta

**“Quais segmentos apresentam taxa de churn mais diferente da média da base?”**

Para cada variável categórica relevante:

1. calcular a taxa geral de churn;
2. calcular a taxa de churn de cada categoria;
3. calcular:

```python
desvio_churn = taxa_churn_segmento - taxa_churn_geral

```

4. ordenar os segmentos;
5. destacar os maiores desvios positivos e negativos.

Mostrar também obrigatoriamente:

- tamanho do segmento;
- número de churners;
- taxa de churn;
- taxa média da base;
- diferença em pontos percentuais.

Não destacar um segmento pequeno sem alertar sobre seu tamanho amostral.

---

# 4. TOP SEGMENTOS DE MAIOR CHURN

Criar um gráfico chamado:

# 🎯 “Onde o churn está mais concentrado?”

Analisar categorias disponíveis como, quando existirem:

- tipo de assinatura;
- canal de aquisição;
- faixa de satisfação;
- faixa de frequência de compras;
- segmentos de gastos;
- nível de engajamento;
- utilização de suporte;
- região;
- dispositivo;
- método de pagamento;
- outras categorias existentes no dataset.

Ordenar pela **taxa proporcional de churn**, e não apenas pela quantidade absoluta.

Sempre mostrar simultaneamente:

- tamanho do grupo;
- quantidade de churners;
- taxa de churn.

---

# 5. DESCOBERTA DE COMBINAÇÕES DE SEGMENTOS

Investigar se determinadas combinações de características apresentam comportamento especialmente diferente.

Exemplos conceituais, somente quando as respectivas variáveis existirem:

```text
Assinatura × Satisfação
Canal × Satisfação
Suporte × Satisfação
Frequência × Engajamento
Assinatura × Frequência
Gasto × Engajamento
Tempo como cliente × Satisfação

```

Não assumir previamente que essas combinações serão relevantes.

Primeiro calcular.

Criar, quando apropriado:

- heatmaps;
- grouped bars;
- stacked bars;
- small multiples;
- gráficos interativos.

Pergunta central:

### ❓

**“Existem combinações de características em que o churn se torna especialmente elevado ou reduzido?”**

---

# 6. HEATMAP DE TAXA DE CHURN

Quando duas variáveis categóricas ou discretizadas forem adequadas, produzir:

# 🔥 Heatmap de taxa de churn por combinação de características

Cada célula deverá representar:

```text
número de churners da combinação
÷
total de clientes da combinação

```

Mostrar no hover:

- categorias;
- número de clientes;
- churners;
- taxa de churn.

Se determinada célula possuir poucas observações, indicar:

**“Amostra reduzida — interpretar com cautela.”**

Nunca tratar uma célula pequena com taxa extrema como evidência forte sem considerar o número de observações.

---

# 7. ANÁLISE POR FAIXAS

Para algumas variáveis numéricas relevantes, não analisar apenas médias.

Criar faixas metodologicamente justificadas utilizando, quando apropriado:

- quantis;
- decis;
- intervalos naturais;
- faixas definidas pelo domínio.

Investigar:

```text
Variável
↓
faixas
↓
número de clientes
↓
taxa de churn

```

Exemplos possíveis, somente se as variáveis existirem:

- satisfação;
- gasto total;
- frequência;
- Lifetime Value;
- número de tickets;
- tempo como cliente;
- visitas;
- duração de sessão;
- dias desde última compra.

Isso pode revelar relações não lineares que seriam escondidas por uma simples correlação.

---

# 8. GRÁFICOS DE DECIS

Quando houver quantidade suficiente de observações, dividir determinadas variáveis numéricas em decis.

Pergunta:

### ❓

**“Como a taxa de churn varia quando percorremos a distribuição dessa característica?”**

Criar gráfico:

```text
Decil da variável × taxa de churn

```

Mostrar também:

- quantidade de clientes em cada decil;
- intervalo de valores do decil;
- churners;
- taxa de churn.

Esse gráfico deverá ajudar a detectar:

- relações lineares;
- relações não lineares;
- thresholds naturais;
- mudanças abruptas de comportamento.

---

# 9. DESCOBERTA DE RELAÇÕES NÃO LINEARES

Não depender apenas de correlações lineares.

Se os dados permitirem, utilizar visualizações como:

- LOWESS;
- bins;
- decis;
- medianas móveis;
- partial dependence depois do modelo;
- SHAP dependence plots.

Pergunta:

### ❓

**“Existe alguma relação não linear entre essa variável e o risco de churn?”**

Nunca interpretar automaticamente uma curva como causalidade.

---

# 10. ANÁLISE DE OUTLIERS COMO POSSÍVEIS DESCOBERTAS

Não tratar todo outlier apenas como problema de limpeza.

Investigar se os outliers representam:

- clientes de alto valor;
- clientes extremamente engajados;
- clientes extremamente insatisfeitos;
- comportamento incomum;
- grupos comerciais legítimos.

Criar visualizações quando houver relevância.

Pergunta:

### ❓

**“Os valores extremos representam erros ou um segmento empresarial importante?”**

---

# 11. DESCOBERTAS TEMPORAIS

Quando existirem datas adequadas, investigar automaticamente:

- churn ao longo do tempo;
- variação mensal;
- sazonalidade;
- mudanças de comportamento;
- evolução da satisfação;
- evolução de engajamento;
- frequência recente;
- compras recentes;
- comportamento antes do churn, se os dados permitirem análise metodologicamente correta.

Produzir gráficos interativos com Plotly sempre que isso melhorar a exploração.

Nunca inferir sequência temporal quando o dataset não possuir informações suficientes para isso.

---

# 12. GRÁFICO DE TENDÊNCIA DO CHURN

Se houver dimensão temporal válida, produzir:

# 📈 “Como o churn evolui ao longo do tempo?”

Mostrar:

- quantidade de clientes;
- quantidade de churners;
- taxa de churn;
- período.

Não confundir aumento do número absoluto de churners com aumento da taxa.

---

# 13. DESCOBERTA DE SEGMENTOS DE RISCO

Criar uma análise exploratória chamada:

# 🧩 “Quais perfis apresentam comportamento de churn diferente?”

Utilizar somente métodos justificáveis.

Poderão ser exploradas combinações derivadas das variáveis realmente disponíveis.

Por exemplo:

```text
Baixa satisfação + alta utilização de suporte

```

somente poderá ser apresentado como segmento importante se os cálculos mostrarem que isso realmente ocorre.

Para cada segmento identificado mostrar:

- definição;
- número de clientes;
- porcentagem da base;
- quantidade de churners;
- taxa de churn;
- comparação com a taxa geral;
- nível de evidência;
- limitações.

---

# 14. GRÁFICO DE RISCO × VALOR DO CLIENTE

Quando houver uma medida válida de valor do cliente e depois que o modelo existir, criar:

# 💰 “Quais clientes combinam alto risco e alto valor?”

Eixos conceituais:

```text
X = valor do cliente
Y = probabilidade prevista de churn

```

Essa visualização deve ajudar a identificar clientes potencialmente prioritários.

Utilizar somente:

- valores reais;
- probabilidades reais do modelo.

Não criar quadrantes arbitrários sem justificar os limites.

Se utilizar quadrantes como:

```text
Alto valor / Alto risco
Alto valor / Baixo risco
Baixo valor / Alto risco
Baixo valor / Baixo risco

```

explicar exatamente como os thresholds foram definidos.

---

# 15. GRÁFICO DE IMPORTÂNCIA × EVIDÊNCIA

Quando possível, comparar diferentes fontes de evidência para as variáveis.

Por exemplo:

```text
EDA
+
tamanho de efeito
+
Permutation Importance
+
SHAP

```

O objetivo é identificar fatores que aparecem de forma consistente em várias análises.

Não transformar isso automaticamente em causalidade.

Criar uma tabela ou visualização chamada:

# 🧠 “Quais fatores aparecem consistentemente como relevantes?”

---

# 16. CONSISTÊNCIA ENTRE EDA E MODELO

Depois do treinamento, comparar:

### Descobertas da EDA

versus

### Descobertas do modelo

Perguntar:

**“Os padrões observados na exploração também aparecem na interpretação do modelo?”**

Classificar os resultados, quando possível, como:

- evidência consistente;
- evidência parcial;
- resultado apenas descritivo;
- resultado apenas preditivo;
- resultado conflitante;
- evidência insuficiente.

Resultados conflitantes são importantes e **não devem ser escondidos**.

---

# 17. GRÁFICOS DE SHAP PARA NOVAS DESCOBERTAS

Não usar SHAP somente para produzir um ranking.

Explorar também, quando tecnicamente apropriado:

### SHAP Beeswarm

para descobrir direção e magnitude dos efeitos.

### SHAP Dependence Plot

para investigar relações específicas entre uma variável e a previsão.

### SHAP Interaction

quando suportado e computacionalmente viável, para investigar possíveis interações.

### SHAP Waterfall

para investigar casos individuais interessantes.

As descobertas de SHAP devem ser tratadas como explicações do comportamento do modelo, não como causalidade.

---

# 18. DESCOBERTAS A PARTIR DOS ERROS DO MODELO

Usar falsos positivos e falsos negativos como fonte de investigação.

Criar gráficos comparando:

```text
True Positives
False Positives
False Negatives
True Negatives

```

Investigar:

- satisfação;
- comportamento;
- frequência;
- gasto;
- suporte;
- engajamento;
- outras features relevantes.

Pergunta:

### ❓

**“Existe algum perfil de cliente que o modelo tenha dificuldade especial para identificar?”**

Essa análise pode revelar padrões importantes que uma métrica agregada não mostra.

---

# 19. GRÁFICO DE DESCOBERTAS MAIS IMPORTANTES

Ao final da EDA e das análises estatísticas, criar uma visualização executiva chamada:

# ⭐ “Principais descobertas encontradas nos dados”

Este gráfico **não deve ser previamente preenchido**.

Primeiro executar todas as análises.

Depois selecionar somente as descobertas realmente sustentadas.

Cada descoberta deverá possuir:

- variável ou segmento;
- tipo de evidência;
- magnitude observada;
- tamanho da amostra quando necessário;
- relação com churn;
- nível de cautela na interpretação.

Não limitar obrigatoriamente a 5 descobertas.

Mostrar apenas quantas forem realmente relevantes.

---

# 20. PAINEL DE DESCOBERTAS

No front-end do RetentionAI, criar uma seção:

# 🔎 Descobertas dos Dados

Ela deverá reunir dinamicamente os insights realmente encontrados.

Poderá apresentar cards como:

### Descoberta

Nome resumido do padrão identificado.

### Evidência

Resultado numérico realmente calculado.

### Visualização relacionada

Gráfico que originou ou sustenta a descoberta.

### Força da evidência

Somente se existir uma metodologia explícita para classificá-la.

### Interpretação

Explicação baseada no resultado real.

### Implicação empresarial

Somente quando houver sustentação suficiente.

---

# 21. BOTÃO/ÁREA “EXPLORAR DESCOBERTAS”

Caso o front-end seja desenvolvido em Streamlit, incluir uma página ou seção dedicada chamada:

# 🔬 Explorar Descobertas

Permitir ao usuário investigar interativamente variáveis reais disponíveis no dataset.

Quando tecnicamente viável, incluir filtros como:

- variável;
- segmento;
- churn;
- faixa de valores;
- categoria;
- período.

Os gráficos devem responder dinamicamente aos filtros.

Não escrever previamente as conclusões.

Sempre recalcular os resultados de acordo com os dados filtrados.

---

# 22. TOP INSIGHTS DINÂMICOS

Criar uma função responsável por reunir possíveis insights encontrados durante a execução.

Conceitualmente:

```python
insights_encontrados = []

```

Cada análise pode adicionar um insight apenas se os critérios definidos forem satisfeitos.

Exemplo estrutural:

```python
insights_encontrados.append({
    "pergunta": ...,
    "resultado": ...,
    "evidencia": ...,
    "grafico_origem": ...,
    "limitacao": ...
})

```

O conteúdo real deverá ser produzido dinamicamente.

Não utilizar textos hardcoded afirmando previamente resultados.

---

# 23. RANKING DE DESCOBERTAS

Se houver várias descobertas, criar uma forma de priorizá-las utilizando critérios transparentes.

Pode considerar, quando apropriado:

- magnitude da diferença;
- tamanho de efeito;
- robustez estatística;
- tamanho da amostra;
- importância preditiva;
- estabilidade;
- relevância empresarial.

Não criar um `score de relevância` arbitrário.

Se for utilizado um score composto, mostrar explicitamente:

- fórmula;
- componentes;
- pesos;
- justificativa.

---

# 24. NÃO CONFUNDIR “INTERESSANTE” COM “SIGNIFICATIVO”

Uma descoberta visualmente impressionante pode ser causada por:

- poucos registros;
- ruído;
- outliers;
- múltiplas comparações;
- categorias raras;
- acaso.

Por isso, antes de destacar uma descoberta importante:

1. verificar tamanho da amostra;
2. verificar qualidade dos dados;
3. calcular magnitude;
4. verificar significância quando apropriado;
5. verificar estabilidade;
6. comparar com o restante da base;
7. avaliar se existe explicação alternativa.

---

# 25. CONTROLE DE MÚLTIPLAS DESCOBERTAS

Como muitas relações poderão ser investigadas, evitar tratar todo p-value menor que um limite como uma grande descoberta.

Quando houver grande quantidade de testes estatísticos:

- reconhecer o problema de múltiplas comparações;
- considerar correção apropriada quando metodologicamente necessária;
- não utilizar somente p-value para selecionar insights;
- considerar tamanho de efeito e relevância prática.

---

# 26. FORMATO OBRIGATÓRIO DOS GRÁFICOS DE DESCOBERTA

Para cada descoberta importante, utilizar:

### ❓ Pergunta

O que estamos investigando?

### 🔬 Método

Como o resultado foi calculado?

### 📊 Gráfico

Visualização construída exclusivamente com dados reais.

### 📌 O que os dados mostram

Resultado objetivo encontrado.

### 💡 Insight

Interpretação sustentada pela evidência.

### 🏢 Impacto empresarial

Por que essa descoberta pode ser relevante?

### ⚠️ Limitação

O que não podemos concluir?

---

# 27. GRÁFICOS INTERATIVOS PARA DESCOBERTAS

Sempre que a exploração pelo usuário puder revelar detalhes adicionais, utilizar Plotly.

Priorizar interatividade para:

- segmentos;
- rankings;
- séries temporais;
- scatterplots;
- decis;
- heatmaps;
- risco × valor;
- comparação de características;
- análise dos erros;
- distribuição das probabilidades;
- Lift;
- SHAP quando houver suporte apropriado.

Nos tooltips mostrar somente informações realmente calculadas.

---

# 28. DESCOBERTAS NEGATIVAS TAMBÉM SÃO RESULTADOS

Caso uma hipótese aparentemente importante não seja sustentada pelos dados, isso deve aparecer como descoberta.

Exemplo estrutural:

### ❓ Pergunta

A variável X está associada ao churn?

### 📌 Resultado

**Os dados analisados não forneceram evidência suficiente de uma associação relevante.**

Isso é um resultado válido.

Não esconder análises apenas porque não apresentaram a relação esperada.

---

# 29. DESCOBERTAS INESPERADAS

Se durante a análise surgir uma relação forte que não estava entre as perguntas iniciais:

1. verificar a qualidade dos dados;
2. verificar se existe leakage;
3. verificar tamanho da amostra;
4. reproduzir o cálculo;
5. investigar possíveis confundidores;
6. validar estatisticamente quando aplicável;
7. somente então apresentá-la como descoberta.

Marcar visualmente como:

# 💡 Descoberta emergente dos dados

Essa categoria deve significar:

**“resultado encontrado durante a exploração”**

e nunca:

**“resultado inventado para enriquecer o projeto”.**

---

# 30. STORYTELLING DAS DESCOBERTAS

As visualizações deverão construir uma narrativa como:

```text
Qual é o cenário geral?
↓
Onde o churn parece diferente?
↓
Quais segmentos se destacam?
↓
Quais comportamentos diferenciam os grupos?
↓
Existem combinações particularmente relevantes?
↓
Quais relações resistem à análise estatística?
↓
O modelo encontra padrões semelhantes?
↓
Quais clientes concentram maior risco?
↓
Por que o modelo atribui esse risco?
↓
Quais descobertas são realmente importantes?
↓
O que pode ser utilizado para apoiar decisões?

```

---

# 31. RESULTADO FINAL DA SEÇÃO DE DESCOBERTAS

Ao final, gerar automaticamente:

# 🏆 Principais Descobertas do Projeto

Para cada descoberta realmente confirmada, apresentar:

```text
Descoberta
↓
Evidência numérica
↓
Visualização de origem
↓
Método de validação
↓
Interpretação
↓
Limitação
↓
Possível implicação empresarial

```

Não definir previamente quantas descobertas deverão existir.

Se houver:

- 3 descobertas relevantes → apresentar 3;
- 7 descobertas relevantes → apresentar 7;
- nenhuma descoberta robusta → informar isso claramente.

**Nunca criar insights apenas para atingir uma quantidade mínima.**

---

# PRINCÍPIO FINAL PARA DESCOBERTAS

A análise não deve perguntar apenas:

**“Os dados confirmam minha hipótese?”**

Ela também deve perguntar:

**“O que existe nestes dados que ainda não percebemos?”**

Porém:

**descoberta ≠ especulação**

**correlação ≠ causalidade**

**diferença visual ≠ evidência suficiente**

**p-value ≠ relevância empresarial**

**importância no modelo ≠ causa do churn**

**segmento pequeno ≠ padrão confiável**

Toda descoberta importante deve nascer da sequência:

**dados reais → cálculo → visualização → validação → interpretação → insight**

e nunca:

**insight desejado → gráfico construído para justificá-lo**.