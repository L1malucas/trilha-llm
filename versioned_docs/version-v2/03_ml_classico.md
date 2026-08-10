---
id: 03_ml_classico
title: "Módulo 03 — Machine Learning Clássico"
sidebar_position: 3
---

# Módulo 03 — Machine Learning Clássico

> **Objetivo**: dominar os algoritmos pré-deep-learning. Não é nostalgia: muitos problemas reais ainda são melhor resolvidos com ML clássico, e os conceitos (overfitting, regularização, validação) são universais.
>
> **Pré-requisitos**: Módulos [01](01_matematica.md) e [02](02_programacao_ferramentas.md).
>
> **Tempo de referência**: 4–6 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar o trade-off bias-variância com um exemplo concreto, não só a definição.
- Escolher entre regressão linear, árvore, boosting e kNN para um problema tabular, justificando a escolha.
- Explicar a diferença estrutural entre bagging (Random Forest) e boosting (XGBoost) — não são a mesma ideia com nome diferente.
- Detectar data leakage num pipeline de ML observando o código, não só a definição do termo.
- Escolher a métrica certa para um problema desbalanceado, e explicar por que accuracy engana nesse caso.

---

## Por que isso importa

- Em problemas tabulares, **gradient boosting frequentemente bate redes neurais** (mesmo em 2025).
- Conceitos como bias-variância, regularização, validação cruzada são ferramentas mentais para *qualquer* modelo, incluindo LLMs.
- Você precisa saber quando *não* usar deep learning. Treinar uma rede para um problema que XGBoost resolve em 5 minutos é desperdício.

---

## 3.1 Fundamentos conceituais

### Tipos de aprendizado
- **Supervisionado**: classificação, regressão.
- **Não-supervisionado**: clustering, redução de dimensionalidade, detecção de anomalias.
- **Semi-supervisionado**: rótulos parciais.
- **Auto-supervisionado** (cross-link com módulo [04](04_ml_moderno.md)): rótulos derivados dos próprios dados.

### Conceitos transversais
- **Bias-variance tradeoff**.
- **Overfitting vs underfitting**.
- **Regularização**: L1 (Lasso), L2 (Ridge), Elastic Net.
- **Validação**: holdout, k-fold cross-validation, stratified k-fold, time series split.
- **Métricas**: accuracy, precision, recall, F1, AUC-ROC, PR-AUC, MSE, MAE, R², log-loss.
- **Curse of dimensionality**.
- **No Free Lunch Theorem**: nenhum algoritmo é ótimo para tudo.
- **Pipeline de ML**: ingestão → limpeza → features → modelo → avaliação → deploy.

> **Intuição — bias-variância**: imagine ajustar uma curva a pontos de dados espalhados. Um modelo com **alto bias** (ex.: reta numa relação claramente curva) é "teimoso demais" — ignora padrões reais nos dados, erra sistematicamente da mesma forma (underfitting). Um modelo com **alta variância** (ex.: uma curva que serpenteia por cada ponto exatamente) é "influenciável demais" — memoriza ruído específico daquele conjunto de treino, e uma leve mudança nos dados produziria uma curva bem diferente (overfitting). O ponto ideal não é zero bias e zero variância (impossível na prática) — é o equilíbrio que minimiza o erro total em dados *novos*, não vistos no treino. Regularização (L1/L2) empurra deliberadamente o modelo pra ter mais bias em troca de menos variância, quando o modelo está memorizando demais.
>
> **Aplicação real**: overfitting em LLMs se manifesta diferente (o modelo "decorando" trechos do corpus de treino em vez de generalizar), mas a lógica é idêntica — é por isso que técnicas de regularização deste módulo (weight decay, early stopping) reaparecem sem mudança conceitual no mod. [05](05_deep_learning.md#53-otimização-e-regularização-para-dl).
>
> **Checkpoint**: sem olhar o texto, dê um exemplo (pode ser hipotético) de um modelo com alto bias e outro com alta variância no mesmo problema. Depois, explique por que "zero erro no treino" não é uma meta desejável.

### Referências
- `Curso` **Stanford CS229** — vídeos + notas. https://cs229.stanford.edu/
- `Livro` **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman. https://hastie.su.domains/ElemStatLearn/
- `Livro` **An Introduction to Statistical Learning** (versão mais acessível do anterior). https://www.statlearning.com/
- `Livro` **Pattern Recognition and Machine Learning** — Bishop.

---

## 3.2 Algoritmos supervisionados

### Lineares
- **Regressão Linear** (OLS, Ridge, Lasso).
- **Regressão Logística** (binária e multiclasse via softmax).
- **Support Vector Machines (SVM)**: hard margin, soft margin, kernel trick (RBF, polinomial).

### Baseados em árvore
- **Decision Trees**: critérios de split (Gini, entropia, MSE), poda.
- **Random Forests**: bagging.
- **Gradient Boosting**: AdaBoost, GBM, **XGBoost**, **LightGBM**, **CatBoost**.

### Baseados em distância
- **k-Nearest Neighbors (kNN)**.

### Probabilísticos
- **Naive Bayes** (multinomial, Gaussian, Bernoulli).
- **Linear/Quadratic Discriminant Analysis (LDA/QDA)**.

> **Intuição por família**: modelos **lineares** traçam uma reta/plano/hiperplano separando os dados — rápidos, interpretáveis, mas limitados a relações (aproximadamente) lineares, a menos que você projete features não-lineares manualmente. **Árvores de decisão** são uma sequência de perguntas sim/não ("idade > 30? renda > X?") que particiona o espaço em regiões — capturam não-linearidade e interações naturalmente, mas uma única árvore profunda overfita fácil. **kNN** não "aprende" nada explicitamente — classifica um ponto novo olhando para seus vizinhos mais próximos no conjunto de treino; simples, mas caro em tempo de inferência e sofre com a maldição da dimensionalidade (em espaços de alta dimensão, "próximo" perde significado).
>
> **Intuição — bagging vs boosting** (a distinção mais confundida deste módulo): **Random Forest** (bagging) treina muitas árvores *independentes*, cada uma numa amostra aleatória dos dados, e faz a média das previsões — reduz variância porque erros aleatórios de árvores individuais se cancelam. **Gradient Boosting** (XGBoost, LightGBM) treina árvores *sequencialmente*, cada nova árvore focando em corrigir os erros que as anteriores ainda cometem (aprendendo a prever o *resíduo* do ensemble atual) — reduz bias porque o ensemble fica progressivamente melhor no que ainda erra. São estratégias opostas: bagging paraleliza e reduz variância; boosting é sequencial e reduz bias. É por isso que gradient boosting costuma vencer em benchmarks tabulares (Kaggle inclusive) — ele ataca diretamente o erro residual, enquanto Random Forest só reduz ruído de árvores já razoavelmente boas.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre bagging e boosting numa frase cada. Depois, explique por que kNN sofre mais com dimensionalidade alta do que uma árvore de decisão.

### Referências (papers fundadores)
- `Paper` **Random Forests** — Breiman (2001). https://link.springer.com/article/10.1023/A:1010933404324
- `Paper` **XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin (2016). https://arxiv.org/abs/1603.02754
- `Paper` **LightGBM** — Ke et al. (2017). https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- `Paper` **A Tutorial on Support Vector Machines for Pattern Recognition** — Burges (1998). https://www.cs.cmu.edu/~cga/ai-course/svmtutorial.pdf

---

## 3.3 Algoritmos não-supervisionados

### Clustering
- **k-Means** (e suas limitações: assume clusters esféricos, sensível a inicialização).
- **k-Means++** (inicialização melhorada).
- **Hierarchical Clustering**: agglomerative, divisive.
- **DBSCAN**: clustering baseado em densidade.
- **Gaussian Mixture Models (GMM)**: clustering probabilístico via EM.

### Redução de dimensionalidade
- **PCA** (já visto no módulo [01](01_matematica.md)).
- **t-SNE**: visualização, não para downstream tasks.
- **UMAP**: alternativa moderna ao t-SNE, mais rápida e preserva mais estrutura.

### Detecção de anomalias
- **Isolation Forest**.
- **One-Class SVM**.
- **Autoencoders** (cross-link com DL — módulo [05](05_deep_learning.md)).

> **Intuição — k-Means**: alterna entre dois passos até estabilizar — (1) atribuir cada ponto ao centróide mais próximo, (2) recalcular cada centróide como a média dos pontos atribuídos a ele. É um processo iterativo simples, mas assume que clusters são "bolhas" aproximadamente esféricas de tamanho parecido — falha visivelmente em clusters alongados, aninhados ou de densidade muito diferente (é aí que DBSCAN, baseado em densidade em vez de distância ao centróide, se sai melhor).
>
> **Intuição — PCA vs t-SNE/UMAP**: PCA (mod. [01](01_matematica.md#11-álgebra-linear)) preserva a variância global dos dados — boa para compressão/reconstrução, mas pode esconder estrutura local (clusters próximos podem ficar sobrepostos na projeção). t-SNE e UMAP otimizam para preservar *vizinhança local* (pontos próximos no espaço original ficam próximos na visualização 2D/3D), o que produz visualizações mais "limpas" com clusters separados — mas a distância *entre* clusters na visualização não tem significado confiável (t-SNE/UMAP não preservam estrutura global), e por isso não devem alimentar um modelo downstream, só servir para inspeção visual.
>
> **Checkpoint**: sem olhar o texto, explique por que k-Means falha em clusters com formato alongado (não esférico). Depois, explique por que você não deveria usar a saída de t-SNE como feature de entrada para outro modelo.

### Referências
- `Paper` **t-SNE** — van der Maaten & Hinton (2008). https://www.jmlr.org/papers/v9/vandermaaten08a.html
- `Paper` **UMAP** — McInnes et al. (2018). https://arxiv.org/abs/1802.03426
- `Paper` **A Density-Based Algorithm (DBSCAN)** — Ester et al. (1996). https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

---

## 3.4 Engenharia de features

- **Encoding categórico**: one-hot, label encoding, target encoding, embeddings.
- **Escalonamento**: standardization (z-score), min-max, robust scaling.
- **Tratamento de missing values**: deleção, imputação simples, imputação multivariada.
- **Feature engineering** clássica: interações, polinomiais, agregações temporais.
- **Feature selection**: filter (correlação, mútua informação), wrapper (RFE), embedded (Lasso).

> **Cuidado com target encoding**: usar a média do target por categoria como feature é poderoso, mas vaza informação do rótulo para a feature — se calculado ingenuamente sobre o dataset inteiro (incluindo o próprio ponto sendo codificado), o modelo "aprende" o vazamento em vez do padrão real, e a performance desaba fora da amostra. A forma correta calcula o encoding só com dados de treino (via cross-validation interna), nunca vendo o ponto que está sendo transformado.

### Referências
- `Livro` **Feature Engineering for Machine Learning** — Zheng & Casari.
- `Ferramenta` **scikit-learn — Preprocessing**. https://scikit-learn.org/stable/modules/preprocessing.html

---

## 3.5 Avaliação rigorosa

- **Treino/validação/teste**: por que precisa de três splits.
- **Cross-validation aninhada** para hyperparameter tuning honesto.
- **Curvas de aprendizado** (learning curves) para diagnosticar bias vs variance.
- **Curvas de validação** (validation curves) para escolher hiperparâmetros.
- **Calibração de probabilidades**: Platt scaling, isotonic regression.
- **Análise de erro**: matriz de confusão, residual plots.

> **Intuição — três splits**: treino ajusta os parâmetros; validação escolhe hiperparâmetros e decide "quando parar"; teste mede a performance final, **uma única vez**. Se você usa o mesmo conjunto pra tudo, o modelo (ou você, escolhendo hiperparâmetros) acaba indiretamente "vendo" o teste múltiplas vezes — o número final fica otimista, porque parte da escolha já foi guiada por aquele conjunto. Curvas de aprendizado (erro de treino e validação em função do tamanho do dataset) são o diagnóstico direto de bias vs variância: gap grande entre as duas curvas indica alta variância (mais dados ajudariam); as duas curvas convergindo num erro alto indica alto bias (mais dados não ajudariam, o modelo é fraco demais).
>
> **Checkpoint**: sem olhar o texto, explique por que "tunar no test set" invalida a métrica final reportada. Depois, descreva o que uma curva de aprendizado mostraria para um modelo com alto bias.

### Referências
- `Paper` **A Survey of Cross-Validation Procedures for Model Selection** — Arlot & Celisse (2010). https://arxiv.org/abs/0907.4728

---

## Projetos práticos

### Projeto 3.1 — Implementar k-Means do zero
**O que prova**: que você entende clustering e otimização iterativa.
- NumPy puro, sem `sklearn`.
- Compare resultados em datasets sintéticos (blobs, moons).
- Discuta limitações empíricas.

> **Variante guiada**: rode primeiro no dataset "blobs" (clusters esféricos — deve funcionar bem) e só depois no "moons" (clusters em forma de lua — deve falhar visivelmente). Ver o próprio algoritmo falhar onde a teoria prevê que ele falharia é mais convincente do que só ler a limitação.

### Projeto 3.2 — Pipeline completo em problema tabular
**O que prova**: que você sabe orquestrar um workflow real.
- Dataset: Titanic, Adult Income, ou California Housing.
- Pipeline com `sklearn.Pipeline` e `ColumnTransformer`.
- Compare ≥4 modelos: regressão logística, Random Forest, XGBoost, LightGBM.
- Use cross-validation aninhada para tuning.
- Análise de erro com SHAP (explica módulo [14](14_avaliacao_e_seguranca.md) também).

> **Variante guiada**: antes de comparar os 4 modelos, formule uma hipótese de qual deve performar melhor no seu dataset específico, baseada na intuição da seção 3.2 — depois confira se a hipótese bateu, e se não bateu, investigue por quê (isso ensina mais do que só rodar e comparar números).

### Projeto 3.3 — Detecção de anomalias
**O que prova**: que você sabe abordar problemas não-supervisionados.
- Dataset: fraude em cartão de crédito (Kaggle) ou rede industrial.
- Compare Isolation Forest, One-Class SVM e Autoencoder.
- Discuta trade-offs.

### Projeto 3.4 — Implementar gradient boosting do zero
**O que prova**: que você entende boosting além do XGBoost-como-caixa-preta.
- Implemente um GBM mínimo em NumPy: árvores fracas (decision stumps), gradiente da loss.
- Compare com XGBoost no mesmo problema.

> **Variante guiada**: implemente com apenas 1 árvore fraca primeiro e confirme que o resíduo (erro que a próxima árvore deve corrigir) faz sentido antes de encadear várias — isso isola o mecanismo central do boosting (cada árvore nova mira o erro da anterior) de bugs de acumulação.

---

## Erros comuns

- **Comparar modelos com hiperparâmetros default.** Default ≠ ótimo.
- **Vazamento de dados** (data leakage) — uma feature do futuro entra no treino.
- **Métrica errada para o problema.** Accuracy em dataset desbalanceado é enganosa.
- **Esquecer de calibrar probabilidades** quando a saída de probabilidade importa.
- **Tunar no test set.** Test set é sagrado, toca-se uma vez no fim.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Bias-variance | DL (regularização), generalização em LLMs |
| Cross-entropy loss | Treinamento de qualquer modelo neural |
| Overfitting | Regularização em DL, RLHF |
| One-hot encoding | Tokenização (módulo [06](06_nlp_classico.md)) |
| Embeddings (target encoding) | Word embeddings (módulo [06](06_nlp_classico.md)) |
| Métricas | Avaliação de LLMs (módulo [14](14_avaliacao_e_seguranca.md)) |
| Pipelines | MLOps (módulo [15](15_engenharia_producao.mdx)) |

---

## Checklist de saída

- [ ] Sei explicar quando preferir XGBoost a uma rede neural.
- [ ] Implementei pelo menos um algoritmo from scratch além de regressão linear.
- [ ] Sei desenhar um pipeline de ML do zero, com tratamento honesto de validação.
- [ ] Entendo SHAP/feature importance suficientemente para diagnosticar modelos.
- [ ] Reconheço sinais de overfitting e leakage em código alheio.
