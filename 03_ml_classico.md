# Módulo 03 — Machine Learning Clássico

> **Objetivo**: dominar os algoritmos pré-deep-learning. Não é nostalgia: muitos problemas reais ainda são melhor resolvidos com ML clássico, e os conceitos (overfitting, regularização, validação) são universais.
>
> **Pré-requisitos**: Módulos 01 e 02.
>
> **Tempo de referência**: 4–6 semanas.

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
- **Auto-supervisionado** (cross-link com módulo 04): rótulos derivados dos próprios dados.

### Conceitos transversais
- **Bias-variance tradeoff**.
- **Overfitting vs underfitting**.
- **Regularização**: L1 (Lasso), L2 (Ridge), Elastic Net.
- **Validação**: holdout, k-fold cross-validation, stratified k-fold, time series split.
- **Métricas**: accuracy, precision, recall, F1, AUC-ROC, PR-AUC, MSE, MAE, R², log-loss.
- **Curse of dimensionality**.
- **No Free Lunch Theorem**: nenhum algoritmo é ótimo para tudo.
- **Pipeline de ML**: ingestão → limpeza → features → modelo → avaliação → deploy.

### Referências
- 🎓 **Stanford CS229** — vídeos + notas. https://cs229.stanford.edu/
- 📚 **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman. https://hastie.su.domains/ElemStatLearn/
- 📚 **An Introduction to Statistical Learning** (versão mais acessível do anterior). https://www.statlearning.com/
- 📚 **Pattern Recognition and Machine Learning** — Bishop.

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

### Referências (papers fundadores)
- 📄 **Random Forests** — Breiman (2001). https://link.springer.com/article/10.1023/A:1010933404324
- 📄 **XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin (2016). https://arxiv.org/abs/1603.02754
- 📄 **LightGBM** — Ke et al. (2017). https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- 📄 **A Tutorial on Support Vector Machines for Pattern Recognition** — Burges (1998). https://www.cs.cmu.edu/~cga/ai-course/svmtutorial.pdf

---

## 3.3 Algoritmos não-supervisionados

### Clustering
- **k-Means** (e suas limitações: assume clusters esféricos, sensível a inicialização).
- **k-Means++** (inicialização melhorada).
- **Hierarchical Clustering**: agglomerative, divisive.
- **DBSCAN**: clustering baseado em densidade.
- **Gaussian Mixture Models (GMM)**: clustering probabilístico via EM.

### Redução de dimensionalidade
- **PCA** (já visto no módulo 01).
- **t-SNE**: visualização, não para downstream tasks.
- **UMAP**: alternativa moderna ao t-SNE, mais rápida e preserva mais estrutura.

### Detecção de anomalias
- **Isolation Forest**.
- **One-Class SVM**.
- **Autoencoders** (cross-link com DL — módulo 05).

### Referências
- 📄 **t-SNE** — van der Maaten & Hinton (2008). https://www.jmlr.org/papers/v9/vandermaaten08a.html
- 📄 **UMAP** — McInnes et al. (2018). https://arxiv.org/abs/1802.03426
- 📄 **A Density-Based Algorithm (DBSCAN)** — Ester et al. (1996). https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

---

## 3.4 Engenharia de features

- **Encoding categórico**: one-hot, label encoding, target encoding, embeddings.
- **Escalonamento**: standardization (z-score), min-max, robust scaling.
- **Tratamento de missing values**: deleção, imputação simples, imputação multivariada.
- **Feature engineering** clássica: interações, polinomiais, agregações temporais.
- **Feature selection**: filter (correlação, mútua informação), wrapper (RFE), embedded (Lasso).

### Referências
- 📚 **Feature Engineering for Machine Learning** — Zheng & Casari.
- 🛠 **scikit-learn — Preprocessing**. https://scikit-learn.org/stable/modules/preprocessing.html

---

## 3.5 Avaliação rigorosa

- **Treino/validação/teste**: por que precisa de três splits.
- **Cross-validation aninhada** para hyperparameter tuning honesto.
- **Curvas de aprendizado** (learning curves) para diagnosticar bias vs variance.
- **Curvas de validação** (validation curves) para escolher hiperparâmetros.
- **Calibração de probabilidades**: Platt scaling, isotonic regression.
- **Análise de erro**: matriz de confusão, residual plots.

### Referências
- 📄 **A Survey of Cross-Validation Procedures for Model Selection** — Arlot & Celisse (2010). https://arxiv.org/abs/0907.4728

---

## 🧪 Projetos práticos

### Projeto 3.1 — Implementar k-Means do zero
**O que prova**: que você entende clustering e otimização iterativa.
- NumPy puro, sem `sklearn`.
- Compare resultados em datasets sintéticos (blobs, moons).
- Discuta limitações empíricas.

### Projeto 3.2 — Pipeline completo em problema tabular
**O que prova**: que você sabe orquestrar um workflow real.
- Dataset: Titanic, Adult Income, ou California Housing.
- Pipeline com `sklearn.Pipeline` e `ColumnTransformer`.
- Compare ≥4 modelos: regressão logística, Random Forest, XGBoost, LightGBM.
- Use cross-validation aninhada para tuning.
- Análise de erro com SHAP (explica módulo 14 também).

### Projeto 3.3 — Detecção de anomalias
**O que prova**: que você sabe abordar problemas não-supervisionados.
- Dataset: fraude em cartão de crédito (Kaggle) ou rede industrial.
- Compare Isolation Forest, One-Class SVM e Autoencoder.
- Discuta trade-offs.

### Projeto 3.4 — Implementar gradient boosting do zero
**O que prova**: que você entende boosting além do XGBoost-como-caixa-preta.
- Implemente um GBM mínimo em NumPy: árvores fracas (decision stumps), gradiente da loss.
- Compare com XGBoost no mesmo problema.

---

## ⚠️ Erros comuns

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
| One-hot encoding | Tokenização (módulo 06) |
| Embeddings (target encoding) | Word embeddings (módulo 06) |
| Métricas | Avaliação de LLMs (módulo 14) |
| Pipelines | MLOps (módulo 15) |

---

## Checklist de saída

- [ ] Sei explicar quando preferir XGBoost a uma rede neural.
- [ ] Implementei pelo menos um algoritmo from scratch além de regressão linear.
- [ ] Sei desenhar um pipeline de ML do zero, com tratamento honesto de validação.
- [ ] Entendo SHAP/feature importance suficientemente para diagnosticar modelos.
- [ ] Reconheço sinais de overfitting e leakage em código alheio.
