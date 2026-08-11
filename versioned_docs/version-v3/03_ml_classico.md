---
id: 03_ml_classico
title: "Módulo 03 — Machine Learning Clássico"
sidebar_position: 18
---

# Módulo 03 — Machine Learning Clássico

> **Objetivo**: dominar os algoritmos pré-deep-learning. Não é nostalgia: muitos problemas reais ainda são melhor resolvidos com ML clássico, e os conceitos (overfitting, regularização, validação) são universais.
>
> **Pré-requisitos**: toda a trilha até aqui — em particular, o uso de XGBoost/LightGBM/CatBoost como caixas relativamente prontas no Projeto 4.1, que aqui você entende por dentro (Projeto 3.4 implementa gradient boosting do zero).
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

Em problemas tabulares, gradient boosting frequentemente bate redes neurais — você já viu XGBoost/LightGBM/CatBoost em ação no Projeto 4.1, tratados como caixas relativamente prontas; este módulo abre essas caixas. Conceitos como bias-variância, regularização, validação cruzada são ferramentas mentais para *qualquer* modelo, incluindo LLMs — e saber quando *não* usar deep learning (treinar uma rede para um problema que XGBoost resolve em 5 minutos é desperdício) é tão parte de ser um bom engenheiro de ML quanto saber treinar a rede em si.

---

## 3.1 Fundamentos conceituais

### Tipos de aprendizado
- **Supervisionado**: classificação, regressão.
- **Não-supervisionado**: clustering, redução de dimensionalidade, detecção de anomalias.
- **Semi-supervisionado**: rótulos parciais.
- **Auto-supervisionado** (mod. 04, já visto): rótulos derivados dos próprios dados.

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
> **Aplicação real**: overfitting em LLMs se manifesta diferente (o modelo "decorando" trechos do corpus de treino em vez de generalizar), mas a lógica é idêntica — é por isso que técnicas de regularização deste módulo (weight decay, early stopping) reaparecem sem mudança conceitual no mod. [05](05_deep_learning.md#53-otimização-e-regularização-para-dl), que você já viu.
>
> **Checkpoint**: sem olhar o texto, dê um exemplo (pode ser hipotético) de um modelo com alto bias e outro com alta variância no mesmo problema. Depois, explique por que "zero erro no treino" não é uma meta desejável.

---

## 3.2 Algoritmos supervisionados

### Lineares
- **Regressão Linear** (OLS, Ridge, Lasso).
- **Regressão Logística** (binária e multiclasse via softmax — o mesmo `softmax` do Projeto 5.1).
- **Support Vector Machines (SVM)**: hard margin, soft margin, kernel trick (RBF, polinomial).

### Baseados em árvore
- **Decision Trees**: critérios de split (Gini, entropia, MSE), poda.
- **Random Forests**: bagging.
- **Gradient Boosting**: AdaBoost, GBM, **XGBoost**, **LightGBM**, **CatBoost** — já usados no Projeto 4.1, implementados do zero no Projeto 3.4.

### Baseados em distância
- **k-Nearest Neighbors (kNN)** — já usado como um dos modelos base do stacking no Projeto 4.1.

### Probabilísticos
- **Naive Bayes** (multinomial, Gaussian, Bernoulli).
- **Linear/Quadratic Discriminant Analysis (LDA/QDA)**.

> **Intuição por família**: modelos **lineares** traçam uma reta/plano/hiperplano separando os dados — rápidos, interpretáveis, mas limitados a relações (aproximadamente) lineares, a menos que você projete features não-lineares manualmente. **Árvores de decisão** são uma sequência de perguntas sim/não ("idade > 30? renda > X?") que particiona o espaço em regiões — capturam não-linearidade e interações naturalmente, mas uma única árvore profunda overfita fácil. **kNN** não "aprende" nada explicitamente — classifica um ponto novo olhando para seus vizinhos mais próximos no conjunto de treino (o mesmo princípio da busca por similaridade de cosseno que você já usa desde o Projeto 8.5, com distância euclidiana no lugar de cosseno); simples, mas caro em tempo de inferência e sofre com a maldição da dimensionalidade (em espaços de alta dimensão, "próximo" perde significado).
>
> **Intuição — bagging vs boosting** (a distinção mais confundida deste módulo): **Random Forest** (bagging) treina muitas árvores *independentes*, cada uma numa amostra aleatória dos dados, e faz a média das previsões — reduz variância porque erros aleatórios de árvores individuais se cancelam. **Gradient Boosting** (XGBoost, LightGBM, que você já usou no Projeto 4.1) treina árvores *sequencialmente*, cada nova árvore focando em corrigir os erros que as anteriores ainda cometem (aprendendo a prever o *resíduo* do ensemble atual) — reduz bias porque o ensemble fica progressivamente melhor no que ainda erra. São estratégias opostas: bagging paraleliza e reduz variância; boosting é sequencial e reduz bias. É por isso que gradient boosting costuma vencer em benchmarks tabulares (Kaggle inclusive) — ele ataca diretamente o erro residual, enquanto Random Forest só reduz ruído de árvores já razoavelmente boas. Você implementa esse mecanismo de correção sequencial do zero no Projeto 3.4.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre bagging e boosting numa frase cada. Depois, explique por que kNN sofre mais com dimensionalidade alta do que uma árvore de decisão.

---

## 3.3 Algoritmos não-supervisionados

### Clustering
- **k-Means** (e suas limitações: assume clusters esféricos, sensível a inicialização) — implementado do zero no Projeto 3.1.
- **k-Means++** (inicialização melhorada).
- **Hierarchical Clustering**: agglomerative, divisive.
- **DBSCAN**: clustering baseado em densidade.
- **Gaussian Mixture Models (GMM)**: clustering probabilístico via EM.

### Redução de dimensionalidade
- **PCA** (mod. 01, já visto).
- **t-SNE**: visualização, não para downstream tasks.
- **UMAP**: alternativa moderna ao t-SNE, mais rápida e preserva mais estrutura.

### Detecção de anomalias
- **Isolation Forest** — implementado no Projeto 3.3.
- **One-Class SVM**.
- **Autoencoders** — você já implementou um (VAE, uma variante mais sofisticada) no Projeto 5.5; a versão vanilla é usada para anomaly detection no Projeto 3.3.

> **Intuição — k-Means**: alterna entre dois passos até estabilizar — (1) atribuir cada ponto ao centróide mais próximo, (2) recalcular cada centróide como a média dos pontos atribuídos a ele. É um processo iterativo simples, mas assume que clusters são "bolhas" aproximadamente esféricas de tamanho parecido — falha visivelmente em clusters alongados, aninhados ou de densidade muito diferente (é aí que DBSCAN, baseado em densidade em vez de distância ao centróide, se sai melhor).
>
> **Intuição — PCA vs t-SNE/UMAP**: PCA preserva a variância global dos dados — boa para compressão/reconstrução, mas pode esconder estrutura local (clusters próximos podem ficar sobrepostos na projeção). t-SNE e UMAP otimizam para preservar *vizinhança local* (pontos próximos no espaço original ficam próximos na visualização 2D/3D), o que produz visualizações mais "limpas" com clusters separados — mas a distância *entre* clusters na visualização não tem significado confiável (t-SNE/UMAP não preservam estrutura global), e por isso não devem alimentar um modelo downstream, só servir para inspeção visual.
>
> **Checkpoint**: sem olhar o texto, explique por que k-Means falha em clusters com formato alongado (não esférico). Depois, explique por que você não deveria usar a saída de t-SNE como feature de entrada para outro modelo.

---

## 3.4 Engenharia de features

- **Encoding categórico**: one-hot, label encoding, target encoding, embeddings.
- **Escalonamento**: standardization (z-score), min-max, robust scaling.
- **Tratamento de missing values**: deleção, imputação simples, imputação multivariada.
- **Feature engineering** clássica: interações, polinomiais, agregações temporais.
- **Feature selection**: filter (correlação, mútua informação), wrapper (RFE), embedded (Lasso).

> **Cuidado com target encoding**: usar a média do target por categoria como feature é poderoso, mas vaza informação do rótulo para a feature — se calculado ingenuamente sobre o dataset inteiro (incluindo o próprio ponto sendo codificado), o modelo "aprende" o vazamento em vez do padrão real, e a performance desaba fora da amostra. A forma correta calcula o encoding só com dados de treino (via cross-validation interna), nunca vendo o ponto que está sendo transformado — o mesmo cuidado do `gerar_previsoes_out_of_fold` do Projeto 4.1.

---

## 3.5 Avaliação rigorosa

- **Treino/validação/teste**: por que precisa de três splits.
- **Cross-validation aninhada** para hyperparameter tuning honesto.
- **Curvas de aprendizado** (learning curves) para diagnosticar bias vs variance.
- **Curvas de validação** (validation curves) para escolher hiperparâmetros.
- **Calibração de probabilidades**: Platt scaling, isotonic regression — o mesmo problema de calibração já discutido no mod. 14.
- **Análise de erro**: matriz de confusão, residual plots, SHAP (usado no Projeto 3.2).

> **Intuição — três splits**: treino ajusta os parâmetros; validação escolhe hiperparâmetros e decide "quando parar"; teste mede a performance final, **uma única vez**. Se você usa o mesmo conjunto pra tudo, o modelo (ou você, escolhendo hiperparâmetros) acaba indiretamente "vendo" o teste múltiplas vezes — o número final fica otimista, porque parte da escolha já foi guiada por aquele conjunto. Curvas de aprendizado (erro de treino e validação em função do tamanho do dataset) são o diagnóstico direto de bias vs variância: gap grande entre as duas curvas indica alta variância (mais dados ajudariam); as duas curvas convergindo num erro alto indica alto bias (mais dados não ajudariam, o modelo é fraco demais).
>
> **Checkpoint**: sem olhar o texto, explique por que "tunar no test set" invalida a métrica final reportada. Depois, descreva o que uma curva de aprendizado mostraria para um modelo com alto bias.

---

## Projetos práticos

### Projeto 3.1 — Implementar k-Means do zero

Você vai implementar o algoritmo completo em NumPy puro, e observar diretamente onde ele funciona bem e onde falha.

**Pré-requisitos**: `pip install numpy matplotlib scikit-learn` (scikit-learn só para gerar os datasets sintéticos de teste).

```python
import numpy as np

def kmeans(X, k, n_iteracoes=100, tol=1e-4):
    indices_iniciais = np.random.choice(len(X), k, replace=False)
    centroides = X[indices_iniciais].copy()

    for iteracao in range(n_iteracoes):
        # passo 1: atribui cada ponto ao centróide mais próximo
        distancias = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)  # (n_pontos, k)
        atribuicoes = distancias.argmin(axis=1)

        # passo 2: recalcula cada centróide como a média dos pontos atribuídos a ele
        novos_centroides = np.array([
            X[atribuicoes == cluster].mean(axis=0) if (atribuicoes == cluster).any() else centroides[cluster]
            for cluster in range(k)
        ])

        deslocamento = np.linalg.norm(novos_centroides - centroides)
        centroides = novos_centroides
        if deslocamento < tol:
            break

    return centroides, atribuicoes
```

`distancias[:, None, :] - centroides[None, :, :]` usa broadcasting do NumPy para calcular, numa única operação vetorizada, a distância de cada ponto a cada centróide (sem loop explícito ponto a ponto) — o resultado tem forma `(n_pontos, k)`, e `argmin(axis=1)` escolhe, para cada ponto, o centróide mais próximo. O `if (atribuicoes == cluster).any() else centroides[cluster]` trata o caso raro de um cluster ficar sem nenhum ponto atribuído (mantém o centróide antigo em vez de tentar calcular a média de um conjunto vazio, que daria `NaN`).

**Rode primeiro em "blobs"** (clusters esféricos gerados por `sklearn.datasets.make_blobs`) — deve convergir para uma separação limpa, visualmente confirmável com um scatter plot colorido por `atribuicoes`. **Depois rode em "moons"** (`sklearn.datasets.make_moons`, dois clusters em forma de meia-lua entrelaçada) — k-Means deve falhar visivelmente aqui, cortando as luas ao meio em vez de respeitar sua forma, exatamente como a Intuição da seção 3.3 prevê (k-Means assume clusters aproximadamente esféricos). Ver o próprio algoritmo que você escreveu falhar onde a teoria prevê que ele falharia é mais convincente do que só ler a limitação.

---

### Projeto 3.2 — Pipeline completo em problema tabular

Você vai construir um pipeline de ML honesto — com pré-processamento declarado explicitamente, tuning validado corretamente, e análise de erro com SHAP — comparando 4 modelos.

**Pré-requisitos**: `pip install scikit-learn xgboost lightgbm shap pandas`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("titanic.csv")  # ou Adult Income / California Housing
colunas_numericas = ["age", "fare", "sibsp", "parch"]
colunas_categoricas = ["sex", "embarked", "pclass"]

pre_processamento = ColumnTransformer([
    ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), colunas_numericas),
    ("categoricas", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), colunas_categoricas),
])

X_treino, X_teste, y_treino, y_teste = train_test_split(df.drop(columns=["survived"]), df["survived"], test_size=0.2, random_state=0)

modelos = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "xgboost": XGBClassifier(n_estimators=200),
    "lightgbm": LGBMClassifier(n_estimators=200, verbose=-1),
}
```

`ColumnTransformer` aplica transformações diferentes a colunas diferentes (imputação + escalonamento nas numéricas, imputação + one-hot nas categóricas) dentro de um único objeto — encapsulando exatamente o tipo de decisão de pré-processamento que, se feita manualmente fora de um pipeline, é fácil de aplicar de forma inconsistente entre treino e teste (uma fonte comum de bugs sutis e de vazamento de dado).

**Cross-validation aninhada para tuning honesto**: o `GridSearchCV` já cuida disso — o `cv` interno dele escolhe hiperparâmetros usando *apenas* dados de treino, e a avaliação final ainda acontece num conjunto de teste nunca visto durante a busca:

```python
resultados = {}
for nome, modelo in modelos.items():
    pipeline_completo = Pipeline([("preprocessamento", pre_processamento), ("modelo", modelo)])
    grid = {"modelo__n_estimators": [100, 200, 300]} if nome != "logistic" else {"modelo__C": [0.1, 1.0, 10.0]}

    busca = GridSearchCV(pipeline_completo, grid, cv=5, scoring="roc_auc")
    busca.fit(X_treino, y_treino)
    resultados[nome] = {"melhor_score_cv": busca.best_score_, "score_teste": busca.score(X_teste, y_teste)}

for nome, r in resultados.items():
    print(f"{nome}: CV={r['melhor_score_cv']:.3f} teste={r['score_teste']:.3f}")
```

**Antes de rodar**, formule uma hipótese de qual modelo deve performar melhor no seu dataset específico, baseada na Intuição da seção 3.2 (dados tabulares com poucas features e relações não-muito-complexas tendem a favorecer boosting; datasets pequenos e limpos às vezes favorecem regressão logística por menor variância) — depois confira se a hipótese bateu, e se não bateu, investigue por quê.

**Análise de erro com SHAP**, no melhor modelo encontrado:

```python
import shap

melhor_pipeline = resultados_pipelines["xgboost"]  # o pipeline treinado do modelo com melhor score_teste
explicador = shap.TreeExplainer(melhor_pipeline.named_steps["modelo"])
X_teste_transformado = melhor_pipeline.named_steps["preprocessamento"].transform(X_teste)
valores_shap = explicador.shap_values(X_teste_transformado)

shap.summary_plot(valores_shap, X_teste_transformado, feature_names=nomes_das_features_apos_encoding)
```

`shap.TreeExplainer` calcula, para cada previsão individual, o quanto cada feature contribuiu positiva ou negativamente para aquela previsão específica (não só uma importância global agregada) — o `summary_plot` mostra essa distribuição de contribuições por feature em todo o conjunto de teste de uma vez, revelando não só quais features importam, mas em que direção e com que consistência.

---

### Projeto 3.3 — Detecção de anomalias

Você vai comparar 3 abordagens de detecção de anomalias na mesma tarefa (detecção de fraude), cada uma com um princípio de funcionamento diferente.

**Pré-requisitos**: `pip install scikit-learn torch`, um dataset de fraude em cartão de crédito (disponível no Kaggle) ou qualquer dataset com uma classe rara de interesse.

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# 1. Isolation Forest — isola pontos anômalos particionando o espaço aleatoriamente;
# anomalias tendem a ser isoladas com menos cortes que pontos normais (mais "fáceis de separar")
iso_forest = IsolationForest(contamination=0.01, random_state=0)
previsoes_iso = iso_forest.fit_predict(X_treino)  # retorna 1 (normal) ou -1 (anomalia)

# 2. One-Class SVM — aprende uma fronteira ao redor da região "normal" dos dados,
# tudo fora dela é anomalia
ocsvm = OneClassSVM(nu=0.01, kernel="rbf")
previsoes_ocsvm = ocsvm.fit_predict(X_treino)

# 3. Autoencoder — treina para reconstruir dados normais; erro de reconstrução alto = anomalia
import torch.nn as nn

class AutoencoderAnomalia(nn.Module):
    def __init__(self, dim_entrada, dim_latente=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_entrada, 32), nn.ReLU(), nn.Linear(32, dim_latente))
        self.decoder = nn.Sequential(nn.Linear(dim_latente, 32), nn.ReLU(), nn.Linear(32, dim_entrada))

    def forward(self, x):
        return self.decoder(self.encoder(x))

autoencoder = AutoencoderAnomalia(dim_entrada=X_treino.shape[1])
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)

# treine só com dados normais (o mesmo princípio de "aprender a reconstruir o padrão comum" do Projeto 5.5)
for epoca in range(50):
    reconstrucao = autoencoder(torch.tensor(X_treino_apenas_normais, dtype=torch.float32))
    loss = nn.functional.mse_loss(reconstrucao, torch.tensor(X_treino_apenas_normais, dtype=torch.float32))
    optimizer.zero_grad(); loss.backward(); optimizer.step()

with torch.no_grad():
    erro_reconstrucao = ((autoencoder(torch.tensor(X_teste, dtype=torch.float32)) - torch.tensor(X_teste, dtype=torch.float32)) ** 2).mean(dim=1)
    limiar = erro_reconstrucao.quantile(0.99)  # os 1% com maior erro de reconstrução são marcados como anomalia
    previsoes_autoencoder = (erro_reconstrucao > limiar).numpy()
```

Os três métodos atacam o problema de ângulos diferentes: Isolation Forest é baseado em quão "fácil" é isolar um ponto via partições aleatórias; One-Class SVM aprende uma fronteira geométrica explícita ao redor da região normal; o Autoencoder aprende a *reconstruir* dados normais bem, e assume que dados anômalos (nunca vistos num padrão parecido durante o treino) serão reconstruídos mal — um erro de reconstrução alto vira o sinal de anomalia.

**Compare os três** com `classification_report` (usando os rótulos reais de fraude, disponíveis no dataset, só para avaliação — os três métodos, no treino, não usam esses rótulos) e discuta os trade-offs observados: Isolation Forest costuma ser o mais rápido de treinar; One-Class SVM pode ser sensível à escolha do kernel/hiperparâmetros; o Autoencoder exige mais dados e tempo de treino, mas pode capturar padrões não-lineares mais ricos que os outros dois.

---

### Projeto 3.4 — Implementar gradient boosting do zero

Você vai implementar um GBM mínimo — árvores fracas (decision stumps) treinadas sequencialmente sobre o resíduo — e comparar com o XGBoost que você já usou como caixa-preta no Projeto 4.1.

**Pré-requisitos**: `pip install scikit-learn numpy`.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GBMSimples:
    def __init__(self, n_arvores=50, learning_rate=0.1, profundidade_maxima=2):
        self.n_arvores = n_arvores
        self.learning_rate = learning_rate
        self.profundidade_maxima = profundidade_maxima
        self.arvores = []
        self.previsao_inicial = None

    def fit(self, X, y):
        self.previsao_inicial = y.mean()   # ponto de partida: prever sempre a média (o "modelo mais simples possível")
        previsao_atual = np.full(len(y), self.previsao_inicial)

        for i in range(self.n_arvores):
            residuo = y - previsao_atual   # o que o ensemble atual ainda está errando
            arvore = DecisionTreeRegressor(max_depth=self.profundidade_maxima)
            arvore.fit(X, residuo)         # a nova árvore aprende a PREVER O RESÍDUO, não y diretamente
            self.arvores.append(arvore)

            previsao_atual += self.learning_rate * arvore.predict(X)

        return self

    def predict(self, X):
        previsao = np.full(len(X), self.previsao_inicial)
        for arvore in self.arvores:
            previsao += self.learning_rate * arvore.predict(X)
        return previsao
```

O mecanismo central, exatamente como descrito na Intuição da seção 3.2: cada nova árvore não tenta prever `y` diretamente — tenta prever `residuo = y - previsao_atual`, o erro que o ensemble construído até agora ainda comete. Somar `learning_rate * arvore.predict(X)` à previsão acumulada, repetidamente, é o que faz o ensemble ficar progressivamente melhor — `learning_rate` pequeno (0.1, como aqui) faz cada árvore contribuir só um pouco, exigindo mais árvores mas geralmente generalizando melhor do que poucas árvores com contribuição grande cada.

**Valide com 1 árvore primeiro**: rode `GBMSimples(n_arvores=1)`, calcule `residuo` manualmente (`y - y.mean()`), e confirme que faz sentido (deveria ter média ≈ 0, e a árvore treinada nele deveria capturar algum padrão real dos dados) antes de encadear várias árvores — isso isola o mecanismo central do boosting de bugs de acumulação ao longo de muitas iterações.

**Compare com XGBoost** no mesmo problema (regressão simples — adapte para classificação binária usando `residuo` baseado em log-loss em vez de MSE, se quiser ir além): treine `GBMSimples` e `XGBRegressor` com hiperparâmetros equivalentes (mesmo número de árvores, profundidade, learning rate) e compare o erro final — não devem ser idênticos (XGBoost tem otimizações substanciais: regularização adicional, uso de segunda derivada da loss, tratamento eficiente de missing values), mas a tendência de queda do erro conforme mais árvores são adicionadas deve ser qualitativamente parecida, confirmando que sua implementação captura o mecanismo real, só sem as otimizações de produção.

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
| Bias-variance | DL (regularização, já visto), generalização em LLMs |
| Cross-entropy loss | Treinamento de qualquer modelo neural (já praticado desde o mod. 08) |
| Overfitting | Regularização em DL, RLHF (já visto) |
| One-hot encoding | Tokenização (mod. 06, já visto) |
| Embeddings (target encoding) | Word embeddings (mod. 06, já visto) |
| Métricas | Avaliação de LLMs (mod. 14, já visto) |
| Pipelines | MLOps (mod. 15, já visto) |

---

## Checklist de saída

- [ ] Sei explicar quando preferir XGBoost a uma rede neural (se não, revise a seção "Por que isso importa").
- [ ] Implementei k-Means e gradient boosting from scratch, não só usei bibliotecas (se não, revise os Projetos 3.1 e 3.4).
- [ ] Sei desenhar um pipeline de ML do zero, com tratamento honesto de validação (`ColumnTransformer`, cross-validation aninhada) (se não, revise o Projeto 3.2).
- [ ] Entendo SHAP/feature importance suficientemente para diagnosticar modelos (se não, revise o Projeto 3.2).
- [ ] Reconheço sinais de overfitting e leakage em código alheio (se não, revise a seção 3.4 e "Erros comuns").
