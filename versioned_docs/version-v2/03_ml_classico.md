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

> **Intuição — regularização e validação**: **L1 (Lasso)** soma o valor absoluto dos pesos à loss — geometricamente, isso empurra coeficientes pouco úteis exatamente a zero, funcionando como seleção automática de features. **L2 (Ridge)** soma o quadrado dos pesos — encolhe todos os coeficientes suavemente em direção a zero, sem zerar nenhum. **Elastic Net** combina os dois, útil quando há grupos de features correlacionadas (L1 sozinho tende a escolher arbitrariamente uma do grupo e zerar as outras). Para validação, cada técnica resolve um problema específico: **holdout** (um único split treino/validação) é rápido, mas com poucos dados a métrica final tem alta variância, dependendo de sorte de quais pontos caíram em cada lado; **k-fold cross-validation** roda o treino/validação `k` vezes, girando qual fatia é validação a cada vez, e reporta a média — usa os dados com muito mais eficiência, ao custo de `k`× o tempo de treino; **stratified k-fold** garante que cada fatia preserve a proporção original de classes, crítico em datasets desbalanceados (sem isso, uma fatia por azar pode ficar quase sem exemplos da classe minoritária); **time series split** respeita a ordem temporal, treinando sempre em dados passados e validando em dados futuros — usar k-fold comum em série temporal deixa o modelo "ver o futuro" durante o treino, inflando a métrica de forma irrealista.
>
> **Intuição — métricas**: para classificação, tudo começa da **matriz de confusão**: cada previsão de um classificador binário cai em um de quatro grupos — verdadeiro positivo (TP), falso positivo (FP), verdadeiro negativo (TN), falso negativo (FN). **Accuracy** é `(TP+TN)/total` — parece natural, mas engana em dados desbalanceados: num dataset com 99% de exemplos negativos, prever sempre "negativo" já dá 99% de accuracy sem aprender nada. **Precision** é `TP/(TP+FP)` — de tudo que o modelo chamou de positivo, quanto realmente era; importa quando um falso positivo é caro (marcar um email legítimo como spam). **Recall** é `TP/(TP+FN)` — de tudo que realmente era positivo, quanto o modelo capturou; importa quando um falso negativo é caro (deixar passar um caso de fraude). Há sempre um trade-off entre os dois. **F1** é a média harmônica de precision e recall — um resumo único que só fica alto quando os dois são altos, diferente da média aritmética, que pode ficar enganosamente alta com um dos dois muito baixo. **AUC-ROC** (Area Under the Curve — Receiver Operating Characteristic) mede a qualidade do ranking de scores do modelo em todos os thresholds possíveis de uma vez, mas fica otimista em datasets muito desbalanceados; **PR-AUC** (Precision-Recall AUC) é a alternativa preferida nesse caso, porque não conta os (numerosos) verdadeiros negativos, focando só na classe rara. Para regressão: **MSE** (Mean Squared Error) penaliza erros grandes desproporcionalmente mais que pequenos, por causa do quadrado — sensível a outliers; **MAE** (Mean Absolute Error) penaliza todo erro proporcionalmente ao seu tamanho — mais robusto a outliers; **R²** mede a fração da variância dos dados que o modelo explica (1 = perfeito, 0 = tão bom quanto prever sempre a média). **Log-loss** é a cross-entropy do mod. [01](01_matematica.md#13-probabilidade-e-estatística) aplicada a probabilidades previstas — diferente de accuracy, penaliza um modelo *confiante e errado* muito mais que um modelo incerto e errado.
>
> **Intuição — dimensionalidade, No Free Lunch e pipeline**: a **maldição da dimensionalidade** é o fenômeno de, conforme o número de features cresce, o volume do espaço crescer exponencialmente enquanto a quantidade de dados fica fixa — os pontos ficam cada vez mais esparsos, e noções de "distância" ou "vizinhança" (usadas por kNN, k-Means) perdem poder discriminativo, porque quase todo par de pontos acaba aproximadamente equidistante. O **No Free Lunch Theorem** formaliza algo já intuído na prática: em média sobre *todos* os problemas possíveis, nenhum algoritmo domina os outros — um algoritmo só é melhor porque explora alguma estrutura específica do seu problema (linearidade, esparsidade, localidade), não porque é "melhor" no abstrato. É por isso que comparar empiricamente vários modelos (Projeto 3.2) não é desleixo, é a única forma correta de escolher, dado o teorema. O **pipeline de ML** (ingestão → limpeza → features → modelo → avaliação → deploy) formaliza a ordem em que essas decisões precisam acontecer — pular uma etapa não elimina o problema, só o adia para depois do deploy, quando é mais caro corrigir.
>
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
- **Support Vector Machines (SVM)**: hard margin, soft margin, kernel trick (RBF — Radial Basis Function, polinomial).

### Baseados em árvore
- **Decision Trees**: critérios de split (Gini, entropia, MSE), poda.
- **Random Forests**: bagging.
- **Gradient Boosting**: AdaBoost, GBM (Gradient Boosting Machine), **XGBoost** (eXtreme Gradient Boosting), **LightGBM** (Light Gradient Boosting Machine), **CatBoost** (Categorical Boosting).

### Baseados em distância
- **k-Nearest Neighbors (kNN)**.

### Probabilísticos
- **Naive Bayes** (multinomial, Gaussian, Bernoulli).
- **Linear/Quadratic Discriminant Analysis (LDA/QDA)**.

> **Intuição por família**: modelos **lineares** traçam uma reta/plano/hiperplano separando os dados — rápidos, interpretáveis, mas limitados a relações (aproximadamente) lineares, a menos que você projete features não-lineares manualmente. **Árvores de decisão** são uma sequência de perguntas sim/não ("idade > 30? renda > X?") que particiona o espaço em regiões — capturam não-linearidade e interações naturalmente, mas uma única árvore profunda overfita fácil. **kNN** não "aprende" nada explicitamente — classifica um ponto novo olhando para seus vizinhos mais próximos no conjunto de treino; simples, mas caro em tempo de inferência e sofre com a maldição da dimensionalidade (em espaços de alta dimensão, "próximo" perde significado).
>
> **Intuição — SVM, split de árvore e modelos probabilísticos**: **SVM** busca o hiperplano que separa as classes com a *maior margem* possível — não qualquer separador, mas o mais "confortavelmente" longe dos pontos de cada classe (os **vetores de suporte**, os pontos mais próximos da fronteira, são os únicos que realmente determinam onde ela fica). **Soft margin** permite alguns pontos do lado errado, trocando margem por tolerância a ruído; o **kernel trick** permite fronteiras não-lineares sem projetar explicitamente os dados num espaço de dimensão maior — calcula o produto interno *como se* os dados já estivessem projetados, sem nunca computar a projeção de fato. Para árvores, o critério de split escolhe, a cada nó, a pergunta que mais reduz a "impureza" dos dois lados resultantes: **Gini** e **entropia** medem impureza de classificação de formas ligeiramente diferentes, mas quase sempre concordam na prática (Gini é computacionalmente mais barata, entropia tem lastro em teoria da informação); **MSE** é o critério equivalente para árvores de regressão, escolhendo o split que mais reduz a variância dentro de cada lado. **Naive Bayes** aplica o Teorema de Bayes do mod. [01](01_matematica.md#13-probabilidade-e-estatística) assumindo — ingenuamente, daí o nome — que as features são independentes entre si dado a classe; uma suposição quase sempre falsa na prática, mas que ainda produz um classificador surpreendentemente bom e extremamente rápido de treinar. **LDA/QDA** assumem que cada classe segue uma distribuição Gaussiana: LDA (Linear Discriminant Analysis) assume que todas as classes compartilham a mesma matriz de covariância, produzindo uma fronteira linear; QDA (Quadratic Discriminant Analysis) permite covariâncias diferentes por classe, fronteira quadrática mais flexível, mas exige mais dados para estimar bem.
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
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise): clustering baseado em densidade.
- **Gaussian Mixture Models (GMM)**: clustering probabilístico via EM (Expectation-Maximization).

### Redução de dimensionalidade
- **PCA** (já visto no módulo [01](01_matematica.md)).
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): visualização, não para downstream tasks.
- **UMAP** (Uniform Manifold Approximation and Projection): alternativa moderna ao t-SNE, mais rápida e preserva mais estrutura.

### Detecção de anomalias
- **Isolation Forest**.
- **One-Class SVM**.
- **Autoencoders** (cross-link com DL — módulo [05](05_deep_learning.md)).

> **Intuição — GMM/EM e detecção de anomalias**: **Gaussian Mixture Models** generalizam k-Means para clustering "suave" (probabilístico): em vez de atribuir cada ponto a exatamente um cluster, GMM modela os dados como uma mistura de várias Gaussianas e atribui a cada ponto uma *probabilidade* de pertencer a cada cluster. O algoritmo **EM** treina isso alternando entre estimar essas probabilidades dado os parâmetros atuais das Gaussianas (passo E, de *expectation*) e reestimar os parâmetros das Gaussianas dado essas probabilidades (passo M, de *maximization*) — a mesma estrutura de alternância de k-Means, generalizada. **DBSCAN** não assume nenhuma forma de cluster: define um cluster como uma região *densa* de pontos conectados — todo ponto com vizinhos suficientes num raio `ε` vira um "núcleo", e núcleos próximos se fundem no mesmo cluster — encontra clusters de formato arbitrário e naturalmente marca pontos isolados como ruído, sem exigir saber `k` de antemão, ao custo de precisar ajustar `ε` e o número mínimo de vizinhos. **Hierarchical clustering** não exige escolher `k` antecipadamente: constrói uma árvore inteira de agrupamentos, do modo *agglomerative* (começa com cada ponto em seu próprio cluster e vai fundindo os mais próximos) ou *divisive* (começa com um único cluster e vai dividindo) — você corta a árvore na altura desejada depois, escolhendo `k` visualmente a partir do dendrograma, em vez de precisar declará-lo de antemão. Para detecção de anomalias, as três famílias citadas atacam o problema de ângulos diferentes: **Isolation Forest** particiona o espaço aleatoriamente e mede quantos cortes são necessários para isolar um ponto — anomalias, por serem "diferentes", tendem a se isolar em poucos cortes; **One-Class SVM** aprende uma fronteira geométrica ao redor da região "normal" dos dados, marcando tudo fora dela como anômalo; **Autoencoders** treinam para reconstruir bem os dados normais, e usam o erro de reconstrução como sinal — um dado nunca visto num padrão parecido durante o treino tende a ser mal reconstruído.
>
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
- **Feature selection**: filter (correlação, mútua informação), wrapper (RFE — Recursive Feature Elimination), embedded (Lasso).

> **Intuição — engenharia de features**: **encoding categórico** tem que virar número de algum jeito: **one-hot** cria uma coluna binária por categoria (seguro, mas explode em dimensionalidade com muitas categorias); **label encoding** atribui um inteiro arbitrário por categoria (compacto, mas introduz uma ordem falsa que modelos lineares podem interpretar erroneamente como significativa); **target encoding** substitui a categoria pela média do target naquela categoria (poderoso, mas vaza informação se não calculado com cuidado — ver o alerta abaixo); **embeddings** (mod. [06](06_nlp_classico.md)) aprendem uma representação vetorial densa por categoria, útil quando há muitas categorias com relações latentes entre si. Para **escalonamento**: **standardization** (z-score) é a escolha padrão para modelos que dependem de distância ou gradiente (SVM, kNN, regressão regularizada, redes neurais); **min-max** comprime tudo para `[0,1]`, útil quando você precisa de limites conhecidos, mas sensível a outliers (um único valor extremo comprime todo o resto pertinho de zero); **robust scaling** usa mediana e IQR em vez de média e desvio-padrão, resistindo a outliers que distorceriam os outros dois métodos. Para **missing values**: deleção é simples mas descarta dados e pode enviesar a amostra se a ausência não for aleatória; imputação simples (média/mediana/moda) é rápida mas ignora relações entre features; imputação multivariada estima o valor ausente a partir de outras features, mais precisa e mais cara. Para **feature selection**: métodos *filter* avaliam cada feature isoladamente antes de qualquer modelo — rápidos, mas ignoram interações; métodos *wrapper* como RFE treinam e re-treinam o modelo removendo features iterativamente — mais precisos, muito mais caros; métodos *embedded* (Lasso) selecionam features como parte do próprio treino do modelo, sem etapa separada.
>
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
- **Análise de erro**: matriz de confusão, residual plots, SHAP (SHapley Additive exPlanations).

> **Intuição — calibração e análise de erro**: um modelo "calibrado" é aquele cujas probabilidades previstas correspondem à frequência real — entre todas as previsões com "70% de confiança", cerca de 70% devem de fato estar corretas. Muitos modelos (SVM, árvores, redes neurais sem cuidado extra) produzem scores que *parecem* probabilidades mas não são calibrados. **Platt scaling** ajusta uma regressão logística simples sobre os scores do modelo já treinado, mapeando-os para probabilidades calibradas; **isotonic regression** faz o mesmo mapeamento sem assumir uma forma funcional específica — mais flexível, mas precisa de mais dados de validação para não overfitar o próprio mapeamento. A **matriz de confusão** (a tabela de TP/FP/TN/FN da seção 3.1) é o ponto de partida de qualquer análise de erro de classificação — revela não só *quanto* o modelo erra, mas *que tipo* de erro predomina. Para regressão, **residual plots** (erro previsto-menos-real no eixo Y, valor previsto no eixo X) revelam padrões que uma métrica agregada esconde — um funil (variância do erro crescendo com o valor previsto) indica heterocedasticidade; uma curva sistemática indica que o modelo está deixando de capturar alguma não-linearidade real dos dados.
>
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

> Quer ver a implementação completa em NumPy puro, com a explicação do truque de broadcasting que evita o loop ponto a ponto? A versão Acelerado tem o código completo no [Projeto 3.1 — Implementar k-Means do zero](/trilha-llm/03_ml_classico#projeto-31--implementar-k-means-do-zero).

### Projeto 3.2 — Pipeline completo em problema tabular
**O que prova**: que você sabe orquestrar um workflow real.
- Dataset: Titanic, Adult Income, ou California Housing.
- Pipeline com `sklearn.Pipeline` e `ColumnTransformer`.
- Compare ≥4 modelos: regressão logística, Random Forest, XGBoost, LightGBM.
- Use cross-validation aninhada para tuning.
- Análise de erro com SHAP (explica módulo [14](14_avaliacao_e_seguranca.md) também).

> **Variante guiada**: antes de comparar os 4 modelos, formule uma hipótese de qual deve performar melhor no seu dataset específico, baseada na intuição da seção 3.2 — depois confira se a hipótese bateu, e se não bateu, investigue por quê (isso ensina mais do que só rodar e comparar números).

> Quer ver esse pipeline completo — `ColumnTransformer`, `GridSearchCV` com cross-validation aninhada, e a análise com SHAP — rodando com código real? A versão Acelerado tem tudo isso no [Projeto 3.2 — Pipeline completo em problema tabular](/trilha-llm/03_ml_classico#projeto-32--pipeline-completo-em-problema-tabular).

### Projeto 3.3 — Detecção de anomalias
**O que prova**: que você sabe abordar problemas não-supervisionados.
- Dataset: fraude em cartão de crédito (Kaggle) ou rede industrial.
- Compare Isolation Forest, One-Class SVM e Autoencoder.
- Discuta trade-offs.

> Quer ver os três métodos implementados e comparados no mesmo dataset de fraude, com o código completo de cada um? A versão Acelerado tem isso no [Projeto 3.3 — Detecção de anomalias](/trilha-llm/03_ml_classico#projeto-33--detecção-de-anomalias).

### Projeto 3.4 — Implementar gradient boosting do zero
**O que prova**: que você entende boosting além do XGBoost-como-caixa-preta.
- Implemente um GBM mínimo em NumPy: árvores fracas (decision stumps), gradiente da loss.
- Compare com XGBoost no mesmo problema.

> **Variante guiada**: implemente com apenas 1 árvore fraca primeiro e confirme que o resíduo (erro que a próxima árvore deve corrigir) faz sentido antes de encadear várias — isso isola o mecanismo central do boosting (cada árvore nova mira o erro da anterior) de bugs de acumulação.

> Quer ver esse GBM mínimo implementado e comparado diretamente com XGBoost no mesmo problema? A versão Acelerado tem o código completo no [Projeto 3.4 — Implementar gradient boosting do zero](/trilha-llm/03_ml_classico#projeto-34--implementar-gradient-boosting-do-zero).

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

## Saiba mais

Alguns tópicos deste módulo foram citados sem profundidade — grandes demais para caber aqui sem desviar do fluxo principal:

- **AdaBoost, o algoritmo original de boosting** (3.2) — historicamente anterior ao Gradient Boosting moderno, pondera exemplos mal-classificados em vez de ajustar sobre o resíduo; entender a diferença ajuda a entender por que GBM generalizou a ideia. `Paper` **A Decision-Theoretic Generalization of On-Line Learning** — Freund & Schapire (1997), o paper original do AdaBoost.
- **Teoria do kernel trick** (3.2) — por que o produto interno num espaço de dimensão maior pode ser calculado sem nunca projetar os dados lá (o Teorema de Mercer). `Livro` *Pattern Recognition and Machine Learning* — Bishop, cap. 6.
- **Modelos clássicos de série temporal** (ARIMA, exponential smoothing) — fora do escopo deste módulo (focado em dados tabulares i.i.d.), mas relevantes sempre que a ordem temporal é a estrutura central dos dados, não só uma restrição de validação (Time Series Split). `Livro` *Forecasting: Principles and Practice* — Hyndman & Athanasopoulos (gratuito). https://otexts.com/fpp3/
- **ML Bayesiano e model averaging** — em vez de escolher um único modelo, combinar previsões ponderadas pela probabilidade posterior de cada um; conecta com inferência Bayesiana do mod. [01](01_matematica.md#13-probabilidade-e-estatística). `Livro` *Probabilistic Machine Learning* (vol. 1) — Kevin Murphy, cap. 18. https://probml.github.io/pml-book/

---

## Checklist de saída

- [ ] Sei explicar quando preferir XGBoost a uma rede neural (se não, revise "Por que isso importa").
- [ ] Implementei pelo menos um algoritmo from scratch além de regressão linear (se não, revise os Projetos 3.1 e 3.4).
- [ ] Sei desenhar um pipeline de ML do zero, com tratamento honesto de validação (se não, revise a seção 3.5 e o Projeto 3.2).
- [ ] Entendo SHAP/feature importance suficientemente para diagnosticar modelos (se não, revise a seção 3.5 e o Projeto 3.2).
- [ ] Reconheço sinais de overfitting e leakage em código alheio (se não, revise as seções 3.1 e 3.4, e "Erros comuns").
