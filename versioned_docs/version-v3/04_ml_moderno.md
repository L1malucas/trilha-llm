---
id: 04_ml_moderno
title: "Módulo 04 — Machine Learning Moderno"
sidebar_position: 17
---

# Módulo 04 — Machine Learning Moderno

> **Objetivo**: cobrir o que **não** é DL profundo nem ML clássico stricto sensu — paradigmas que emergiram entre os dois, e que são essenciais para entender LLMs (especialmente self-supervised learning) e problemas reais.
>
> **Pré-requisitos**: toda a trilha até aqui — em particular, self-supervised learning já praticado (masked language modeling no Projeto 7.3, Word2Vec no mod. 06), transfer learning (ResNet no Projeto 16.1, LoRA no mod. 09), e few-shot prompting (mod. 11). O mod. [03](03_ml_classico.md), que aprofunda árvores de decisão e gradient boosting (XGBoost, LightGBM), vem logo a seguir nesta trilha — o Projeto 4.1 usa essas bibliotecas de forma prática, com uma explicação mínima inline, antes do aprofundamento formal.
>
> **Tempo de referência**: 3–5 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar por que self-supervised learning não é a mesma coisa que unsupervised learning, com um exemplo de tarefa de pretexto.
- Explicar por que transfer learning funciona — o que exatamente é "transferido" entre tarefas.
- Conectar few-shot prompting (o que LLMs fazem em produção) com few-shot learning clássico.
- Explicar a diferença entre correlação e causa com um exemplo de confounding.
- Justificar por que um modelo com probabilidade calibrada é mais útil em produção que um só "mais acurado".

---

## Por que isso importa

A distinção "ML clássico vs Deep Learning" é didática mas incompleta. Existe um terreno intermediário rico — self-supervised learning (a base conceitual de todo pretraining de LLM, que você já praticou no Projeto 7.3), transfer learning (o princípio "treina uma vez, usa muitas" por trás de todo fine-tuning que você já fez), aprendizado em poucos exemplos, ensembles modernos, AutoML, e causal inference (para sair de "correlação" e ir a "causa"). Sem esse terreno intermediário mapeado, LLMs continuam parecendo mágica mesmo depois de você saber usá-los bem.

---

## 4.1 Self-Supervised Learning (SSL)

### Conceito
Criar tarefas de pretexto onde o **rótulo é derivado do próprio dado** — sem anotação humana. A tarefa é "fingir" que parte do dado é desconhecida e prevê-la.

Exemplos canônicos:
- **Mascarar palavras** numa frase e prever as escondidas (BERT — o que você já implementou no Projeto 7.3).
- **Prever a próxima palavra** dada a anterior (GPT — o `MiniLlama` do Projeto 8.3).
- **Prever rotação** de uma imagem.
- **Contrastive learning**: aproximar pares positivos, afastar negativos (SimCLR, MoCo, CLIP — você já usou CLIP nos mod. 16/18, e implementa SimCLR do zero no Projeto 4.2).

### Por que isso é o coração dos LLMs
LLMs modernos são **modelos auto-supervisionados em escala massiva**. Você já treinou dois deles (decoder-only e encoder-only) — este módulo formaliza o princípio geral por trás dos dois.

> **Intuição**: a diferença entre SSL e unsupervised learning tradicional (mod. [03](03_ml_classico.md), que vem a seguir) é sutil mas fundamental: unsupervised learning (k-means, PCA) não tem noção de "certo" ou "errado" — só busca estrutura. SSL **inventa** um problema supervisionado a partir do próprio dado: "esconda a última palavra da frase, tente prevê-la" tem uma resposta certa verificável (a palavra que realmente estava lá), mesmo sem nenhum humano ter rotulado nada — exatamente o `mascarar_tokens` do Projeto 7.3, ou o par `(x, y)` deslocado de um caractere do `get_batch` do Projeto 8.3. É esse "rótulo de graça, derivado da estrutura dos dados" que permite treinar em quantidades de texto que seriam impossíveis de rotular manualmente — todo o texto da internet, no caso de um LLM.
>
> **Exemplo resolvido — contrastive learning**: imagine duas versões da mesma foto de um cachorro (uma cortada, uma com cor alterada) — são um "par positivo": o modelo deve aprender a representá-las de forma parecida no espaço de embeddings. Uma foto de um gato é o "par negativo": deve ficar distante no espaço de embeddings da foto do cachorro. Repetindo isso em milhões de pares, o modelo aprende uma noção de "similaridade visual" sem nenhum humano ter dito "isto é um cachorro" — é assim que CLIP (mod. 16/18) aprende embeddings de imagem alinhados com texto, sem rotulagem manual de categoria. Você implementa essa mecânica do zero no Projeto 4.2.
>
> **Aplicação real**: next-token prediction (GPT, o que você já implementou) e masked language modeling (BERT, o que você já implementou) são as duas tarefas de pretexto que definem os dois grandes ramos de LLMs modernos (decoder-only vs encoder-only) — a escolha da tarefa de pretexto molda o que o modelo fica bom em fazer depois.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre SSL e unsupervised learning tradicional. Depois, explique por que "prever a próxima palavra" conta como uma tarefa supervisionada, mesmo sem rótulo humano.

---

## 4.2 Transfer Learning

### Conceito
Treinar em uma tarefa-fonte com muito dado, transferir para tarefa-alvo com pouco dado. **É o paradigma dominante** em DL moderno — o que você já fez toda vez que carregou um modelo pré-treinado (`ResNet50` no Projeto 16.1, `Qwen2.5-0.5B` desde o Projeto 9.1).

### Estratégias
- **Feature extraction**: congelar o backbone, treinar só a cabeça nova — o que você fez com `resnet.fc` no Projeto 16.1.
- **Fine-tuning**: descongelar parcialmente ou totalmente — o que LoRA (mod. 09) faz de forma parcial e eficiente.
- **Fine-tuning gradual** (gradual unfreezing).
- **Domain adaptation**: ajustar a distribuição-alvo — o mesmo princípio do continued pretraining (Projeto 9.3).

> **Intuição**: um modelo pré-treinado numa tarefa-fonte massiva (todo texto da internet, ou milhões de imagens do ImageNet) aprende representações internas genéricas e reutilizáveis — as primeiras camadas de uma CNN de visão aprendem bordas e texturas, úteis para qualquer tarefa visual; as camadas de um LLM aprendem sintaxe, semântica e conhecimento de mundo, úteis para qualquer tarefa de texto. Transfer learning reaproveita essas representações já aprendidas em vez de aprender do zero — é a diferença entre ensinar alguém que já sabe ler a ler um domínio técnico específico, versus ensinar alguém analfabeto a ler *e* entender o domínio técnico ao mesmo tempo. Feature extraction (congelar tudo, só treinar a cabeça) é apropriado quando a tarefa-alvo é parecida com a tarefa-fonte e há pouco dado; fine-tuning completo faz sentido quando há mais dado disponível e a tarefa-alvo é mais distinta.
>
> **Aplicação real**: todo fine-tuning de LLM (LoRA, QLoRA, full fine-tuning — mod. 09, já praticado) é transfer learning: o pré-treinamento massivo em texto genérico é a tarefa-fonte, e adaptar o modelo pra seguir instruções ou uma tarefa específica é a tarefa-alvo, feita com ordens de magnitude menos dado do que seria necessário do zero.
>
> **Checkpoint**: sem olhar o texto, explique por que as primeiras camadas de uma rede pré-treinada costumam ser mais reaproveitáveis entre tarefas do que as últimas.

---

## 4.3 Few-Shot e Meta-Learning

### Conceitos
- **Few-shot learning**: aprender com poucos exemplos por classe.
- **Zero-shot learning**: prever classes nunca vistas no treinamento — o que você já fez com CLIP no Projeto 16.6.
- **Meta-learning** ("aprender a aprender"): MAML (Model-Agnostic Meta-Learning), Prototypical Networks.

### Por que importa para LLMs
Few-shot prompting (mod. 11, já praticado extensivamente) é um caso emergente de few-shot learning sem gradient updates — comportamento "in-context" que aparece com escala.

> **Intuição — MAML e Prototypical Networks**: **MAML** não aprende a resolver uma tarefa específica, aprende um *ponto de partida* de pesos a partir do qual poucas iterações de gradient descent, em qualquer nova tarefa da mesma família, já convergem bem. **Prototypical Networks** são mais diretas: para cada classe nova (com poucos exemplos), calcula um "protótipo" (a média dos embeddings dos exemplos daquela classe) e classifica um ponto novo pela distância ao protótipo mais próximo — mais parecido com kNN (mod. 03) num espaço de embeddings aprendido do que com ajuste de gradiente por tarefa.
>
> **Intuição**: few-shot/meta-learning clássico (MAML, Prototypical Networks) ainda envolve *algum* ajuste de parâmetros — o modelo é explicitamente treinado para se adaptar rápido a partir de poucos exemplos. O que LLMs grandes fazem é mais surpreendente: dado alguns exemplos *no prompt* (sem nenhum gradient update, sem tocar nos pesos — o que você já explorou no Projeto 11.1), o modelo "aprende" o padrão só olhando o contexto — é literalmente inferência, não treinamento. Esse comportamento (in-context learning) não foi projetado explicitamente; ele emerge como consequência do pré-treinamento em escala e é um dos fenômenos mais estudados (e ainda parcialmente misteriosos) em LLMs modernos.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre few-shot learning clássico (MAML) e few-shot prompting num LLM — qual dos dois ajusta pesos?

---

## 4.4 Ensembles modernos e Stacking

### Além de Random Forest e XGBoost
- **Stacking**: meta-modelo aprende a combinar predições de modelos base — implementado no Projeto 4.1.
- **Blending**: variante simples de stacking com holdout.
- **Bayesian Model Averaging**.
- **Snapshot Ensembles** (em DL).

**Sobre gradient boosting (XGBoost, LightGBM, CatBoost), rapidamente**: são famílias de modelos que constroem uma sequência de árvores de decisão pequenas, cada nova árvore tentando corrigir especificamente os erros da soma das árvores anteriores (diferente de Random Forest, onde as árvores são treinadas independentemente e depois combinadas por média/voto). São, na prática, os modelos que mais vencem competições em dados tabulares — o mod. [03](03_ml_classico.md), logo a seguir nesta trilha, aprofunda o mecanismo exato. Aqui, no Projeto 4.1, você os usa como caixas relativamente prontas (`modelo.fit(X, y)`, `modelo.predict(X_novo)`) para montar um ensemble.

> Stacking generaliza a ideia de "combinar modelos" (mod. 03, bagging/boosting) um passo além: em vez de uma regra fixa de combinação (média, voto), um **meta-modelo aprende** a melhor forma de combinar as previsões dos modelos base — útil quando os modelos base têm pontos fortes complementares (um lida melhor com um subgrupo de casos, outro com outro). **Blending** é a mesma ideia com um procedimento mais simples que o `gerar_previsoes_out_of_fold` do Projeto 4.1: em vez de cross-validation, separa-se de antemão um único holdout para treinar o meta-modelo — mais rápido, mas usa os dados de forma menos eficiente. **Bayesian Model Averaging** troca o meta-modelo aprendido por uma combinação ponderada pela probabilidade posterior de cada modelo estar correto, mantendo a incerteza sobre qual modelo é o certo em vez de escolher um vencedor. **Snapshot Ensembles** obtêm o efeito de um ensemble a partir de um único treino de rede neural: um learning rate schedule cíclico faz o treino passar por vários mínimos locais diferentes, salvando um "snapshot" dos pesos em cada um — o mesmo `AdamW` que você já configura, só com um schedule cíclico em vez de decay monotônico.

---

## 4.5 AutoML e Neural Architecture Search (NAS)

### AutoML
- **Hyperparameter optimization**: random search, Bayesian optimization (Optuna, Hyperopt — usado no Projeto 4.3), Hyperband, BOHB (Bayesian Optimization and Hyperband).
- **AutoML frameworks**: AutoSklearn, Auto-Keras, FLAML, H2O AutoML.

### NAS
- **Differentiable NAS** (DARTS — Differentiable ARchiTecture Search).
- **Reinforcement-learning-based NAS**.
- **Evolutionary NAS**.

> **Intuição — busca de hiperparâmetros e NAS**: random search amostra combinações ao acaso — competitivo com grid search porque poucos hiperparâmetros costumam ser "críticos", e random search explora esses poucos com mais variedade do que uma grade fixa gastaria em combinações irrelevantes. A **otimização bayesiana** que o Optuna usa por padrão (Projeto 4.3) mantém um modelo probabilístico de "quão boa deve ser cada região ainda não testada" e escolhe o próximo ponto equilibrando exploração e aproveitamento — cada trial novo é informado por todos os anteriores. **Hyperband** aloca pouco orçamento a muitas configurações no início e mata cedo as que já parecem ruins, concentrando orçamento nas sobreviventes; **BOHB** combina essa alocação adaptativa com a inteligência da busca bayesiana. **NAS** aplica a mesma lógica de busca ao espaço de *arquiteturas*: evolutivo e baseado em RL tratam a arquitetura como algo a "descobrir" por busca; **DARTS** torna a busca diferenciável, relaxando a escolha discreta de operações numa camada para uma combinação contínua ponderada, otimizável por gradient descent como qualquer peso da rede — a mesma mecânica de otimização do mod. 01, aplicada aqui à própria estrutura da rede, não só aos pesos.

### Crítica honesta
NAS prometeu mais do que entregou para uso geral. Para a maioria dos casos, transferir uma arquitetura existente (seção 4.2) é melhor. Mas conhecer o conceito é importante.

---

## 4.6 Aprendizado com poucos rótulos

### Técnicas
- **Active learning**: o modelo escolhe quais exemplos rotular — implementado no Projeto 4.4.
- **Weak supervision**: labels imprecisos/programáticos (Snorkel).
- **Semi-supervised learning**: combinar rotulado + não-rotulado.
- **Pseudo-labeling**, **co-training**.
- **Data augmentation** como forma de "criar" supervisão — o mesmo princípio das augmentations do Projeto 4.2 e do Projeto 5.2.

> **Intuição — as outras técnicas de poucos rótulos**: **weak supervision** (Snorkel) substitui rótulos exatos por várias regras heurísticas imprecisas combinadas estatisticamente — cada regra sozinha é ruidosa, mas o sinal agregado costuma bastar para treinar um modelo sem rotulagem manual. **Semi-supervised learning** usa uma pequena porção rotulada junto com uma grande porção não-rotulada, assumindo que a estrutura dos dados não-rotulados carrega informação útil sobre a fronteira de decisão. **Pseudo-labeling** treina um modelo inicial nos poucos dados rotulados, usa esse modelo para "rotular" (com filtro de confiança) os dados não-rotulados, e re-treina incluindo esses pseudo-rótulos. **Co-training** usa duas visões diferentes dos mesmos dados que se ensinam mutuamente. **Data augmentation** "cria" supervisão sem dado novo de verdade — a mesma transformação (rotação, crop, color jitter) do `augmentation_simclr` do Projeto 4.2, assumindo que ela não muda o rótulo e forçando o modelo a aprender essa invariância.
>
> **Intuição**: active learning inverte a pergunta usual — em vez de "que modelo treinar com os dados que tenho", pergunta "quais exemplos, se rotulados, mais reduziriam a incerteza do modelo". Tipicamente, o modelo é usado pra identificar os casos em que está mais inseguro (probabilidade próxima de 50% numa classificação binária, por exemplo), e esses viram prioridade de rotulagem humana — extrai mais valor de um orçamento fixo de anotação do que rotular aleatoriamente.

---

## 4.7 Causal Inference (opcional, mas recomendado)

### Por que
Modelos preditivos respondem "qual é a probabilidade de Y dado X?" mas frequentemente queremos "o que acontece com Y se eu intervir em X?". Esses são problemas diferentes.

### Conceitos
- **Confounding** e variáveis de confundimento.
- **DAGs causais** (Directed Acyclic Graphs, Judea Pearl).
- **Counterfactuals**.
- **Propensity score matching** — implementado no Projeto 4.5.
- **Instrumental variables**.
- **Difference-in-differences**.

> **Intuição — confounding**: imagine observar que cidades com mais sorveterias têm mais afogamentos — correlação real nos dados. Seria um erro concluir que sorvete causa afogamento. A variável escondida (confounder) é a temperatura: dias quentes aumentam venda de sorvete *e* aumentam natação, que aumenta afogamentos. Um modelo puramente preditivo (mod. 03) captura a correlação perfeitamente bem e seria "acurado" prevendo afogamentos a partir de vendas de sorvete — mas seria inútil (e enganoso) para decidir uma intervenção ("vamos fechar sorveterias pra reduzir afogamentos?" não funcionaria). Causal inference existe exatamente pra distinguir essas duas perguntas.

### Conexão com IA
- Avaliação justa de modelos.
- Compreensão de viés algorítmico (mod. 14).
- Em RL (mod. 17): causalidade é fundamental.

> **Intuição — as outras ferramentas de causal inference**: um **DAG causal** desenha explicitamente as relações de causa assumidas (setas de causa → efeito) — no exemplo do sorvete, `temperatura → vendas` e `temperatura → afogamentos`, sem seta direta entre sorvete e afogamento. Um **counterfactual** é a pergunta central de toda causal inference: "o que teria acontecido com esta mesma pessoa se ela tivesse recebido o tratamento oposto?" — nunca observável diretamente, por isso todo método de causal inference estima esse contrafactual não-observado a partir de outros dados. **Propensity score matching**, implementado no Projeto 4.5, estima a probabilidade de cada indivíduo ter recebido o tratamento dado suas características, e compara resultados entre tratados e não-tratados com propensity score parecido. **Instrumental variables** resolvem o problema quando existe confounding não-observado (que propensity score não corrige): usam uma terceira variável que afeta o tratamento mas não o resultado, exceto através do tratamento. **Difference-in-differences** compara a *mudança* ao longo do tempo entre grupo tratado e controle, cancelando tendências que afetariam os dois igualmente mesmo sem o tratamento.
>
> **Checkpoint**: sem olhar o texto, explique o exemplo do sorvete/afogamento com suas próprias palavras — qual é o confounder, e por que ele engana um modelo puramente correlacional?

---

## 4.8 Modelos probabilísticos modernos

### Tópicos
- **Probabilistic Programming**: PyMC, Stan, NumPyro, Pyro.
- **Variational Inference** — o mesmo princípio por trás do VAE que você já implementou no Projeto 5.5.
- **Normalizing Flows** (relacionado à difusão do Projeto 19.3).
- **Bayesian Deep Learning**: dropout como Bayesian approximation, deep ensembles como uncertainty.

### Por que importa
- LLMs são fundamentalmente modelos probabilísticos sobre tokens — o `softmax` no final de todo forward pass do Projeto 8.3.
- Uncertainty quantification é o que diferencia "modelo bom" de "modelo confiável".

> **Intuição — as ferramentas de modelagem probabilística**: **Probabilistic Programming** (PyMC, Stan, NumPyro, Pyro) deixa você declarar um modelo probabilístico numa linguagem parecida com código normal, e a ferramenta cuida da inferência automaticamente por baixo. **Variational Inference** aproxima uma posterior impossível de calcular exatamente por uma distribuição mais simples, otimizando os parâmetros dessa aproximação por gradiente para minimizar a KL-divergência entre ela e a posterior real — exatamente o princípio do VAE que você já implementou no Projeto 5.5. **Normalizing Flows** constroem distribuições complexas e exatamente calculáveis (não só aproximadas) aplicando uma sequência de transformações invertíveis a uma distribuição simples — cada transformação "distorce" a distribuição, e por serem invertíveis, dá pra calcular a densidade exata via mudança de variável, uma ideia próxima da difusão do Projeto 19.3. **Bayesian Deep Learning** traz incerteza bayesiana pra redes neurais sem o custo de um treino totalmente bayesiano: dropout mantido ativo *durante a inferência* (não só no treino) aproxima amostrar de uma posterior sobre os pesos; **deep ensembles** treinam várias redes independentes e usam a discordância entre elas como proxy de incerteza.
>
> **Intuição**: um modelo que erra sabendo que está incerto é mais útil que um que erra com a mesma confiança de quando acerta — em produção, "não sei" é uma resposta valiosa que um modelo mal calibrado nunca dá. Uncertainty quantification (deep ensembles, dropout bayesiano) tenta extrair essa informação de "quão confiante devo estar nesta previsão" de modelos que, por padrão, sempre dão uma resposta com a mesma confiança aparente. Em LLMs, isso se relaciona diretamente com alucinação e calibração (mod. 14, já visto): um modelo "sabe" quando está extrapolando além do que aprendeu com confiança, na maioria dos casos, mas expressar essa incerteza de forma calibrada continua sendo um problema em aberto.

---

## Projetos práticos

### Projeto 4.1 — Stacking em problema tabular real

Você vai combinar 5 modelos heterogêneos num ensemble de stacking e comparar com cada modelo isolado, usando previsões out-of-fold para evitar vazamento de dado entre o treino dos modelos base e o treino do meta-modelo.

**Pré-requisitos**: `pip install xgboost lightgbm catboost scikit-learn`.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

modelos_base = {
    "xgboost": XGBClassifier(n_estimators=200, max_depth=4),
    "lightgbm": LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1),
    "catboost": CatBoostClassifier(n_estimators=200, depth=4, verbose=0),
    "logistic": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(n_neighbors=15),
}

def gerar_previsoes_out_of_fold(modelo, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    previsoes_oof = np.zeros(len(X))
    for idx_treino, idx_val in kf.split(X):
        modelo_fold = clone_modelo(modelo)  # uma cópia não treinada do modelo, para não vazar entre folds
        modelo_fold.fit(X[idx_treino], y[idx_treino])
        previsoes_oof[idx_val] = modelo_fold.predict_proba(X[idx_val])[:, 1]
    return previsoes_oof

from sklearn.base import clone as clone_modelo

previsoes_meta_treino = np.column_stack([
    gerar_previsoes_out_of_fold(modelo, X_treino, y_treino) for modelo in modelos_base.values()
])

meta_modelo = LogisticRegression()
meta_modelo.fit(previsoes_meta_treino, y_treino)
```

`gerar_previsoes_out_of_fold` é a peça que evita vazamento: se você treinasse cada modelo base no conjunto de treino inteiro e depois usasse as previsões *desse mesmo conjunto* para treinar o meta-modelo, o meta-modelo veria previsões "otimistas demais" (os modelos base já viram aqueles exemplos durante o próprio treino) — em vez disso, para cada exemplo, a previsão usada é sempre de um modelo que *não* viu aquele exemplo durante o treino daquele fold específico (o mesmo princípio de cross-validation honesta). O meta-modelo (`LogisticRegression`, simples de propósito) aprende então a combinar as 5 previsões — pode aprender, por exemplo, que XGBoost é mais confiável nalguma região do espaço de features e KNN noutra.

**Compare o stacking com cada modelo isolado** (treinado no conjunto de treino completo, avaliado no mesmo conjunto de teste) — o resultado esperado, mas não garantido, é que o stacking iguale ou supere levemente o melhor modelo individual; documente se isso se confirma no seu dataset, e se a complexidade adicional (5 modelos + 1 meta-modelo, em vez de 1) vale o ganho observado.

---

### Projeto 4.2 — SimCLR em CIFAR-10 (contrastive learning do zero)

Você vai implementar SimCLR — treinar um encoder de imagens sem nenhum rótulo, usando só pares de versões aumentadas da mesma imagem — e avaliar a qualidade das representações aprendidas via linear probing.

**Pré-requisitos**: os mesmos do Projeto 5.2 (CNN, CIFAR-10).

**1. Gere pares positivos via augmentation** — cada imagem do batch gera duas versões aumentadas diferentes:

```python
import torchvision.transforms as T

augmentation_simclr = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
])

def gerar_par_positivo(imagem_pil):
    return augmentation_simclr(imagem_pil), augmentation_simclr(imagem_pil)  # duas augmentations independentes da mesma imagem
```

**Antes de treinar, visualize alguns pares** — confirme visualmente que as duas versões aumentadas de uma imagem ainda "parecem" a mesma imagem para um humano (mesmo objeto reconhecível), e que negativos (imagens diferentes) de fato parecem diferentes. Um crop agressivo demais que corta o objeto principal da imagem produz um par positivo que atrapalha o treino tanto quanto um bug de código.

**2. Encoder + projeção, e a loss contrastiva (NT-Xent)**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLREncoder(nn.Module):
    def __init__(self, dim_representacao=128, dim_projecao=64):
        super().__init__()
        self.backbone = CNNSimples(usar_batchnorm=True)  # reaproveita o encoder do Projeto 5.2, sem a camada final de classificação
        self.backbone.fc = nn.Identity()  # remove a cabeça de classificação — só queremos as features
        self.projecao = nn.Sequential(nn.Linear(128 * 4 * 4, dim_representacao), nn.ReLU(), nn.Linear(dim_representacao, dim_projecao))

    def forward(self, x):
        h = self.backbone(x)
        z = self.projecao(h)
        return h, F.normalize(z, dim=-1)  # normaliza para usar produto interno como cosseno

def nt_xent_loss(z1, z2, temperatura=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2*batch_size vetores: pares (i, i+batch_size) são positivos entre si
    similaridades = z @ z.T / temperatura

    mascara_diagonal = torch.eye(2 * batch_size, dtype=torch.bool)
    similaridades.masked_fill_(mascara_diagonal, float("-inf"))  # um vetor não deve se comparar consigo mesmo

    alvos = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)])
    return F.cross_entropy(similaridades, alvos)
```

`nt_xent_loss` é a mesma ideia de contrastive learning descrita na Intuição da seção 4.1, formalizada como uma loss: para cada vetor, o "rótulo certo" (`alvos`) é a posição do seu par positivo entre todos os outros `2*batch_size - 1` vetores do batch — `cross_entropy` empurra a similaridade com o par positivo pra cima e com todos os outros (tratados implicitamente como negativos) pra baixo, exatamente o mesmo `F.cross_entropy` que você já usa desde o Projeto 8.3, só que aqui "classificando" qual vetor é o par certo, não qual token é o certo. `temperatura` controla o quão "afiada" fica essa distribuição — o mesmo papel da temperatura de sampling (mod. 10), aplicado aqui à loss de treino, não à geração.

**3. Treine sem nenhum rótulo**:

```python
encoder = SimCLREncoder()
optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4)

for epoca in range(50):
    for imagens, _ in loader_treino:  # o "_" é o rótulo do CIFAR-10 — deliberadamente ignorado
        x1 = torch.stack([augmentation_simclr(img) for img in imagens])
        x2 = torch.stack([augmentation_simclr(img) for img in imagens])
        _, z1 = encoder(x1)
        _, z2 = encoder(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**4. Avalie via linear probing**: congele `encoder.backbone` (já treinado, sem rótulos) e treine só uma nova camada linear em cima das features `h` (não `z` — a camada de projeção `projecao` é descartada depois do pré-treino contrastivo, usada só para estabilizar o treino da loss), agora *com* os rótulos do CIFAR-10:

```python
for param in encoder.backbone.parameters():
    param.requires_grad = False

classificador_linear = nn.Linear(128 * 4 * 4, 10)
optimizer_linear = torch.optim.AdamW(classificador_linear.parameters(), lr=1e-3)

for epoca in range(20):
    for imagens, rotulos in loader_treino:
        with torch.no_grad():
            h, _ = encoder(imagens)
        logits = classificador_linear(h)
        loss = F.cross_entropy(logits, rotulos)
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_linear.step()
```

**O que isso prova**: se a accuracy dessa camada linear simples (treinada em cima de features aprendidas *sem nenhum rótulo*) chegar razoavelmente perto da CNN totalmente supervisionada do Projeto 5.2, isso demonstra que self-supervision pode aprender representações boas o bastante para que uma tarefa supervisionada, depois, precise só de um classificador simples por cima — a essência de por que SSL é o motor do pretraining de LLMs em escala.

---

### Projeto 4.3 — Hyperparameter optimization sério com Optuna

Você vai otimizar os hiperparâmetros de um XGBoost com busca bayesiana (Optuna), comparando com random search e grid search no mesmo orçamento de tentativas.

**Pré-requisitos**: `pip install optuna xgboost matplotlib`.

```python
import optuna
from sklearn.model_selection import cross_val_score

def funcao_objetivo(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    modelo = XGBClassifier(**params)
    scores = cross_val_score(modelo, X_treino, y_treino, cv=5, scoring="accuracy")
    return scores.mean()

estudo = optuna.create_study(direction="maximize")
estudo.optimize(funcao_objetivo, n_trials=50)

print(f"Melhores hiperparâmetros: {estudo.best_params}")
print(f"Melhor score: {estudo.best_value:.4f}")
```

`trial.suggest_int`/`suggest_float` declaram o espaço de busca para cada hiperparâmetro (com `log=True` em `learning_rate` porque faz mais sentido buscar uniformemente em escala logarítmica para uma variável que varia em ordens de grandeza, o mesmo argumento do `plt.xscale("log")` do Projeto 8.4). Diferente de grid search (testa todas as combinações de uma grade fixa) ou random search (amostra aleatoriamente), Optuna usa otimização bayesiana por padrão: cada `trial` novo é escolhido considerando os resultados dos trials anteriores, focando a busca em regiões do espaço que parecem promissoras — em vez de gastar tentativas uniformemente por todo o espaço.

**Compare os três métodos com o mesmo orçamento** (50 tentativas cada): implemente random search (`trial.suggest_*` já usado dentro de um loop que não usa a lógica bayesiana — ou simplesmente `optuna.samplers.RandomSampler()`) e grid search (`optuna.samplers.GridSampler()` com uma grade pré-definida), e plote a evolução do melhor score encontrado até cada tentativa (`estudo.trials_dataframe()` tem essa informação) para os três — a expectativa é que a busca bayesiana atinja bons resultados com menos tentativas, por focar a busca de forma mais inteligente.

---

### Projeto 4.4 — Active learning loop

Você vai implementar um loop de active learning e comparar a curva de aprendizado contra amostragem aleatória, no mesmo orçamento de rótulos.

**Pré-requisitos**: `pip install scikit-learn matplotlib`, um dataset com bastante dado disponível (você vai "esconder" a maioria dos rótulos e revelá-los progressivamente).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def active_learning_loop(X_pool, y_pool, X_teste, y_teste, n_inicial=100, n_por_rodada=50, n_rodadas=10):
    indices_rotulados = list(np.random.choice(len(X_pool), n_inicial, replace=False))
    indices_disponiveis = list(set(range(len(X_pool))) - set(indices_rotulados))
    historico_accuracy = []

    for rodada in range(n_rodadas):
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_pool[indices_rotulados], y_pool[indices_rotulados])
        acc = accuracy_score(y_teste, modelo.predict(X_teste))
        historico_accuracy.append(acc)

        probs = modelo.predict_proba(X_pool[indices_disponiveis])
        incerteza = 1 - probs.max(axis=1)  # quanto mais perto de 50/50, mais incerto o modelo está
        indices_mais_incertos = np.array(indices_disponiveis)[np.argsort(-incerteza)[:n_por_rodada]]

        indices_rotulados.extend(indices_mais_incertos.tolist())
        indices_disponiveis = list(set(indices_disponiveis) - set(indices_mais_incertos.tolist()))

    return historico_accuracy

def random_sampling_loop(X_pool, y_pool, X_teste, y_teste, n_inicial=100, n_por_rodada=50, n_rodadas=10):
    indices_rotulados = list(np.random.choice(len(X_pool), n_inicial, replace=False))
    indices_disponiveis = list(set(range(len(X_pool))) - set(indices_rotulados))
    historico_accuracy = []

    for rodada in range(n_rodadas):
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_pool[indices_rotulados], y_pool[indices_rotulados])
        historico_accuracy.append(accuracy_score(y_teste, modelo.predict(X_teste)))

        novos_indices = list(np.random.choice(indices_disponiveis, n_por_rodada, replace=False))
        indices_rotulados.extend(novos_indices)
        indices_disponiveis = list(set(indices_disponiveis) - set(novos_indices))

    return historico_accuracy
```

A única diferença entre as duas funções é como os próximos `n_por_rodada` exemplos são escolhidos: `active_learning_loop` usa `incerteza = 1 - probs.max(axis=1)` (a Intuição da seção 4.6, em código — quanto mais próxima de uniforme a distribuição de probabilidade prevista, mais incerto o modelo está naquele exemplo) para escolher os exemplos onde o modelo mais "hesita"; `random_sampling_loop` escolhe uniformemente ao acaso.

**Plote as duas curvas** (`historico_accuracy` de cada função, no eixo X o número de exemplos rotulados acumulados) no mesmo gráfico — a curva de active learning deve subir mais rápido (atingir uma accuracy dada com menos exemplos rotulados) que a de random sampling, e essa diferença visual é a evidência concreta do valor de active learning, não só um argumento teórico.

---

### Projeto 4.5 (opcional) — Análise causal simples

Você vai estimar o efeito de um tratamento usando duas abordagens — uma ingênua (regressão OLS direta) e uma que corrige por confounding (propensity score matching) — e entender por que os resultados diferem.

**Pré-requisitos**: `pip install statsmodels scikit-learn pandas`, o dataset Lalonde (clássico em causal inference, disponível via `pip install causaldata` ou como CSV público).

```python
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

df = pd.read_csv("lalonde.csv")  # colunas típicas: treat, age, educ, race, married, re74, re75, re78 (renda em 1978, o "Y")

# 1. Estimativa ingênua: OLS direto, comparando quem recebeu o tratamento (treat=1) com quem não recebeu
modelo_ols = smf.ols("re78 ~ treat + age + educ + married", data=df).fit()
print("Efeito estimado (OLS ingênuo):", modelo_ols.params["treat"])

# 2. Propensity score: probabilidade de ter recebido o tratamento, dado as covariáveis observadas
covariaveis = ["age", "educ", "married", "re74", "re75"]
modelo_propensity = LogisticRegression(max_iter=1000)
modelo_propensity.fit(df[covariaveis], df["treat"])
df["propensity_score"] = modelo_propensity.predict_proba(df[covariaveis])[:, 1]

# 3. Matching: para cada indivíduo tratado, encontra o não-tratado com propensity score mais parecido
tratados = df[df["treat"] == 1]
controles = df[df["treat"] == 0]
nn = NearestNeighbors(n_neighbors=1).fit(controles[["propensity_score"]])
_, indices_pareados = nn.kneighbors(tratados[["propensity_score"]])

efeito_matching = (tratados["re78"].values - controles.iloc[indices_pareados.flatten()]["re78"].values).mean()
print("Efeito estimado (propensity score matching):", efeito_matching)
```

O `propensity_score` (seção 4.7) resume "quão provável era esse indivíduo ter sido tratado, dado suas características observáveis" — parear cada tratado com um controle de propensity score parecido simula, aproximadamente, um experimento controlado: comparando pessoas que *pareciam* igualmente prováveis de receber o tratamento, mas uma recebeu e a outra não, o efeito estimado fica menos contaminado pelo confounding (pessoas com características muito diferentes sistematicamente entre grupos tratado/controle) do que a regressão OLS ingênua, que só "controla" por covariáveis de forma linear e global.

**Compare os dois números** (`modelo_ols.params["treat"]` vs `efeito_matching`) — no dataset Lalonde, essa diferença é historicamente grande e didática: o experimento original (um estudo com atribuição aleatória real do tratamento) serve de referência, e o objetivo do exercício é entender por que a estimativa observacional ingênua (OLS) se desvia tanto dela, enquanto matching (ou outras técnicas de causal inference) se aproxima mais.

---

## Erros comuns

- **Confundir SSL com unsupervised learning.** SSL gera supervisão sintética; unsupervised tradicional não.
- **Achar que ensembles sempre ajudam.** Em modelos já bem ajustados, o ganho marginal pode não compensar a complexidade (documente isso no Projeto 4.1).
- **AutoML como muleta.** É ferramenta de busca, não substitui entendimento.
- **Ignorar incerteza no output.** Modelo que dá probabilidade calibrada é muito mais útil em produção.
- **Vazamento de dado no meta-modelo de stacking.** Sem previsões out-of-fold (Projeto 4.1), o meta-modelo aprende a "confiar" demais nos modelos base.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Self-supervised learning | Pretraining de LLMs (já praticado desde o mod. 08) |
| Transfer learning | Fine-tuning de LLMs (já praticado no mod. 09) |
| Contrastive learning | CLIP, embeddings (já praticado nos mod. 12, 16, 18) |
| Few-shot | In-context learning (já praticado no mod. 11) |
| Probabilistic models | Sampling em LLMs (mod. 10) |
| Uncertainty | Hallucination, calibração (mod. 14, já visto) |

---

## Saiba mais

Alguns tópicos deste módulo foram citados sem profundidade — grandes demais para caber aqui sem desviar do fluxo principal:

- **BYOL e métodos contrastivos sem negativos** (4.1) — SimCLR/MoCo (que você implementou no Projeto 4.2) precisam de pares negativos explícitos; BYOL (Bootstrap Your Own Latent) aprende representações contrastivas sem eles, usando uma rede "professora" atualizada por média móvel. `Paper` **Bootstrap Your Own Latent** — Grill et al. (2020).
- **Gradual unfreezing e domain adaptation a fundo** (4.2) — estratégias mais refinadas de transfer learning para quando a tarefa-alvo é bem diferente da tarefa-fonte. `Paper` **ULMFiT** — Howard & Ruder (2018).
- **AutoML frameworks completos** (4.5) — AutoSklearn, FLAML, H2O AutoML automatizam não só hiperparâmetros mas a escolha do próprio algoritmo; vale a pena só depois de já saber fazer manualmente o que essas ferramentas fazem por baixo, como no Projeto 4.3. `Ferramenta` **FLAML Documentation**. https://microsoft.github.io/FLAML/
- **Modelos probabilísticos além do escopo aqui** (4.8) — processos de Dirichlet, campos aleatórios de Markov, inferência exata em modelos gráficos pequenos. `Livro` *Probabilistic Machine Learning* (vol. 2) — Kevin Murphy.

---

## Checklist de saída

- [ ] Sei explicar a diferença entre supervised, unsupervised e self-supervised, com exemplos próprios já implementados (se não, revise a seção 4.1 e o Projeto 7.3).
- [ ] Entendo intuitivamente por que pretraining em escala funciona (se não, revise a seção 4.2).
- [ ] Implementei SimCLR do zero e validei com linear probing (se não, revise o Projeto 4.2).
- [ ] Sei usar Optuna para HPO em problema real, e sei por que a busca bayesiana costuma vencer grid/random search (se não, revise o Projeto 4.3).
- [ ] Implementei um loop de active learning e comparei a curva com random sampling (se não, revise o Projeto 4.4).
- [ ] Entendo o que é confounding e por que correlação ≠ causação importa, com um exemplo numérico próprio (se não, revise a seção 4.7 e o Projeto 4.5).
