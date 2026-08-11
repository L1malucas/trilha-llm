---
id: 04_ml_moderno
title: "Módulo 04 — Machine Learning Moderno"
sidebar_position: 4
---

# Módulo 04 — Machine Learning Moderno

> **Objetivo**: cobrir o que **não** é DL profundo nem ML clássico stricto sensu — paradigmas que emergiram entre os dois, e que são essenciais para entender LLMs (especialmente self-supervised learning) e problemas reais.
>
> **Pré-requisitos**: Módulo [03](03_ml_classico.md).
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

A distinção "ML clássico vs Deep Learning" é didática mas incompleta. Existe um terreno intermediário rico:

- **Self-supervised learning** — base conceitual do pretraining de LLMs.
- **Transfer learning** — princípio de "treina uma vez, usa muitas".
- **Aprendizado em poucos exemplos** (few-shot, meta-learning).
- **Ensemble e stacking modernos**.
- **AutoML e Neural Architecture Search**.
- **Causal Inference** — para sair de "correlação" e ir a "causa".

Sem isso, você usa LLMs como mágica e não entende por que eles funcionam.

---

## 4.1 Self-Supervised Learning (SSL)

### Conceito
Criar tarefas de pretexto onde o **rótulo é derivado do próprio dado** — sem anotação humana. A tarefa é "fingir" que parte do dado é desconhecida e prevê-la.

Exemplos canônicos:
- **Mascarar palavras** numa frase e prever as escondidas (BERT).
- **Prever a próxima palavra** dada a anterior (GPT).
- **Prever rotação** de uma imagem.
- **Contrastive learning**: aproximar pares positivos, afastar negativos (SimCLR, MoCo, CLIP).

### Por que isso é o coração dos LLMs
LLMs modernos são **modelos auto-supervisionados em escala massiva**. Entender SSL é entender o pretraining.

> **Intuição**: a diferença entre SSL e unsupervised learning tradicional (mod. [03](03_ml_classico.md#33-algoritmos-não-supervisionados)) é sutil mas fundamental: unsupervised learning (k-means, PCA) não tem noção de "certo" ou "errado" — só busca estrutura. SSL **inventa** um problema supervisionado a partir do próprio dado: "esconda a última palavra da frase, tente prevê-la" tem uma resposta certa verificável (a palavra que realmente estava lá), mesmo sem nenhum humano ter rotulado nada. É esse "rótulo de graça, derivado da estrutura dos dados" que permite treinar em quantidades de texto que seriam impossíveis de rotular manualmente — todo o texto da internet, no caso de um LLM.
>
> **Exemplo resolvido — contrastive learning**: imagine duas versões da mesma foto de um cachorro (uma cortada, uma com cor alterada) — são um "par positivo": o modelo deve aprender a representá-las de forma parecida no espaço de embeddings. Uma foto de um gato é o "par negativo": deve ficar distante no espaço de embeddings da foto do cachorro. Repetindo isso em milhões de pares, o modelo aprende uma noção de "similaridade visual" sem nenhum humano ter dito "isto é um cachorro" — é assim que CLIP (mod. [18](18_multimodal.mdx)) aprende embeddings de imagem alinhados com texto, sem rotulagem manual de categoria.
>
> **Aplicação real**: next-token prediction (GPT) e masked language modeling (BERT) são as duas tarefas de pretexto que definem os dois grandes ramos de LLMs modernos (decoder-only vs encoder-only, mod. [07](07_transformers.mdx#74-arquiteturas-de-transformer)) — a escolha da tarefa de pretexto molda o que o modelo fica bom em fazer depois.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre SSL e unsupervised learning tradicional. Depois, explique por que "prever a próxima palavra" conta como uma tarefa supervisionada, mesmo sem rótulo humano.

### Papers fundamentais
- `Paper` **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** — Chen et al. (2020). https://arxiv.org/abs/2002.05709
- `Paper` **Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)** — He et al. (2019). https://arxiv.org/abs/1911.05722
- `Paper` **Self-Supervised Learning: Generative or Contrastive** — Liu et al. (2020). https://arxiv.org/abs/2006.08218 (survey)
- `Paper` **Bootstrap Your Own Latent (BYOL)** — Grill et al. (2020). https://arxiv.org/abs/2006.07733

### Cursos
- `Curso` **NYU Deep Learning (Yann LeCun & Alfredo Canziani)** — aulas dedicadas a SSL. https://atcold.github.io/NYU-DLSP21/

---

## 4.2 Transfer Learning

### Conceito
Treinar em uma tarefa-fonte com muito dado, transferir para tarefa-alvo com pouco dado. **É o paradigma dominante** em DL moderno.

### Estratégias
- **Feature extraction**: congelar o backbone, treinar só a cabeça nova.
- **Fine-tuning**: descongelar parcialmente ou totalmente.
- **Fine-tuning gradual** (gradual unfreezing).
- **Domain adaptation**: ajustar a distribuição-alvo.

> **Intuição**: um modelo pré-treinado numa tarefa-fonte massiva (todo texto da internet, ou milhões de imagens do ImageNet) aprende representações internas genéricas e reutilizáveis — as primeiras camadas de uma CNN de visão aprendem bordas e texturas, úteis para qualquer tarefa visual; as camadas de um LLM aprendem sintaxe, semântica e conhecimento de mundo, úteis para qualquer tarefa de texto. Transfer learning reaproveita essas representações já aprendidas em vez de aprender do zero — é a diferença entre ensinar alguém que já sabe ler a ler um domínio técnico específico, versus ensinar alguém analfabeto a ler *e* entender o domínio técnico ao mesmo tempo. Feature extraction (congelar tudo, só treinar a cabeça) é apropriado quando a tarefa-alvo é parecida com a tarefa-fonte e há pouco dado; fine-tuning completo faz sentido quando há mais dado disponível e a tarefa-alvo é mais distinta.
>
> **Aplicação real**: todo fine-tuning de LLM (LoRA, QLoRA, full fine-tuning — mod. [09](09_treinamento_e_alinhamento.mdx)) é transfer learning: o pré-treinamento massivo em texto genérico é a tarefa-fonte, e adaptar o modelo pra seguir instruções ou uma tarefa específica é a tarefa-alvo, feita com ordens de magnitude menos dado do que seria necessário do zero.
>
> **Checkpoint**: sem olhar o texto, explique por que as primeiras camadas de uma rede pré-treinada costumam ser mais reaproveitáveis entre tarefas do que as últimas.

### Referências
- `Paper` **A Survey on Transfer Learning** — Pan & Yang (2010). https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf
- `Paper` **Universal Language Model Fine-tuning (ULMFiT)** — Howard & Ruder (2018). https://arxiv.org/abs/1801.06146 (clássico em NLP)

---

## 4.3 Few-Shot e Meta-Learning

### Conceitos
- **Few-shot learning**: aprender com poucos exemplos por classe.
- **Zero-shot learning**: prever classes nunca vistas no treinamento.
- **Meta-learning** ("aprender a aprender"): MAML (Model-Agnostic Meta-Learning), Prototypical Networks.

### Por que importa para LLMs
Few-shot prompting (mod. [11](11_prompt_engineering.md)) é um caso emergente de few-shot learning sem gradient updates — comportamento "in-context" que aparece com escala.

> **Intuição — MAML e Prototypical Networks**: **MAML** não aprende a resolver uma tarefa específica, aprende um *ponto de partida* de pesos a partir do qual poucas iterações de gradient descent, em qualquer nova tarefa da mesma família, já convergem bem — treina explicitamente para "estar a poucos passos de gradiente de qualquer tarefa", simulando esse cenário repetidamente durante o próprio treino. **Prototypical Networks** são mais diretas: para cada classe nova (com poucos exemplos), calcula um "protótipo" (a média dos embeddings dos exemplos daquela classe) e classifica um ponto novo pela distância ao protótipo mais próximo — mais parecido com kNN (mod. [03](03_ml_classico.md#32-algoritmos-supervisionados)) num espaço de embeddings aprendido do que com ajuste de gradiente por tarefa.
>
> **Intuição**: few-shot/meta-learning clássico (MAML, Prototypical Networks) ainda envolve *algum* ajuste de parâmetros — o modelo é explicitamente treinado para se adaptar rápido a partir de poucos exemplos. O que LLMs grandes fazem é mais surpreendente: dado alguns exemplos *no prompt* (sem nenhum gradient update, sem tocar nos pesos), o modelo "aprende" o padrão só olhando o contexto — é literalmente inferência, não treinamento. Esse comportamento (in-context learning) não foi projetado explicitamente; ele emerge como consequência do pré-treinamento em escala e é um dos fenômenos mais estudados (e ainda parcialmente misteriosos) em LLMs modernos.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre few-shot learning clássico (MAML) e few-shot prompting num LLM — qual dos dois ajusta pesos?

### Papers
- `Paper` **Model-Agnostic Meta-Learning (MAML)** — Finn et al. (2017). https://arxiv.org/abs/1703.03400
- `Paper` **Matching Networks for One Shot Learning** — Vinyals et al. (2016). https://arxiv.org/abs/1606.04080
- `Paper` **Prototypical Networks for Few-shot Learning** — Snell et al. (2017). https://arxiv.org/abs/1703.05175

---

## 4.4 Ensembles modernos e Stacking

### Além de Random Forest e XGBoost
- **Stacking**: meta-modelo aprende a combinar predições de modelos base.
- **Blending**: variante simples de stacking com holdout.
- **Bayesian Model Averaging**.
- **Snapshot Ensembles** (em DL).

> Stacking generaliza a ideia de "combinar modelos" (mod. [03](03_ml_classico.md#32-algoritmos-supervisionados), bagging/boosting) um passo além: em vez de uma regra fixa de combinação (média, voto), um **meta-modelo aprende** a melhor forma de combinar as previsões dos modelos base — útil quando os modelos base têm pontos fortes complementares (um lida melhor com um subgrupo de casos, outro com outro). **Blending** é a mesma ideia com um procedimento mais simples: em vez de gerar previsões out-of-fold via cross-validation para treinar o meta-modelo, separa-se de antemão um único holdout para essa finalidade — mais rápido de implementar, mas usa os dados de forma menos eficiente. **Bayesian Model Averaging** troca o meta-modelo aprendido por uma combinação ponderada pela probabilidade posterior de cada modelo estar correto (dado os dados observados) — em vez de escolher um vencedor, mantém a incerteza sobre qual modelo é o certo. **Snapshot Ensembles** obtêm o efeito de um ensemble a partir de um único treino de rede neural: usando um learning rate schedule cíclico (mod. [01](01_matematica.md#14-otimização)), o treino passa por vários mínimos locais diferentes, e um "snapshot" dos pesos é salvo em cada um — combinando esses snapshots como se fossem modelos independentes, sem o custo de treinar vários do zero.

### Referências
- `Livro` **Ensemble Methods: Foundations and Algorithms** — Zhou.
- `Paper` **Stacked Generalization** — Wolpert (1992). https://www.researchgate.net/publication/222467943_Stacked_Generalization

---

## 4.5 AutoML e Neural Architecture Search (NAS)

### AutoML
- **Hyperparameter optimization**: random search, Bayesian optimization (Optuna, Hyperopt), Hyperband, BOHB (Bayesian Optimization and Hyperband).
- **AutoML frameworks**: AutoSklearn, Auto-Keras, FLAML, H2O AutoML.

### NAS
- **Differentiable NAS** (DARTS — Differentiable ARchiTecture Search).
- **Reinforcement-learning-based NAS**.
- **Evolutionary NAS**.

> **Intuição — busca de hiperparâmetros e NAS**: **random search** amostra combinações de hiperparâmetros ao acaso — surpreendentemente competitivo com grid search, porque a maioria dos hiperparâmetros importa muito menos que um ou dois "críticos", e random search explora esses poucos importantes com mais variedade do que uma grade fixa gastaria com combinações de parâmetros irrelevantes. **Otimização bayesiana** (Optuna, Hyperopt) vai além: mantém um modelo probabilístico de "quão boa deve ser cada região do espaço de hiperparâmetros ainda não testada", e escolhe o próximo ponto a testar equilibrando explorar regiões incertas e explorar regiões já promissoras — cada tentativa nova é informada por todas as anteriores, ao contrário de random/grid search. **Hyperband** ataca um problema diferente: em vez de rodar cada configuração até o fim, aloca pouco orçamento a muitas configurações no início e mata cedo as que já parecem ruins, concentrando o orçamento nas sobreviventes — **BOHB** combina essa alocação adaptativa de orçamento com a inteligência da busca bayesiana na escolha de quais configurações testar. **NAS** aplica a mesma lógica de busca, mas ao espaço de *arquiteturas* em vez de hiperparâmetros: **NAS evolutivo** e **baseado em RL** tratam a arquitetura como algo a ser "descoberto" por busca ou por um agente que aprende a propor arquiteturas melhores; **DARTS** torna a busca diferenciável — em vez de testar arquiteturas discretas uma a uma (caríssimo), relaxa a escolha de operações numa camada para uma combinação contínua ponderada, otimizável por gradient descent como qualquer peso da rede, e discretiza só no final.

### Crítica honesta
NAS prometeu mais do que entregou para uso geral. Para a maioria dos casos, transferir uma arquitetura existente é melhor. Mas conhecer o conceito é importante.

### Referências
- `Paper` **Neural Architecture Search: A Survey** — Elsken et al. (2018). https://arxiv.org/abs/1808.05377
- `Paper` **DARTS: Differentiable Architecture Search** — Liu et al. (2018). https://arxiv.org/abs/1806.09055
- `Ferramenta` **Optuna**. https://optuna.org/

---

## 4.6 Aprendizado com poucos rótulos

### Técnicas
- **Active learning**: o modelo escolhe quais exemplos rotular.
- **Weak supervision**: labels imprecisos/programáticos (Snorkel).
- **Semi-supervised learning**: combinar rotulado + não-rotulado.
- **Pseudo-labeling**, **co-training**.
- **Data augmentation** como forma de "criar" supervisão.

> **Intuição — as outras técnicas de poucos rótulos**: **weak supervision** (Snorkel) substitui rótulos exatos por várias regras heurísticas imprecisas ("se a mensagem contém a palavra 'grátis', provavelmente é spam") combinadas estatisticamente — cada regra sozinha é ruidosa e às vezes conflita com as outras, mas combinando muitas o sinal agregado costuma ser bom o bastante para treinar um modelo, sem exigir nenhuma rotulagem manual. **Semi-supervised learning** usa uma pequena porção rotulada junto com uma grande porção não-rotulada — a suposição por trás é que a estrutura dos dados não-rotulados (onde eles se agrupam, mod. [03](03_ml_classico.md#33-algoritmos-não-supervisionados)) carrega informação útil sobre a fronteira de decisão, mesmo sem rótulo. **Pseudo-labeling** é uma forma concreta disso: treina um modelo inicial nos poucos dados rotulados, usa esse modelo para "rotular" (com algum filtro de confiança) os dados não-rotulados, e re-treina incluindo esses pseudo-rótulos como se fossem reais. **Co-training** usa duas visões diferentes dos mesmos dados (duas features distintas, ou dois modelos com viés diferente) que se ensinam mutuamente: cada modelo rotula os exemplos em que está mais confiante para o outro modelo treinar. **Data augmentation** (rotações, crops, ruído) "cria" supervisão adicional sem dado novo de verdade — assume que a transformação não muda o rótulo (uma foto de gato rotacionada continua sendo um gato), e força o modelo a aprender essa invariância explicitamente.
>
> **Intuição**: active learning inverte a pergunta usual — em vez de "que modelo treinar com os dados que tenho", pergunta "quais exemplos, se rotulados, mais reduziriam a incerteza do modelo". Tipicamente, o modelo é usado pra identificar os casos em que está mais inseguro (probabilidade próxima de 50% numa classificação binária, por exemplo), e esses viram prioridade de rotulagem humana — extrai mais valor de um orçamento fixo de anotação do que rotular aleatoriamente.

### Referências
- `Paper` **Snorkel: Rapid Training Data Creation with Weak Supervision** — Ratner et al. (2017). https://arxiv.org/abs/1711.10160
- `Livro` **Active Learning** — Burr Settles. http://burrsettles.com/pub/settles.activelearning.pdf

---

## 4.7 Causal Inference (opcional, mas recomendado)

### Por que
Modelos preditivos respondem "qual é a probabilidade de Y dado X?" mas frequentemente queremos "o que acontece com Y se eu intervir em X?". Esses são problemas diferentes.

### Conceitos
- **Confounding** e variáveis de confundimento.
- **DAGs causais** (Directed Acyclic Graphs, Judea Pearl).
- **Counterfactuals**.
- **Propensity score matching**.
- **Instrumental variables**.
- **Difference-in-differences**.

> **Intuição — confounding**: imagine observar que cidades com mais sorveterias têm mais afogamentos — correlação real nos dados. Seria um erro concluir que sorvete causa afogamento. A variável escondida (confounder) é a temperatura: dias quentes aumentam venda de sorvete *e* aumentam natação, que aumenta afogamentos. Um modelo puramente preditivo (mod. [03](03_ml_classico.md)) captura a correlação perfeitamente bem e seria "acurado" prevendo afogamentos a partir de vendas de sorvete — mas seria inútil (e enganoso) para decidir uma intervenção ("vamos fechar sorveterias pra reduzir afogamentos?" não funcionaria). Causal inference existe exatamente pra distinguir essas duas perguntas.

### Conexão com IA
- Avaliação justa de modelos.
- Compreensão de viés algorítmico.
- Em RL: causalidade é fundamental.

> **Intuição — as outras ferramentas de causal inference**: um **DAG causal** desenha explicitamente as relações de causa assumidas entre variáveis (setas de causa → efeito) — no exemplo do sorvete, `temperatura → vendas de sorvete` e `temperatura → afogamentos`, sem seta direta entre sorvete e afogamento; desenhar o DAG é o que permite identificar formalmente quais variáveis precisam ser "controladas" para isolar um efeito causal. Um **counterfactual** é a pergunta contrafactual central de toda causal inference: "o que teria acontecido com *esta mesma pessoa específica* se ela tivesse recebido o tratamento oposto?" — nunca observável diretamente (uma pessoa não pode simultaneamente tomar e não tomar um remédio), por isso todo método de causal inference é, no fundo, uma forma de estimar esse contrafactual não-observado a partir de outros dados. **Propensity score matching** estima a probabilidade de cada indivíduo ter recebido o tratamento dado suas características, e compara resultados entre tratados e não-tratados com propensity score parecido — aproximando um experimento controlado a partir de dados observacionais. **Instrumental variables** resolvem o problema quando existe confounding não-observado (que propensity score não corrige, já que só usa variáveis observadas): usam uma terceira variável (o "instrumento") que afeta o tratamento mas não afeta o resultado diretamente, exceto através do tratamento — isolando a variação "como se fosse aleatória" no tratamento. **Difference-in-differences** compara a *mudança* ao longo do tempo entre um grupo tratado e um grupo controle (em vez da diferença num único momento), cancelando tendências que afetariam os dois grupos igualmente mesmo sem o tratamento.
>
> **Checkpoint**: sem olhar o texto, explique o exemplo do sorvete/afogamento com suas próprias palavras — qual é o confounder, e por que ele engana um modelo puramente correlacional?

### Referências
- `Livro` **Causal Inference: The Mixtape** — Cunningham (gratuito). https://mixtape.scunning.com/
- `Livro` **The Book of Why** — Judea Pearl (acessível).
- `Livro` **Causal Inference: What If** — Hernán & Robins (gratuito). https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

---

## 4.8 Modelos probabilísticos modernos

### Tópicos
- **Probabilistic Programming**: PyMC, Stan, NumPyro, Pyro.
- **Variational Inference**.
- **Normalizing Flows** (preparação para módulo [19](19_topicos_avancados.md)).
- **Bayesian Deep Learning**: dropout como Bayesian approximation, deep ensembles como uncertainty.

### Por que importa
- LLMs são fundamentalmente modelos probabilísticos sobre tokens.
- Uncertainty quantification é o que diferencia "modelo bom" de "modelo confiável".

> **Intuição — as ferramentas de modelagem probabilística**: **Probabilistic Programming** (PyMC, Stan, NumPyro, Pyro) deixa você declarar um modelo probabilístico (priors, distribuições, dependências entre variáveis) numa linguagem parecida com código normal, e a ferramenta cuida da inferência — amostrar da distribuição posterior — automaticamente por baixo, sem você precisar derivar as fórmulas de atualização bayesiana à mão. **Variational Inference** é uma das formas de fazer essa inferência quando a posterior exata é impossível de calcular (o caso comum): em vez de calcular a posterior exata, aproxima por uma distribuição mais simples (ex.: Gaussiana) e otimiza os parâmetros dessa aproximação por gradiente, minimizando a KL-divergência (mod. [01](01_matematica.md#13-probabilidade-e-estatística)) entre a aproximação e a posterior real — exatamente o princípio de treino de um VAE (Variational Autoencoder). **Normalizing Flows** constroem distribuições complexas e ainda assim exatamente calculáveis (não só aproximadas) aplicando uma sequência de transformações invertíveis a uma distribuição simples (ex.: Gaussiana) — cada transformação "distorce" a distribuição um pouco, e por serem invertíveis, dá pra calcular a densidade exata de probabilidade do resultado final via mudança de variável, não só amostrar dele. **Bayesian Deep Learning** tenta trazer incerteza bayesiana pra redes neurais sem o custo de um treino totalmente bayesiano: **dropout como aproximação bayesiana** interpreta manter o dropout ativo *durante a inferência* (não só no treino) e rodar várias previsões como uma aproximação barata de amostrar de uma distribuição posterior sobre os pesos; **deep ensembles** treinam várias redes independentes (com seeds/inicializações diferentes) e usam a discordância entre elas como proxy de incerteza — quanto mais as previsões divergem entre os membros do ensemble, menos confiável é a previsão média.
>
> **Intuição**: um modelo que erra sabendo que está incerto é mais útil que um que erra com a mesma confiança de quando acerta — em produção, "não sei" é uma resposta valiosa que um modelo mal calibrado nunca dá. Uncertainty quantification (deep ensembles, dropout bayesiano) tenta extrair essa informação de "quão confiante devo estar nesta previsão" de modelos que, por padrão, sempre dão uma resposta com a mesma confiança aparente. Em LLMs, isso se relaciona diretamente com alucinação (mod. [14](14_avaliacao_e_seguranca.md)): um modelo "sabe" quando está extrapolando além do que aprendeu com confiança, na maioria dos casos, mas expressar essa incerteza de forma calibrada continua sendo um problema em aberto.

### Referências
- `Livro` **Probabilistic Machine Learning (vol. 2: Advanced Topics)** — Murphy. https://probml.github.io/pml-book/
- `Paper` **Auto-Encoding Variational Bayes (VAE, Variational Autoencoder)** — Kingma & Welling (2013). https://arxiv.org/abs/1312.6114

---

## Projetos práticos

### Projeto 4.1 — Stacking em problema tabular real
- Combine 5 modelos heterogêneos (XGBoost, LightGBM, CatBoost, Logistic Regression, KNN) com um meta-modelo.
- Compare com cada modelo isolado.
- Use cross-validation honesta (out-of-fold predictions).

> Quer ver esse stacking implementado com previsões out-of-fold explícitas, evitando o vazamento entre modelo base e meta-modelo? A versão Acelerado tem o código completo no [Projeto 4.1 — Stacking em problema tabular real](/trilha-llm/04_ml_moderno#projeto-41--stacking-em-problema-tabular-real).

### Projeto 4.2 — SimCLR em CIFAR-10 (preview de contrastive learning)
- Implemente em PyTorch (ou siga tutorial da Lightning).
- Treine encoder sem rótulos.
- Avalie via linear probing (treine só uma cabeça linear sobre os embeddings).
- **O que isso prova**: que self-supervision pode aprender boas representações sem rótulos.

> **Variante guiada**: antes de treinar com contrastive loss, visualize alguns pares positivos/negativos gerados pelas augmentations — confirme que os pares positivos ainda "parecem" a mesma imagem pra um humano, e os negativos não. Um par positivo distorcido demais (irreconhecível) atrapalha o treino tanto quanto um bug de código.

> Quer ver o SimCLR completo — augmentations, encoder, a loss NT-Xent explicada linha a linha, e a avaliação por linear probing? A versão Acelerado tem o código completo no [Projeto 4.2 — SimCLR em CIFAR-10](/trilha-llm/04_ml_moderno#projeto-42--simclr-em-cifar-10-contrastive-learning-do-zero).

### Projeto 4.3 — Hyperparameter optimization sério
- Aplique Optuna num XGBoost para um problema com tempo de treinamento ~minutos.
- Compare com random search e grid search.
- Plote a evolução do score conforme trials.

> Quer ver essa comparação (Optuna vs random vs grid search) implementada e plotada? A versão Acelerado tem o código completo no [Projeto 4.3 — Hyperparameter optimization sério com Optuna](/trilha-llm/04_ml_moderno#projeto-43--hyperparameter-optimization-sério-com-optuna).

### Projeto 4.4 — Active learning loop
- Comece com 100 exemplos rotulados.
- Treine modelo, identifique 50 exemplos mais incertos no pool não-rotulado.
- "Rotule" (consultando ground-truth disponível) e retreine.
- Compare curva de aprendizado vs random sampling.

> **Variante guiada**: rode o loop de active learning e o de random sampling lado a lado, plotando accuracy vs número de exemplos rotulados para os dois — a diferença entre as curvas é a evidência concreta do valor de active learning.

> Quer ver esse loop de active learning implementado e comparado lado a lado com random sampling? A versão Acelerado tem o código completo no [Projeto 4.4 — Active learning loop](/trilha-llm/04_ml_moderno#projeto-44--active-learning-loop).

### Projeto 4.5 (opcional) — Análise causal simples
- Use o dataset Lalonde (clássico em causal inference).
- Estime efeito do tratamento via propensity score matching e via OLS ingênuo.
- Compare resultados; entenda por que diferem.

> Quer ver essa comparação implementada — OLS ingênuo vs propensity score matching no mesmo dataset Lalonde? A versão Acelerado tem o código completo no [Projeto 4.5 — Análise causal simples](/trilha-llm/04_ml_moderno#projeto-45-opcional--análise-causal-simples).

---

## Erros comuns

- **Confundir SSL com unsupervised learning.** SSL gera supervisão sintética; unsupervised tradicional não.
- **Achar que ensembles sempre ajudam.** Em modelos já bem ajustados, o ganho marginal pode não compensar a complexidade.
- **AutoML como muleta.** É ferramenta de busca, não substitui entendimento.
- **Ignorar incerteza no output.** Modelo que dá probabilidade calibrada é muito mais útil em produção.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Self-supervised learning | Pretraining de LLMs (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Transfer learning | Fine-tuning de LLMs (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Contrastive learning | CLIP, embeddings (mod. [12](12_rag.mdx), [18](18_multimodal.mdx)) |
| Few-shot | In-context learning (mod. [11](11_prompt_engineering.md)) |
| Probabilistic models | Sampling em LLMs (mod. [11](11_prompt_engineering.md)) |
| Uncertainty | Hallucination, calibração (mod. [14](14_avaliacao_e_seguranca.md)) |

---

## Saiba mais

Alguns tópicos deste módulo foram citados sem profundidade — grandes demais para caber aqui sem desviar do fluxo principal:

- **BYOL e métodos contrastivos sem negativos** (4.1) — SimCLR/MoCo precisam de pares negativos explícitos; BYOL (Bootstrap Your Own Latent) aprende representações contrastivas sem eles, usando uma rede "professora" atualizada por média móvel. `Paper` **Bootstrap Your Own Latent** — Grill et al. (2020), já referenciado acima.
- **Gradual unfreezing e domain adaptation a fundo** (4.2) — estratégias mais refinadas de transfer learning para quando a tarefa-alvo é bem diferente da tarefa-fonte. `Paper` **ULMFiT** — Howard & Ruder (2018), já referenciado acima.
- **AutoML frameworks completos** (4.5) — AutoSklearn, FLAML, H2O AutoML automatizam não só hiperparâmetros mas a escolha do próprio algoritmo e do pré-processamento; vale a pena só depois de já saber fazer manualmente o que essas ferramentas fazem por baixo. `Ferramenta` **FLAML Documentation**. https://microsoft.github.io/FLAML/
- **Modelos probabilísticos além do escopo aqui** (4.8) — processos de Dirichlet, campos aleatórios de Markov, e inferência exata em modelos gráficos pequenos. `Livro` *Probabilistic Machine Learning* (vol. 2) — Kevin Murphy, já referenciado acima.

---

## Checklist de saída

- [ ] Sei explicar a diferença entre supervised, unsupervised e self-supervised (se não, revise a seção 4.1).
- [ ] Entendo intuitivamente por que pretraining em escala funciona (se não, revise a seção 4.2).
- [ ] Implementei pelo menos uma técnica de SSL (contrastive ou masked) (se não, revise o Projeto 4.2).
- [ ] Sei usar Optuna para HPO em problema real (se não, revise a seção 4.5 e o Projeto 4.3).
- [ ] Entendo o que é confounding e por que correlação ≠ causação importa (se não, revise a seção 4.7 e o Projeto 4.5).
