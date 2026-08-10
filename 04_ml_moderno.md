# Módulo 04 — Machine Learning Moderno

> **Objetivo**: cobrir o que **não** é DL profundo nem ML clássico stricto sensu — paradigmas que emergiram entre os dois, e que são essenciais para entender LLMs (especialmente self-supervised learning) e problemas reais.
>
> **Pré-requisitos**: Módulo 03.
>
> **Tempo de referência**: 3–5 semanas.

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

### Papers fundamentais
- 📄 **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** — Chen et al. (2020). https://arxiv.org/abs/2002.05709
- 📄 **Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)** — He et al. (2019). https://arxiv.org/abs/1911.05722
- 📄 **Self-Supervised Learning: Generative or Contrastive** — Liu et al. (2020). https://arxiv.org/abs/2006.08218 (survey)
- 📄 **Bootstrap Your Own Latent (BYOL)** — Grill et al. (2020). https://arxiv.org/abs/2006.07733

### Cursos
- 🎓 **NYU Deep Learning (Yann LeCun & Alfredo Canziani)** — aulas dedicadas a SSL. https://atcold.github.io/NYU-DLSP21/

---

## 4.2 Transfer Learning

### Conceito
Treinar em uma tarefa-fonte com muito dado, transferir para tarefa-alvo com pouco dado. **É o paradigma dominante** em DL moderno.

### Estratégias
- **Feature extraction**: congelar o backbone, treinar só a cabeça nova.
- **Fine-tuning**: descongelar parcialmente ou totalmente.
- **Fine-tuning gradual** (gradual unfreezing).
- **Domain adaptation**: ajustar a distribuição-alvo.

### Referências
- 📄 **A Survey on Transfer Learning** — Pan & Yang (2010). https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf
- 📄 **Universal Language Model Fine-tuning (ULMFiT)** — Howard & Ruder (2018). https://arxiv.org/abs/1801.06146 (clássico em NLP)

---

## 4.3 Few-Shot e Meta-Learning

### Conceitos
- **Few-shot learning**: aprender com poucos exemplos por classe.
- **Zero-shot learning**: prever classes nunca vistas no treinamento.
- **Meta-learning** ("aprender a aprender"): MAML, Prototypical Networks.

### Por que importa para LLMs
Few-shot prompting (mod. 11) é um caso emergente de few-shot learning sem gradient updates — comportamento "in-context" que aparece com escala.

### Papers
- 📄 **Model-Agnostic Meta-Learning (MAML)** — Finn et al. (2017). https://arxiv.org/abs/1703.03400
- 📄 **Matching Networks for One Shot Learning** — Vinyals et al. (2016). https://arxiv.org/abs/1606.04080
- 📄 **Prototypical Networks for Few-shot Learning** — Snell et al. (2017). https://arxiv.org/abs/1703.05175

---

## 4.4 Ensembles modernos e Stacking

### Além de Random Forest e XGBoost
- **Stacking**: meta-modelo aprende a combinar predições de modelos base.
- **Blending**: variante simples de stacking com holdout.
- **Bayesian Model Averaging**.
- **Snapshot Ensembles** (em DL).

### Referências
- 📚 **Ensemble Methods: Foundations and Algorithms** — Zhou.
- 📄 **Stacked Generalization** — Wolpert (1992). https://www.researchgate.net/publication/222467943_Stacked_Generalization

---

## 4.5 AutoML e Neural Architecture Search (NAS)

### AutoML
- **Hyperparameter optimization**: random search, Bayesian optimization (Optuna, Hyperopt), Hyperband, BOHB.
- **AutoML frameworks**: AutoSklearn, Auto-Keras, FLAML, H2O AutoML.

### NAS
- **Differentiable NAS** (DARTS).
- **Reinforcement-learning-based NAS**.
- **Evolutionary NAS**.

### Crítica honesta
NAS prometeu mais do que entregou para uso geral. Para a maioria dos casos, transferir uma arquitetura existente é melhor. Mas conhecer o conceito é importante.

### Referências
- 📄 **Neural Architecture Search: A Survey** — Elsken et al. (2018). https://arxiv.org/abs/1808.05377
- 📄 **DARTS: Differentiable Architecture Search** — Liu et al. (2018). https://arxiv.org/abs/1806.09055
- 🛠 **Optuna**. https://optuna.org/

---

## 4.6 Aprendizado com poucos rótulos

### Técnicas
- **Active learning**: o modelo escolhe quais exemplos rotular.
- **Weak supervision**: labels imprecisos/programáticos (Snorkel).
- **Semi-supervised learning**: combinar rotulado + não-rotulado.
- **Pseudo-labeling**, **co-training**.
- **Data augmentation** como forma de "criar" supervisão.

### Referências
- 📄 **Snorkel: Rapid Training Data Creation with Weak Supervision** — Ratner et al. (2017). https://arxiv.org/abs/1711.10160
- 📚 **Active Learning** — Burr Settles. http://burrsettles.com/pub/settles.activelearning.pdf

---

## 4.7 Causal Inference (opcional, mas recomendado)

### Por que
Modelos preditivos respondem "qual é a probabilidade de Y dado X?" mas frequentemente queremos "o que acontece com Y se eu intervir em X?". Esses são problemas diferentes.

### Conceitos
- **Confounding** e variáveis de confundimento.
- **DAGs causais** (Judea Pearl).
- **Counterfactuals**.
- **Propensity score matching**.
- **Instrumental variables**.
- **Difference-in-differences**.

### Conexão com IA
- Avaliação justa de modelos.
- Compreensão de viés algorítmico.
- Em RL: causalidade é fundamental.

### Referências
- 📚 **Causal Inference: The Mixtape** — Cunningham (gratuito). https://mixtape.scunning.com/
- 📚 **The Book of Why** — Judea Pearl (acessível).
- 📚 **Causal Inference: What If** — Hernán & Robins (gratuito). https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

---

## 4.8 Modelos probabilísticos modernos

### Tópicos
- **Probabilistic Programming**: PyMC, Stan, NumPyro, Pyro.
- **Variational Inference**.
- **Normalizing Flows** (preparação para módulo 19).
- **Bayesian Deep Learning**: dropout como Bayesian approximation, deep ensembles como uncertainty.

### Por que importa
- LLMs são fundamentalmente modelos probabilísticos sobre tokens.
- Uncertainty quantification é o que diferencia "modelo bom" de "modelo confiável".

### Referências
- 📚 **Probabilistic Machine Learning (vol. 2: Advanced Topics)** — Murphy. https://probml.github.io/pml-book/
- 📄 **Auto-Encoding Variational Bayes (VAE)** — Kingma & Welling (2013). https://arxiv.org/abs/1312.6114

---

## 🧪 Projetos práticos

### Projeto 4.1 — Stacking em problema tabular real
- Combine 5 modelos heterogêneos (XGBoost, LightGBM, CatBoost, Logistic Regression, KNN) com um meta-modelo.
- Compare com cada modelo isolado.
- Use cross-validation honesta (out-of-fold predictions).

### Projeto 4.2 — SimCLR em CIFAR-10 (preview de contrastive learning)
- Implemente em PyTorch (ou siga tutorial da Lightning).
- Treine encoder sem rótulos.
- Avalie via linear probing (treine só uma cabeça linear sobre os embeddings).
- **O que isso prova**: que self-supervision pode aprender boas representações sem rótulos.

### Projeto 4.3 — Hyperparameter optimization sério
- Aplique Optuna num XGBoost para um problema com tempo de treinamento ~minutos.
- Compare com random search e grid search.
- Plote a evolução do score conforme trials.

### Projeto 4.4 — Active learning loop
- Comece com 100 exemplos rotulados.
- Treine modelo, identifique 50 exemplos mais incertos no pool não-rotulado.
- "Rotule" (consultando ground-truth disponível) e retreine.
- Compare curva de aprendizado vs random sampling.

### Projeto 4.5 (opcional) — Análise causal simples
- Use o dataset Lalonde (clássico em causal inference).
- Estime efeito do tratamento via propensity score matching e via OLS ingênuo.
- Compare resultados; entenda por que diferem.

---

## ⚠️ Erros comuns

- **Confundir SSL com unsupervised learning.** SSL gera supervisão sintética; unsupervised tradicional não.
- **Achar que ensembles sempre ajudam.** Em modelos já bem ajustados, o ganho marginal pode não compensar a complexidade.
- **AutoML como muleta.** É ferramenta de busca, não substitui entendimento.
- **Ignorar incerteza no output.** Modelo que dá probabilidade calibrada é muito mais útil em produção.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Self-supervised learning | Pretraining de LLMs (mod. 09) |
| Transfer learning | Fine-tuning de LLMs (mod. 09) |
| Contrastive learning | CLIP, embeddings (mod. 12, 18) |
| Few-shot | In-context learning (mod. 11) |
| Probabilistic models | Sampling em LLMs (mod. 11) |
| Uncertainty | Hallucination, calibração (mod. 14) |

---

## Checklist de saída

- [ ] Sei explicar a diferença entre supervised, unsupervised e self-supervised.
- [ ] Entendo intuitivamente por que pretraining em escala funciona.
- [ ] Implementei pelo menos uma técnica de SSL (contrastive ou masked).
- [ ] Sei usar Optuna para HPO em problema real.
- [ ] Entendo o que é confounding e por que correlação ≠ causação importa.
