---
id: 01_matematica
title: "Módulo 01 — Matemática (Fundamentos do Zero)"
sidebar_position: 1
---

# Módulo 01 — Matemática (Fundamentos do Zero)

> **Objetivo**: construir a intuição e a mecânica matemática que aparecerão em **todos** os módulos seguintes. Matemática aqui não é exibicionismo: é a linguagem que descreve como redes neurais aprendem.
>
> **Pré-requisitos**: nenhum. Comece do zero.
>
> **Tempo de referência (não prazo)**: 4–8 semanas em ritmo moderado.

---

## Por que isso importa

- **Álgebra linear** descreve o que uma rede neural *é*: composições de transformações lineares com não-linearidades.
- **Cálculo** descreve como ela *aprende*: gradientes ajustam parâmetros para minimizar erro.
- **Probabilidade** descreve o que ela *modela*: distribuições sobre dados, incerteza, geração.
- **Otimização** descreve o *processo*: como a rede converge (ou não).

Sem isso, você não consegue ler papers, não entende por que algo falha, e fica preso em "tunar prompt".

---

## 1.1 Álgebra Linear

### Conceitos obrigatórios

- Vetores: norma (L1, L2, L∞), produto interno, projeção, ortogonalidade.
- Matrizes: produto matricial, transposição, inversa, traço, determinante.
- Espaços vetoriais: base, dimensão, posto (rank), espaço nulo.
- Transformações lineares: matrizes como funções; rotação, escala, projeção.
- Autovalores e autovetores; diagonalização.
- Decomposições: **SVD** (Singular Value Decomposition), **PCA** (Principal Component Analysis, como aplicação de SVD), QR, Cholesky.

### Conceitos opcionais (aprofundamento)

- Espaços de Hilbert (aparece em kernels).
- Tensores como generalização de matrizes (essencial para DL).
- Decomposição de matrizes esparsas.

### Da matemática ao código

- Implementar produto matricial em Python puro, depois NumPy. Comparar tempo.
- Implementar PCA via SVD a partir do zero (sem `sklearn`).
- Visualizar autovetores como direções preservadas por uma matriz.

> Aprofundar: teoria completa em [v2 — 1.1 Álgebra Linear](/trilha-llm/v2/01_matematica#11-álgebra-linear); código completo em [v3 — 1.1 Álgebra Linear](/trilha-llm/01_matematica#11-álgebra-linear).

### Referências

- `Curso` **MIT 18.06 — Linear Algebra**, Gilbert Strang (vídeos + livro). https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- `Curso` **3Blue1Brown — Essence of Linear Algebra** (intuição visual, indispensável). https://www.3blue1brown.com/topics/linear-algebra
- `Livro` **Mathematics for Machine Learning**, cap. 2–4. https://mml-book.github.io/
- `Livro` **Linear Algebra Done Right** — Sheldon Axler (mais rigoroso, opcional).

---

## 1.2 Cálculo

### Conceitos obrigatórios

- Limites e continuidade (intuição, não formalismo pesado).
- Derivada: definição, regras (produto, quociente, **regra da cadeia**).
- Derivadas parciais e gradiente (∇f).
- Jacobiano e Hessiana.
- Integral (definida e indefinida) — menos central em DL, mas aparece em probabilidade.
- Séries de Taylor (justifica linearizações em otimização).

### Da matemática ao código

- Implementar derivada numérica (diferença finita) e comparar com derivada analítica.
- Visualizar gradiente como campo vetorial em superfícies 2D.
- Implementar **autograd** simplificado (forward + backward em uma rede de 1 camada).

> Aprofundar: teoria completa em [v2 — 1.2 Cálculo](/trilha-llm/v2/01_matematica#12-cálculo); código completo em [v3 — 1.2 Cálculo](/trilha-llm/01_matematica#12-cálculo).

### Por que regra da cadeia é o coração do DL

Backpropagation é, literalmente, regra da cadeia aplicada a uma composição muito longa de funções. Quando você entende isso, "treinar uma rede neural" deixa de ser mágica.

### Referências

- `Curso` **3Blue1Brown — Essence of Calculus**. https://www.3blue1brown.com/topics/calculus
- `Curso` **MIT 18.01 — Single Variable Calculus**. https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/
- `Curso` **MIT 18.02 — Multivariable Calculus**. https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/
- `Livro` **Mathematics for Machine Learning**, cap. 5. https://mml-book.github.io/
- `Paper` **Automatic Differentiation in Machine Learning: a Survey** — Baydin et al. (2018). https://arxiv.org/abs/1502.05767

---

## 1.3 Probabilidade e Estatística

### Conceitos obrigatórios

- Espaço amostral, eventos, probabilidade condicional.
- **Teorema de Bayes** (fundamental para inferência).
- Variáveis aleatórias: discretas vs contínuas.
- Distribuições importantes: Bernoulli, Binomial, Categórica, Gaussiana (Normal), Poisson, Exponencial.
- Esperança, variância, covariância, correlação.
- **Estimador de Máxima Verossimilhança (MLE)** — base do treinamento de quase todo modelo.
- Cross-entropy e KL-divergência (Kullback-Leibler; loss functions).
- Inferência Bayesiana (prior, posterior, likelihood).

### Conceitos opcionais

- Cadeias de Markov (relevantes em RL).
- Processos Gaussianos.
- Inferência variacional (aparece em VAEs).

### Da matemática ao código

- Simular o teorema de Bayes em um problema clássico (teste médico, problema do táxi).
- Calcular MLE para uma Gaussiana a partir de amostras.
- Implementar cross-entropy loss à mão e comparar com `torch.nn.CrossEntropyLoss`.

> Aprofundar: teoria completa em [v2 — 1.3 Probabilidade e Estatística](/trilha-llm/v2/01_matematica#13-probabilidade-e-estatística); código completo em [v3 — 1.3 Probabilidade e Estatística](/trilha-llm/01_matematica#13-probabilidade-e-estatística).

### Referências

- `Curso` **MIT 6.041 — Probabilistic Systems Analysis**. https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/
- `Livro` **Mathematics for Machine Learning**, cap. 6. https://mml-book.github.io/
- `Livro` **Pattern Recognition and Machine Learning** — Bishop, cap. 1–2.
- `Livro` **Probabilistic Machine Learning** (vol. 1) — Kevin Murphy. https://probml.github.io/pml-book/

---

## 1.4 Otimização

### Conceitos obrigatórios

- Função convexa vs não-convexa.
- Mínimo local vs global.
- **Gradient Descent** (vanilla).
- Stochastic Gradient Descent (SGD), mini-batch.
- Momentum, Nesterov.
- Adaptativos: AdaGrad, RMSProp, **Adam**, AdamW.
- Learning rate schedules: warmup, cosine, linear decay.

### Conceitos opcionais

- Otimização restrita (Lagrangianos, KKT — Karush-Kuhn-Tucker).
- Métodos de segunda ordem (Newton, L-BFGS — Limited-memory BFGS).
- Otimização sem gradiente (CMA-ES — Covariance Matrix Adaptation Evolution Strategy) — aparece em RL e busca de hiperparâmetros.

### Da matemática ao código

- Implementar SGD do zero para minimizar f(x) = x² + ruído.
- Implementar Adam do zero a partir do paper original.
- Plotar trajetórias de diferentes otimizadores em uma superfície de loss conhecida.

> Aprofundar: teoria completa em [v2 — 1.4 Otimização](/trilha-llm/v2/01_matematica#14-otimização); código completo em [v3 — 1.4 Otimização](/trilha-llm/01_matematica#14-otimização).

### Referências

- `Paper` **Adam: A Method for Stochastic Optimization** — Kingma & Ba (2014). https://arxiv.org/abs/1412.6980
- `Paper` **Decoupled Weight Decay Regularization (AdamW)** — Loshchilov & Hutter (2017). https://arxiv.org/abs/1711.05101
- `Livro` **Convex Optimization** — Boyd & Vandenberghe (livro completo gratuito). https://web.stanford.edu/~boyd/cvxbook/
- `Curso` **Stanford EE364A — Convex Optimization**. https://web.stanford.edu/class/ee364a/

---

## Projetos práticos (curtos, obrigatórios)

Cada projeto deve ser implementado **sem usar bibliotecas que escondam a matemática**. Use NumPy puro. O ponto é "ver" a matemática rodar.

### Projeto 1.1 — PCA from scratch
**O que prova**: que você entende SVD, autovalores, e redução de dimensionalidade.
- Carregue o dataset MNIST (Modified National Institute of Standards and Technology; 28×28 = 784 dimensões).
- Implemente PCA via SVD em NumPy.
- Reduza para 50 dimensões; reconstrua; compare visualmente.
- **Sem `sklearn.decomposition.PCA`**.

### Projeto 1.2 — Regressão linear from scratch (3 vias)
**O que prova**: que você entende MLE, gradient descent e álgebra linear.
- Solução fechada (equação normal): w = (XᵀX)⁻¹Xᵀy.
- Gradient descent manual.
- SGD em mini-batches.
- Compare convergência e custo computacional.

### Projeto 1.3 — Autograd minimalista
**O que prova**: que você entende a regra da cadeia mecanicamente.
- Implemente um motor de autodiferenciação reverso para escalares (estilo `micrograd` do Karpathy, mas escreva sozinho primeiro).
- Treine uma rede de 1 camada para XOR.
- **Referência conceitual**: `Curso` Karpathy — *The spelled-out intro to neural networks and backpropagation: building micrograd*. https://www.youtube.com/watch?v=VMj-3S1tku0

### Projeto 1.4 — Comparativo de otimizadores
**O que prova**: que você entende as diferenças entre SGD, Momentum, Adam.
- Minimize a função de Rosenbrock com cada otimizador.
- Plote trajetórias.
- Faça o mesmo em uma rede neural simples.

---

## Erros comuns nesta fase

- **Ler sem implementar.** Matemática não-implementada não vira intuição.
- **Pular álgebra linear** porque "parece básico". Toda operação em DL é álgebra linear; se você não enxerga isso, vai sofrer no módulo [7](07_transformers.mdx).
- **Estudar otimização antes de cálculo.** Ordem importa.
- **Achar que `numpy` é trivial.** Não é. Broadcasting, vetorização e *shapes* são onde a maioria dos bugs nasce.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Produto matricial | DL (forward pass) — Mod. [5](05_deep_learning.md), Transformers — Mod. [7](07_transformers.mdx) |
| Regra da cadeia | Backpropagation — Mod. [5](05_deep_learning.md) |
| MLE / Cross-entropy | Loss functions — Mod. [5](05_deep_learning.md), [9](09_treinamento_e_alinhamento.mdx) |
| SVD | Embeddings, LoRA — Mod. [6](06_nlp_classico.md), [10](10_eficiencia_e_inferencia_local.md) |
| Adam | Treinamento de qualquer rede moderna |
| Distribuições | Sampling em LLMs (temperature, top-k) — Mod. [11](11_prompt_engineering.md) |
| Gradient descent | Tudo |

---

## Checklist de saída

Você está pronto para o módulo [02](02_programacao_ferramentas.md) quando consegue:

- [ ] Explicar o que é o gradiente de uma função multivariada e por que apontamos *contra* ele em GD.
- [ ] Derivar à mão a backprop de uma rede com 1 camada oculta.
- [ ] Explicar por que cross-entropy loss é equivalente a MLE para classificação.
- [ ] Implementar PCA, regressão linear e SGD em NumPy sem consultar.
- [ ] Ler um paper que tenha equações sem travar nas notações ∇, ∂, 𝔼, ∑.
