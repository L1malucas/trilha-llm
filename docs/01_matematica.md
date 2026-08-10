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

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar o que uma matriz *faz* com um vetor, geometricamente, sem recorrer só à definição algébrica.
- Explicar por que o gradiente aponta na direção de maior crescimento — e por que otimização caminha *contra* ele.
- Derivar a regra da cadeia para uma composição simples de funções e reconhecer isso como o mecanismo por trás de backpropagation.
- Aplicar o Teorema de Bayes para atualizar uma crença a partir de nova evidência, num exemplo concreto.
- Explicar por que minimizar cross-entropy é equivalente a maximizar verossimilhança (MLE).
- Implementar PCA, regressão linear e SGD em NumPy puro, sem consultar.

Este módulo constrói cada conceito a partir de uma intuição geométrica ou de um exemplo numérico pequeno antes de qualquer notação formal — e fecha cada seção com um checkpoint de autoverificação.

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
- Decomposições: **SVD** (Singular Value Decomposition), **PCA** (como aplicação de SVD), QR, Cholesky.

> **Intuição**: pense num vetor como uma seta partindo da origem, e numa matriz como uma "máquina" que pega essa seta e a empurra pra uma nova direção — é isso que **transformação linear** significa na prática. Produto interno mede o quanto dois vetores "apontam para o mesmo lugar" (alto quando alinhados, zero quando perpendiculares — ortogonais). Rank é quantas direções *independentes* a matriz realmente usa: uma matriz de rank baixo colapsa o espaço numa "sombra" de menor dimensão, o que é exatamente a ideia por trás de compressão (SVD/PCA) e, mais adiante na trilha, de LoRA.
>
> **Exemplo resolvido — autovalores e autovetores**: a maioria dos vetores muda de direção ao passar por uma matriz — exceto alguns "eixos especiais", que só esticam ou encolhem, sem virar. Esses são os **autovetores**; o quanto eles esticam é o **autovalor**. Para `A = [[2,0],[0,3]]`, o vetor `(1,0)` vira `(2,0)` — mesma direção, só esticado por 2. É autovetor com autovalor 2. Já um vetor como `(1,1)` viraria `(2,3)` — mudou de direção, não é autovetor dessa matriz. **SVD** generaliza essa ideia para matrizes não-quadradas, decompondo qualquer transformação em rotação → escala → rotação; **PCA** usa os autovetores da matriz de covariância dos dados para achar as direções de maior variância — é assim que os 784 pixels do MNIST viram 50 dimensões sem perder muita informação (Projeto 1.1, abaixo).
>
> **Aplicação real**: todo forward pass de toda rede neural (inclusive Transformers, mod. [07](07_transformers.mdx)) é, mecanicamente, uma sequência de produtos matriciais. Entender "matriz = transformação" é entender o que uma camada de rede neural literalmente faz com sua entrada.
>
> **Checkpoint**: sem olhar o texto, explique em duas frases o que um autovetor representa geometricamente. Depois, explique por que uma matriz de rank baixo pode ser usada para comprimir informação.

### Conceitos opcionais (aprofundamento)

- Espaços de Hilbert (aparece em kernels).
- Tensores como generalização de matrizes (essencial para DL).
- Decomposição de matrizes esparsas.

### Da matemática ao código

- Implementar produto matricial em Python puro, depois NumPy. Comparar tempo.
- Implementar PCA via SVD a partir do zero (sem `sklearn`).
- Visualizar autovetores como direções preservadas por uma matriz.

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

> **Intuição**: a derivada de uma função num ponto é a inclinação da reta tangente ali — "se eu andar um pouquinho pra direita, o quanto a função sobe ou desce?". O **gradiente** (∇f) generaliza isso para funções de várias variáveis: é um vetor apontando na direção de **maior crescimento** da função. É por isso que otimização caminha *contra* o gradiente (gradient descent) — se você quer descer, vá na direção oposta à de maior subida.
>
> **Exemplo resolvido — regra da cadeia**: seja `f(x) = (2x + 1)²`. Podemos ver isso como composição de duas funções: `u(x) = 2x + 1` (a parte de dentro) e `g(u) = u²` (a parte de fora), onde `f(x) = g(u(x))`. A regra da cadeia diz `df/dx = dg/du × du/dx`. Aqui, `dg/du = 2u` e `du/dx = 2`, então `df/dx = 2u × 2 = 4u = 4(2x+1) = 8x + 4`. Confira: expandindo diretamente, `f(x) = 4x² + 4x + 1`, e `df/dx = 8x + 4` — bate. Backpropagation é exatamente isso, só que com uma cadeia muito mais longa: dezenas ou centenas de funções compostas (uma por camada/operação), e a regra da cadeia aplicada de trás pra frente para descobrir a contribuição de cada parâmetro no erro final.
>
> **Aplicação real**: quando você entende que backprop é "só" regra da cadeia numa composição longa, "treinar uma rede neural" deixa de ser mágica — é cálculo mecânico, aplicado sistematicamente. O mod. [05](05_deep_learning.md) usa exatamente essa mecânica para treinar tudo, do MLP mais simples ao maior Transformer.
>
> **Checkpoint**: sem olhar o texto, explique por que o gradiente aponta na direção de *maior crescimento*, e não na de maior decrescimento. Depois, aplique a regra da cadeia você mesmo em `f(x) = (3x - 2)³` e confira expandindo o cubo.

### Da matemática ao código

- Implementar derivada numérica (diferença finita) e comparar com derivada analítica.
- Visualizar gradiente como campo vetorial em superfícies 2D.
- Implementar **autograd** simplificado (forward + backward em uma rede de 1 camada).

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
- Cross-entropy e KL-divergência (loss functions).
- Inferência Bayesiana (prior, posterior, likelihood).

> **Intuição — Teorema de Bayes**: é a fórmula matemática de "atualizar sua crença diante de nova evidência". Exemplo clássico: um teste médico para uma doença rara (1% da população) tem 95% de acerto (tanto para detectar quem tem quanto pra descartar quem não tem). Você testou positivo — qual a chance de você realmente ter a doença? A intuição ingênua diz "95%", mas Bayes mostra que é bem menor: como a doença é rara, a maioria dos positivos vem de falsos positivos entre a enorme população saudável, não de verdadeiros positivos entre a pequena população doente. Formalmente: `P(doença|positivo) = P(positivo|doença) × P(doença) / P(positivo)` — o "prior" (1% de prevalência) pesa muito no resultado final.
>
> **Intuição — MLE e cross-entropy**: MLE pergunta "quais parâmetros tornam os dados que eu observei mais prováveis?". Para classificação, minimizar cross-entropy loss é *matematicamente equivalente* a maximizar a verossimilhança dos rótulos corretos sob a distribuição prevista pelo modelo — são a mesma otimização, só escrita de formas diferentes (cross-entropy é o negativo do log-likelihood). KL-divergência mede o quanto duas distribuições diferem; cross-entropy é KL-divergência mais um termo constante (a entropia da distribuição real) — por isso minimizar uma equivale a minimizar a outra na prática de treino.
>
> **Aplicação real**: todo treinamento de LLM por next-token prediction (mod. [09](09_treinamento_e_alinhamento.mdx)) é MLE disfarçado de cross-entropy loss — o modelo está literalmente aprendendo os parâmetros que tornam o texto de treino mais provável sob a distribuição que ele produz. Sampling com temperature e top-k (mod. [11](11_prompt_engineering.md)) manipula diretamente essa distribuição de probabilidade aprendida.
>
> **Checkpoint**: sem olhar o texto, explique por que testar positivo num exame raro não significa necessariamente que você tem a doença — o que o prior tem a ver com isso? Depois, explique em uma frase por que "minimizar cross-entropy" e "maximizar verossimilhança" são a mesma coisa.

### Conceitos opcionais

- Cadeias de Markov (relevantes em RL).
- Processos Gaussianos.
- Inferência variacional (aparece em VAEs).

### Da matemática ao código

- Simular o teorema de Bayes em um problema clássico (teste médico, problema do táxi).
- Calcular MLE para uma Gaussiana a partir de amostras.
- Implementar cross-entropy loss à mão e comparar com `torch.nn.CrossEntropyLoss`.

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

> **Intuição**: uma função **convexa** tem um único "vale" — qualquer caminho de descida chega ao mesmo mínimo global, como uma bacia. Uma função **não-convexa** (o caso de praticamente toda rede neural) tem múltiplos vales, platôs e selas — não há garantia de que gradient descent encontre o "melhor" mínimo, só *um* mínimo razoável. Isso é menos assustador na prática do que parece: em redes grandes, a maioria dos mínimos locais encontrados por SGD tem qualidade similar.
>
> O mod. [05](05_deep_learning.md#53-otimização-e-regularização-para-dl) tem um exemplo numérico resolvido, passo a passo, de gradient descent convergindo (`L(w) = (w-3)²`) e a intuição completa de Momentum/Adam como "inércia" e "passo adaptativo" — vale revisar aquela seção junto com esta, já que é a mesma matemática, só que aqui é onde ela nasce.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre mínimo local e mínimo global — e por que essa diferença preocupa menos na prática de deep learning do que a definição sugere à primeira vista.

### Conceitos opcionais

- Otimização restrita (Lagrangianos, KKT).
- Métodos de segunda ordem (Newton, L-BFGS).
- Otimização sem gradiente (CMA-ES) — aparece em RL e busca de hiperparâmetros.

### Da matemática ao código

- Implementar SGD do zero para minimizar f(x) = x² + ruído.
- Implementar Adam do zero a partir do paper original.
- Plotar trajetórias de diferentes otimizadores em uma superfície de loss conhecida.

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
- Carregue o dataset MNIST (28×28 = 784 dimensões).
- Implemente PCA via SVD em NumPy.
- Reduza para 50 dimensões; reconstrua; compare visualmente.
- **Sem `sklearn.decomposition.PCA`**.

> **Variante guiada**: antes de reduzir para 50 dimensões, reduza primeiro para 2 e plote os pontos coloridos por dígito — isso te dá uma verificação visual rápida (dígitos iguais devem formar aglomerados) antes de confiar na reconstrução em 50 dimensões.

### Projeto 1.2 — Regressão linear from scratch (3 vias)
**O que prova**: que você entende MLE, gradient descent e álgebra linear.
- Solução fechada (equação normal): w = (XᵀX)⁻¹Xᵀy.
- Gradient descent manual.
- SGD em mini-batches.
- Compare convergência e custo computacional.

> **Variante guiada**: implemente a solução fechada primeiro — ela te dá a resposta "certa" de referência. Só depois implemente GD e SGD, comparando se convergem para o mesmo `w` (devem, num problema convexo como este).

### Projeto 1.3 — Autograd minimalista
**O que prova**: que você entende a regra da cadeia mecanicamente.
- Implemente um motor de autodiferenciação reverso para escalares (estilo `micrograd` do Karpathy, mas escreva sozinho primeiro).
- Treine uma rede de 1 camada para XOR.
- **Referência conceitual**: `Curso` Karpathy — *The spelled-out intro to neural networks and backpropagation: building micrograd*. https://www.youtube.com/watch?v=VMj-3S1tku0

> **Variante guiada**: comece implementando só a operação de multiplicação com seu gradiente (`d(a*b)/da = b`), confirme que bate com diferenciação numérica, depois adicione soma, depois funções não-lineares — construir o motor operação por operação, validando cada uma, evita um emaranhado de bugs no final.

### Projeto 1.4 — Comparativo de otimizadores
**O que prova**: que você entende as diferenças entre SGD, Momentum, Adam.
- Minimize a função de Rosenbrock com cada otimizador.
- Plote trajetórias.
- Faça o mesmo em uma rede neural simples.

> **Variante guiada**: antes de plotar, formule uma hipótese de qual otimizador vai convergir mais rápido na Rosenbrock (uma função conhecida por ter um "vale" estreito e curvo) e por quê — depois confira se sua intuição sobre Momentum/Adam bateu com o resultado.

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
