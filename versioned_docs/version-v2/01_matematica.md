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

> **Intuição — vetores e normas**: um vetor é uma seta partindo da origem; suas componentes dizem até onde ir em cada direção. A **norma** mede o "tamanho" dessa seta, e existe mais de uma forma sensata de medir tamanho. Para `v = (3, -4)`: a **norma L2** (euclidiana, distância "em linha reta") é `√(3² + (-4)²) = √25 = 5` — a que aparece por padrão em quase toda fórmula de distância. A **norma L1** (Manhattan) soma os valores absolutos: `|3| + |-4| = 7` — a distância que você percorreria andando só em linhas horizontais/verticais, como numa grade de ruas; penalizar por L1 empurra parâmetros exatamente a zero, por isso aparece em regularização quando o objetivo é esparsidade (Lasso, mod. [04](04_ml_moderno.md)). A **norma L∞** pega só a maior componente em valor absoluto: `max(3, 4) = 4` — usada quando importa o pior caso, não a soma (ex.: bounds de robustez adversarial).
>
> **Intuição — produto interno, projeção e ortogonalidade**: o **produto interno** `u·v = Σ uᵢvᵢ` mede o quanto dois vetores "apontam para o mesmo lugar": positivo quando alinhados, zero quando perpendiculares (**ortogonais**), negativo quando opostos. Para `u = (1, 0)` e `v = (1, 1)`, `u·v = 1`. A **projeção** de `v` sobre `u` é a "sombra" que `v` projeta na direção de `u`: `proj_u(v) = (u·v / u·u) u`. Com os mesmos vetores, `proj_u(v) = (1/1)(1,0) = (1,0)` — a componente de `v` na direção de `u` é exatamente 1, e a parte ortogonal que sobra (`v - proj_u(v) = (0,1)`) não tem nenhuma componente na direção de `u`. Esse mecanismo — decompor um vetor numa parte alinhada com uma direção e numa parte ortogonal a ela — é o que constrói bases ortogonais (o processo de Gram-Schmidt por trás da decomposição QR, mais abaixo).
>
> **Intuição — matrizes como máquinas**: uma matriz é uma "máquina" que pega um vetor e o empurra pra uma nova direção — é isso que **transformação linear** significa na prática. Mecanicamente, `Av` é uma combinação linear das colunas de `A`, pesada pelas componentes de `v`; para multiplicar `AB`, cada elemento `(AB)ᵢⱼ` é o produto interno da linha `i` de `A` com a coluna `j` de `B` — para `A = [[1,2],[3,4]]` e `B = [[5,6],[7,8]]`, `(AB)₁₁ = 1×5 + 2×7 = 19`. A **transposição** (`Aᵀ`) espelha a matriz na diagonal, trocando linhas por colunas — é o que transforma produto interno em produto matricial (`u·v = uᵀv`). A **inversa** `A⁻¹` é a matriz que desfaz a transformação de `A` (`A⁻¹A = I`) — só existe quando `A` não "achata" o espaço no caminho. O **determinante** mede exatamente isso: é o fator pelo qual `A` escala área (2D) ou volume (3D); `det(A) = 0` significa que `A` colapsa o espaço numa dimensão menor, e é exatamente aí que `A⁻¹` deixa de existir. O **traço** (soma da diagonal) parece mais discreto, mas conecta com autovalores adiante: a soma dos autovalores de `A` é sempre igual ao traço de `A`.
>
> **Intuição — espaços vetoriais, base, dimensão, rank e espaço nulo**: uma **base** é o menor conjunto de vetores que reconstrói qualquer outro vetor do espaço por combinação linear; o número de vetores dessa base é a **dimensão** do espaço. O **rank** (posto) de uma matriz é a dimensão do espaço gerado pelas suas colunas — quantas direções de saída ela realmente produz, não importa quantas colunas tenha. Para `A = [[1, 2], [2, 4]]`, a segunda coluna é só a primeira vezes 2 — não traz direção nova, então `rank(A) = 1` mesmo `A` sendo 2×2. O **espaço nulo** é o conjunto de vetores que `A` manda para zero: nessa mesma `A`, `(2, -1)` satisfaz `A(2,-1) = (0,0)` — está no espaço nulo. O **teorema do posto-nulidade** amarra os três: `rank(A) + dim(espaço nulo) = número de colunas` — aqui, `1 + 1 = 2`, bate. Rank baixo significa que a matriz colapsa o espaço numa "sombra" de menor dimensão — a ideia central por trás de compressão (SVD/PCA, abaixo) e, mais adiante na trilha, de LoRA (mod. [10](10_eficiencia_e_inferencia_local.md)): a atualização de pesos durante fine-tuning tem rank intrinsecamente baixo, por isso duas matrizes bem menores conseguem aproximá-la.
>
> **Intuição — transformações lineares como rotação, escala e projeção**: toda matriz quadrada é uma receita geométrica. `R = [[cos θ, -sin θ], [sin θ, cos θ]]` gira qualquer vetor por um ângulo `θ` em torno da origem sem mudar seu tamanho — para `θ = 90°`, `R = [[0,-1],[1,0]]` e `R(1,0) = (0,1)`: o vetor girou 90° anti-horário. `S = [[2,0],[0,0.5]]` escala: estica o eixo x por 2, encolhe o eixo y pela metade. Uma matriz de **projeção** (como `P = [[1,0],[0,0]]`) achata o espaço, jogando tudo sobre um eixo — perde informação de propósito, a mesma ideia da projeção vetorial acima, só que aplicada ao espaço inteiro de uma vez, não a um único vetor.
>
> **Exemplo resolvido — autovalores, autovetores e diagonalização**: a maioria dos vetores muda de direção ao passar por uma matriz — exceto alguns "eixos especiais", que só esticam ou encolhem, sem virar. Esses são os **autovetores**; o quanto eles esticam é o **autovalor**. Para `A = [[2,0],[0,3]]`, o vetor `(1,0)` vira `(2,0)` — mesma direção, só esticado por 2. É autovetor com autovalor 2. Já um vetor como `(1,1)` viraria `(2,3)` — mudou de direção, não é autovetor dessa matriz. Quando uma matriz `n×n` tem `n` autovetores independentes, ela pode ser **diagonalizada**: `A = PDP⁻¹`, com `D` diagonal (os autovalores) e `P` tendo os autovetores como colunas — isso transforma operações caras (como `A¹⁰⁰`) em baratas, já que elevar uma matriz diagonal a uma potência é só elevar cada elemento da diagonal.
>
> **Intuição — decomposições**: **SVD** (Singular Value Decomposition) generaliza autovalores/autovetores para matrizes não-quadradas, decompondo qualquer transformação em rotação → escala → rotação; **PCA** (Principal Component Analysis) usa os autovetores da matriz de covariância dos dados para achar as direções de maior variância — é assim que os 784 pixels do MNIST viram 50 dimensões sem perder muita informação (Projeto 1.1, abaixo). A decomposição **QR** fatora qualquer matriz em `A = QR`, com `Q` de colunas ortogonais (construídas via Gram-Schmidt, a mesma ideia de ortogonalidade acima) e `R` triangular superior — é a forma numericamente mais estável de resolver mínimos quadrados, a mesma conta por trás da regressão linear (Projeto 1.2, abaixo). A decomposição de **Cholesky** (`A = LLᵀ`, `L` triangular inferior) só existe para matrizes positivas-definidas (como matrizes de covariância) e custa metade do esforço de uma decomposição genérica — é a forma padrão de amostrar de uma Gaussiana multivariada e de resolver sistemas lineares em otimização de segunda ordem.
>
> **Aplicação real**: todo forward pass de toda rede neural (inclusive Transformers, mod. [07](07_transformers.mdx)) é, mecanicamente, uma sequência de produtos matriciais. Entender "matriz = transformação" é entender o que uma camada de rede neural literalmente faz com sua entrada.
>
> **Checkpoint**: sem olhar o texto, explique em duas frases o que um autovetor representa geometricamente. Depois, calcule de cabeça as normas L1, L2 e L∞ de `(6, -8)`, e explique por que uma matriz de rank baixo pode ser usada para comprimir informação — e por que o determinante de uma matriz de rank baixo é sempre zero.

### Da matemática ao código

- Implementar produto matricial em Python puro, depois NumPy. Comparar tempo.
- Implementar PCA via SVD a partir do zero (sem `sklearn`).
- Visualizar autovetores como direções preservadas por uma matriz.

> Quer ver esse código rodando de verdade, com a comparação de tempo cronometrada e a visualização plotada? A versão Acelerado implementa os três pontos acima com código completo, na seção [1.1 — Da álgebra ao código](/trilha-llm/01_matematica#da-álgebra-ao-código) do Módulo 01.

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
> **Intuição — Jacobiano e Hessiana**: para uma função que também *produz* várias saídas (`f: ℝⁿ → ℝᵐ`), o gradiente sozinho não basta — o **Jacobiano** é a matriz `m×n` de todas as derivadas parciais, uma linha por saída, uma coluna por entrada. É a generalização de "derivada" para funções vetoriais, e é exatamente o que a regra da cadeia multiplica de camada em camada durante backprop numa rede real, já que cada camada tem seu próprio Jacobiano. A **Hessiana** é a matriz de *segundas* derivadas parciais de uma função escalar (`∂²f/∂xᵢ∂xⱼ`) — descreve a curvatura: autovalores positivos em todas as direções indicam um mínimo local (a função curva para cima em todo lugar), autovalores de sinais mistos indicam um ponto de sela. Métodos de segunda ordem (Newton, seção 1.4) usam a Hessiana para dar um passo mais informado que gradient descent, ao custo de calculá-la — proibitivo para redes com milhões de parâmetros, por isso praticamente ausente em deep learning.
>
> **Intuição — Séries de Taylor**: qualquer função suave pode ser aproximada perto de um ponto por um polinômio construído a partir de suas derivadas ali: `f(x+h) ≈ f(x) + f'(x)h + ½f''(x)h² + ...`. Gradient descent usa implicitamente a aproximação de primeira ordem (só o termo `f'(x)h`) para justificar que um passo pequeno na direção `-∇f` deve diminuir `f` — e é por isso que o learning rate precisa ser pequeno o suficiente para essa aproximação linear continuar válida: um passo grande demais sai da região onde a aproximação é confiável, e o algoritmo diverge em vez de convergir.
>
> **Aplicação real**: quando você entende que backprop é "só" regra da cadeia numa composição longa, "treinar uma rede neural" deixa de ser mágica — é cálculo mecânico, aplicado sistematicamente. O mod. [05](05_deep_learning.md) usa exatamente essa mecânica para treinar tudo, do MLP mais simples ao maior Transformer.
>
> **Checkpoint**: sem olhar o texto, explique por que o gradiente aponta na direção de *maior crescimento*, e não na de maior decrescimento. Depois, aplique a regra da cadeia você mesmo em `f(x) = (3x - 2)³` e confira expandindo o cubo — e explique em uma frase por que a Hessiana, e não só o gradiente, é necessária para distinguir um mínimo de um ponto de sela.

### Da matemática ao código

- Implementar derivada numérica (diferença finita) e comparar com derivada analítica.
- Visualizar gradiente como campo vetorial em superfícies 2D.
- Implementar **autograd** simplificado (forward + backward em uma rede de 1 camada).

> Quer ver esse código completo — a comparação numérica vs. analítica, o campo de gradiente plotado, e um motor de autodiferenciação genérico (não só para uma arquitetura fixa)? A versão Acelerado implementa os dois primeiros pontos na seção [1.2 — Do cálculo ao código](/trilha-llm/01_matematica#do-cálculo-ao-código), e o autograd completo no [Projeto 1.3 — Autograd minimalista](/trilha-llm/01_matematica#projeto-13--autograd-minimalista).

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
> **Intuição — variáveis aleatórias e distribuições**: uma variável aleatória mapeia o resultado de um experimento para um número; é **discreta** quando esse número vem de um conjunto contável (resultado de um dado) ou **contínua** quando vem de um intervalo (altura de uma pessoa). Cada distribuição importante é a resposta natural para um tipo de pergunta: **Bernoulli** modela um único evento binário (moeda, sim/não); **Binomial** conta quantos "sucessos" saem de várias Bernoullis independentes (quantas caras em 10 lançamentos); **Categórica** generaliza Bernoulli para mais de duas categorias — é exatamente a distribuição que a saída de uma rede de classificação, depois do softmax, representa; **Gaussiana (Normal)** aparece quando um valor é soma de muitos fatores pequenos e independentes (erro de medição, ruído) — o Teorema Central do Limite garante isso; **Poisson** conta eventos raros num intervalo de tempo fixo (chegadas num servidor); **Exponencial** modela o tempo até o próximo evento raro acontecer. Escolher uma distribuição é assumir uma hipótese sobre como os dados foram gerados — e é essa hipótese que MLE (abaixo) explora para estimar parâmetros.
>
> **Intuição — esperança, variância, covariância e correlação**: a **esperança** `E[X]` é a média ponderada pelas probabilidades — o valor esperado "no longo prazo". A **variância** `Var(X) = E[(X - E[X])²]` mede o quanto os valores se espalham em torno da média; sua raiz quadrada é o desvio-padrão, na mesma unidade da variável original. A **covariância** `Cov(X,Y)` generaliza variância para duas variáveis: positiva quando crescem juntas, negativa quando uma cresce enquanto a outra cai, zero quando não há relação linear entre elas. A **correlação** normaliza a covariância para ficar entre -1 e 1 (dividindo pelo produto dos desvios-padrão), tornando comparável a força da relação entre pares de variáveis em escalas diferentes. Para `X = (1,2,3)` e `Y = (2,4,6)` (proporcionais), `Cov(X,Y) > 0` e a correlação é exatamente 1 — relação linear perfeita.
>
> **Intuição — MLE e cross-entropy**: MLE pergunta "quais parâmetros tornam os dados que eu observei mais prováveis?". Para classificação, minimizar cross-entropy loss é *matematicamente equivalente* a maximizar a verossimilhança dos rótulos corretos sob a distribuição prevista pelo modelo — são a mesma otimização, só escrita de formas diferentes (cross-entropy é o negativo do log-likelihood). KL-divergência mede o quanto duas distribuições diferem; cross-entropy é KL-divergência mais um termo constante (a entropia da distribuição real) — por isso minimizar uma equivale a minimizar a outra na prática de treino.
>
> **Aplicação real**: todo treinamento de LLM por next-token prediction (mod. [09](09_treinamento_e_alinhamento.mdx)) é MLE disfarçado de cross-entropy loss — o modelo está literalmente aprendendo os parâmetros que tornam o texto de treino mais provável sob a distribuição que ele produz. Sampling com temperature e top-k (mod. [11](11_prompt_engineering.md)) manipula diretamente essa distribuição de probabilidade aprendida.
>
> **Checkpoint**: sem olhar o texto, explique por que testar positivo num exame raro não significa necessariamente que você tem a doença — o que o prior tem a ver com isso? Depois, explique em uma frase por que "minimizar cross-entropy" e "maximizar verossimilhança" são a mesma coisa.

### Da matemática ao código

- Simular o teorema de Bayes em um problema clássico (teste médico, problema do táxi).
- Calcular MLE para uma Gaussiana a partir de amostras.
- Implementar cross-entropy loss à mão e comparar com `torch.nn.CrossEntropyLoss`.

> Quer ver os três pontos acima implementados — a simulação de Monte Carlo do problema do teste médico comparada com a fórmula fechada, o MLE calculado numa amostra real, e a cross-entropy à mão batendo com `torch.nn.functional.cross_entropy`? A versão Acelerado tem o código completo dos três na seção [1.3 — Da probabilidade ao código](/trilha-llm/01_matematica#da-probabilidade-ao-código).

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
> **Intuição — de SGD a Adam**: gradient descent puro dá um passo de tamanho fixo na direção do gradiente atual, ignorando tudo que veio antes — isso faz ele oscilar em vales estreitos e curvos. **Momentum** acumula uma média móvel das direções passadas, como uma bola ganhando inércia: se o gradiente aponta consistentemente para o mesmo lado, o passo cresce; se ele muda de sinal a cada iteração, os passos se cancelam parcialmente. **AdaGrad** ataca um problema diferente: dá um learning rate menor para parâmetros que já acumularam gradientes grandes, e maior para os que quase não mudaram — útil quando features têm escalas muito diferentes, mas o acúmulo é monotônico e o learning rate efetivo eventualmente vai a zero, mesmo longe do mínimo. **RMSProp** corrige isso trocando a soma acumulada por uma média móvel exponencial dos gradientes ao quadrado, então o learning rate efetivo pode voltar a crescer se os gradientes recentes diminuírem. **Adam** combina os dois: Momentum (média móvel do gradiente) mais a escala adaptativa de RMSProp (média móvel do gradiente ao quadrado), com uma correção de viés para os primeiros passos — por isso converge bem na maioria dos problemas sem tuning cuidadoso, e é o otimizador padrão de facto em deep learning. **AdamW** desacopla o weight decay do gradiente, em vez de somá-lo à loss antes de derivar, evitando que ele interaja mal com a escala adaptativa de Adam — hoje é a variante preferida na prática.
>
> **Intuição — learning rate schedules**: o learning rate não precisa ser constante durante o treino. **Warmup** começa pequeno e cresce gradualmente nas primeiras iterações — com pesos ainda aleatórios os gradientes podem ser grandes e ruidosos, e um passo grande de cara pode desestabilizar o treino antes dele começar de verdade. Depois do warmup, um **decay** reduz o learning rate ao longo do treino: **cosine** segue metade de uma onda de cosseno (suave, terminando perto de zero), **linear** decai numa reta até zero. Ambos partem da mesma ideia: passos grandes são bons no início para explorar rápido, passos pequenos são melhores perto do fim para não pular o mínimo.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre mínimo local e mínimo global — e por que essa diferença preocupa menos na prática de deep learning do que a definição sugere à primeira vista. Depois, explique em uma frase o que Adam adiciona a SGD com Momentum, e por que warmup existe.

### Da matemática ao código

- Implementar SGD do zero para minimizar f(x) = x² + ruído.
- Implementar Adam do zero a partir do paper original.
- Plotar trajetórias de diferentes otimizadores em uma superfície de loss conhecida.

> Quer ver SGD, Momentum e Adam implementados do zero e suas trajetórias comparadas na função de Rosenbrock (e depois numa rede neural real)? A versão Acelerado tem o código completo no [Projeto 1.4 — Comparativo de otimizadores](/trilha-llm/01_matematica#projeto-14--comparativo-de-otimizadores).

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

## Saiba mais

Alguns tópicos citados ao longo do módulo são grandes demais para caber numa explicação completa aqui sem desviar do fluxo principal. Você não precisa deles para seguir para o módulo [02](02_programacao_ferramentas.md), mas se quiser aprofundar:

- **Espaços de Hilbert** (1.1) — a generalização de espaço vetorial com produto interno para dimensão infinita; aparece quando kernels em SVMs (mod. [04](04_ml_moderno.md)) projetam dados implicitamente num espaço de features de dimensão muito alta. `Livro` *Mathematics for Machine Learning*, cap. 12 (kernels). https://mml-book.github.io/
- **Tensores** (1.1) como generalização de matrizes — matriz é um tensor de ordem 2; ativações de uma CNN são tensores de ordem 4 (batch × canal × altura × largura). `Curso` MIT 18.06, aula sobre tensores.
- **Decomposição de matrizes esparsas** (1.1) — versões de SVD/Cholesky otimizadas para matrizes com maioria de zeros, comuns em grafos e sistemas de recomendação em escala. `Ferramenta` `scipy.sparse.linalg`.
- **Cálculo de variações** (1.2) — otimização sobre *funções*, não sobre números; base de controle ótimo e de alguns métodos de ML informados por física. `Livro` *Calculus of Variations* — Gelfand & Fomin.
- **Cadeias de Markov** (1.3) — processos onde o próximo estado só depende do atual; base matemática de RL (mod. [17](17_aprendizado_reforco.md)) e de MCMC (amostragem de distribuições complexas). `Livro` *Pattern Recognition and Machine Learning* — Bishop, cap. 11.
- **Processos Gaussianos** (1.3) — distribuições sobre funções, usados em otimização bayesiana de hiperparâmetros. `Livro` *Gaussian Processes for Machine Learning* — Rasmussen & Williams (gratuito). http://gaussianprocess.org/gpml/
- **Inferência variacional** (1.3) — a técnica por trás do treino de VAEs (Variational Autoencoders, mod. [05](05_deep_learning.md)): aproximar uma distribuição posterior impossível de calcular exatamente por uma mais simples, otimizável por gradiente. `Paper` *Auto-Encoding Variational Bayes* — Kingma & Welling (2013). https://arxiv.org/abs/1312.6114
- **KKT** (Karush-Kuhn-Tucker) e Lagrangianos (1.4) — condições que caracterizam o ótimo de um problema com restrições; aparecem na formulação dual de SVMs (mod. [04](04_ml_moderno.md)). `Livro` *Convex Optimization* — Boyd & Vandenberghe, cap. 5 (gratuito). https://web.stanford.edu/~boyd/cvxbook/
- **Métodos de segunda ordem** (1.4) — Newton e **L-BFGS** (Limited-memory BFGS) usam a Hessiana (ou uma aproximação dela, ver seção 1.2) para dar passos mais informados que gradient descent; caros demais para redes com milhões de parâmetros, mas padrão em otimização clássica de menor escala.
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy, 1.4) — otimização sem gradiente, útil quando a função não é diferenciável (RL, mod. [17](17_aprendizado_reforco.md); busca de hiperparâmetros).

---

## Checklist de saída

Você está pronto para o módulo [02](02_programacao_ferramentas.md) quando consegue:

- [ ] Explicar o que é o gradiente de uma função multivariada e por que apontamos *contra* ele em GD (se não, revise a seção 1.2).
- [ ] Derivar à mão a backprop de uma rede com 1 camada oculta (se não, revise a seção 1.2 e o mod. [05](05_deep_learning.md)).
- [ ] Explicar por que cross-entropy loss é equivalente a MLE para classificação (se não, revise a seção 1.3).
- [ ] Implementar PCA, regressão linear e SGD em NumPy sem consultar (se não, revise os Projetos 1.1, 1.2 e 1.4).
- [ ] Ler um paper que tenha equações sem travar nas notações ∇, ∂, 𝔼, ∑.
