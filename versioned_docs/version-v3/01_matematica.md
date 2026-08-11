---
id: 01_matematica
title: "Módulo 01 — Matemática (Fundamentos do Zero)"
sidebar_position: 20
---

# Módulo 01 — Matemática (Fundamentos do Zero)

> **Objetivo**: construir, formalmente, a matemática que já sustentou silenciosamente tudo que você implementou nesta trilha — álgebra linear, cálculo, probabilidade, otimização. Este é o último módulo da trilha Acelerado, não porque a matemática seja menos importante, mas porque agora cada conceito tem um código concreto, já testado por você, para se ancorar.
>
> **Pré-requisitos**: toda a trilha (módulos [08](08_llms_arquiteturas.md)–[02](02_programacao_ferramentas.md)). Nenhum conhecimento prévio de matemática além do que você já absorveu na prática.
>
> **Tempo de referência**: 4–8 semanas em ritmo moderado.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar o que uma matriz *faz* com um vetor, geometricamente, sem recorrer só à definição algébrica.
- Explicar por que o gradiente aponta na direção de maior crescimento — e por que otimização caminha *contra* ele.
- Derivar a regra da cadeia para uma composição simples de funções e reconhecer isso como o mecanismo por trás de todo `.backward()` que você já chamou desde o Projeto 8.3.
- Aplicar o Teorema de Bayes para atualizar uma crença a partir de nova evidência, num exemplo concreto.
- Explicar por que minimizar cross-entropy é equivalente a maximizar verossimilhança (MLE) — a mesma loss que você usa desde o primeiro loop de treino da trilha.
- Implementar PCA, regressão linear e um motor de autodiferenciação em NumPy puro, sem consultar.

Este módulo constrói cada conceito a partir de uma intuição geométrica ou de um exemplo numérico pequeno antes de qualquer notação formal — e, diferente de como seria se este fosse o primeiro módulo da trilha, cada seção também aponta para o código específico, já escrito por você, onde aquele conceito estava operando por baixo dos panos.

---

## Por que isso importa

Álgebra linear descreve o que uma rede neural *é*: composições de transformações lineares com não-linearidades — literalmente o `MiniLlama` do Projeto 8.3. Cálculo descreve como ela *aprende*: gradientes ajustam parâmetros para minimizar erro — o que `loss.backward()` fez, silenciosamente, em toda linha de treino que você escreveu. Probabilidade descreve o que ela *modela*: distribuições sobre dados, incerteza, geração — o `softmax` no fim de todo forward pass, o `torch.multinomial` de toda função `generate`. Otimização descreve o *processo*: como a rede converge (ou não) — o `AdamW` que você configurou dezenas de vezes. Você já usou tudo isso. Este módulo é sobre parar de usar e começar a enxergar.

---

## 1.1 Álgebra Linear

### Conceitos obrigatórios

- Vetores: norma (L1, L2, L∞), produto interno, projeção, ortogonalidade.
- Matrizes: produto matricial, transposição, inversa, traço, determinante.
- Espaços vetoriais: base, dimensão, posto (rank), espaço nulo.
- Transformações lineares: matrizes como funções; rotação, escala, projeção.
- Autovalores e autovetores; diagonalização.
- Decomposições: **SVD** (Singular Value Decomposition), **PCA** (como aplicação de SVD), QR, Cholesky.

> **Intuição**: pense num vetor como uma seta partindo da origem, e numa matriz como uma "máquina" que pega essa seta e a empurra pra uma nova direção — é isso que **transformação linear** significa na prática, e é exatamente o que `self.q_proj(x)` fez em `GQACausalSelfAttention` no Projeto 8.3: pegou o vetor de cada token e o empurrou para um novo espaço (o espaço de Queries). Produto interno mede o quanto dois vetores "apontam para o mesmo lugar" (alto quando alinhados, zero quando perpendiculares — ortogonais) — é literalmente o `Q·Kᵀ` da attention, e é a mesma operação por trás de toda similaridade de cosseno que você calculou desde o Projeto 8.5. Rank é quantas direções *independentes* a matriz realmente usa: uma matriz de rank baixo colapsa o espaço numa "sombra" de menor dimensão — exatamente a hipótese por trás de LoRA (mod. 09): a *atualização* de pesos durante fine-tuning tem rank intrinsecamente baixo, por isso `A` e `B` (duas matrizes bem menores) conseguem aproximá-la.
>
> **Exemplo resolvido — autovalores e autovetores**: a maioria dos vetores muda de direção ao passar por uma matriz — exceto alguns "eixos especiais", que só esticam ou encolhem, sem virar. Esses são os **autovetores**; o quanto eles esticam é o **autovalor**. Para `A = [[2,0],[0,3]]`, o vetor `(1,0)` vira `(2,0)` — mesma direção, só esticado por 2. É autovetor com autovalor 2. Já um vetor como `(1,1)` viraria `(2,3)` — mudou de direção, não é autovetor dessa matriz. **SVD** generaliza essa ideia para matrizes não-quadradas, decompondo qualquer transformação em rotação → escala → rotação; **PCA** usa os autovetores da matriz de covariância dos dados para achar as direções de maior variância — é assim que os 784 pixels do MNIST viram 50 dimensões sem perder muita informação (Projeto 1.1, abaixo) — o mesmo tipo de compressão de informação, em espírito, que o bottleneck do VAE (Projeto 5.5) faz de forma aprendida em vez de fechada algebricamente.
>
> **Aplicação real**: todo forward pass de toda rede neural que você já treinou nesta trilha é, mecanicamente, uma sequência de produtos matriciais. Entender "matriz = transformação" é entender o que uma camada de rede neural literalmente faz com sua entrada — não como analogia, mas como descrição exata.
>
> **Checkpoint**: sem olhar o texto, explique em duas frases o que um autovetor representa geometricamente. Depois, explique por que uma matriz de rank baixo pode ser usada para comprimir informação — e cite onde você já viu essa ideia aplicada nesta trilha.

---

## 1.2 Cálculo

### Conceitos obrigatórios

- Limites e continuidade (intuição, não formalismo pesado).
- Derivada: definição, regras (produto, quociente, **regra da cadeia**).
- Derivadas parciais e gradiente (∇f).
- Jacobiano e Hessiana.
- Integral (definida e indefinida) — menos central em DL, mas aparece em probabilidade.
- Séries de Taylor (justifica linearizações em otimização).

> **Intuição**: a derivada de uma função num ponto é a inclinação da reta tangente ali — "se eu andar um pouquinho pra direita, o quanto a função sobe ou desce?". O **gradiente** (∇f) generaliza isso para funções de várias variáveis: é um vetor apontando na direção de **maior crescimento** da função. É por isso que otimização caminha *contra* o gradiente (gradient descent) — se você quer descer, vá na direção oposta à de maior subida. `optimizer.step()` faz exatamente isso, para milhões de parâmetros simultaneamente, toda vez que você chamou.
>
> **Exemplo resolvido — regra da cadeia**: seja `f(x) = (2x + 1)²`. Podemos ver isso como composição de duas funções: `u(x) = 2x + 1` (a parte de dentro) e `g(u) = u²` (a parte de fora), onde `f(x) = g(u(x))`. A regra da cadeia diz `df/dx = dg/du × du/dx`. Aqui, `dg/du = 2u` e `du/dx = 2`, então `df/dx = 2u × 2 = 4u = 4(2x+1) = 8x + 4`. Confira: expandindo diretamente, `f(x) = 4x² + 4x + 1`, e `df/dx = 8x + 4` — bate. Backpropagation é exatamente isso, só que com uma cadeia muito mais longa: dezenas ou centenas de funções compostas (uma por camada/operação), e a regra da cadeia aplicada de trás pra frente para descobrir a contribuição de cada parâmetro no erro final — é literalmente o que você implementou manualmente em `backward()` no Projeto 5.1, camada por camada, e o Projeto 1.3 (abaixo) generaliza isso num motor que faz esse trabalho para *qualquer* composição de operações, não só uma arquitetura fixa.
>
> **Aplicação real**: quando você entende que backprop é "só" regra da cadeia numa composição longa, "treinar uma rede neural" deixa de ser mágica — é cálculo mecânico, aplicado sistematicamente, exatamente o que você já viu funcionar dezenas de vezes.
>
> **Checkpoint**: sem olhar o texto, explique por que o gradiente aponta na direção de *maior crescimento*, e não na de maior decrescimento. Depois, aplique a regra da cadeia você mesmo em `f(x) = (3x - 2)³` e confira expandindo o cubo.

---

## 1.3 Probabilidade e Estatística

### Conceitos obrigatórios

- Espaço amostral, eventos, probabilidade condicional.
- **Teorema de Bayes** (fundamental para inferência).
- Variáveis aleatórias: discretas vs contínuas.
- Distribuições importantes: Bernoulli, Binomial, Categórica, Gaussiana (Normal), Poisson, Exponencial.
- Esperança, variância, covariância, correlação.
- **Estimador de Máxima Verossimilhança (MLE)** — base do treinamento de quase todo modelo que você já treinou.
- Cross-entropy e KL-divergência (loss functions).
- Inferência Bayesiana (prior, posterior, likelihood).

> **Intuição — Teorema de Bayes**: é a fórmula matemática de "atualizar sua crença diante de nova evidência". Exemplo clássico: um teste médico para uma doença rara (1% da população) tem 95% de acerto (tanto para detectar quem tem quanto pra descartar quem não tem). Você testou positivo — qual a chance de você realmente ter a doença? A intuição ingênua diz "95%", mas Bayes mostra que é bem menor: como a doença é rara, a maioria dos positivos vem de falsos positivos entre a enorme população saudável, não de verdadeiros positivos entre a pequena população doente. Formalmente: `P(doença|positivo) = P(positivo|doença) × P(doença) / P(positivo)` — o "prior" (1% de prevalência) pesa muito no resultado final. Esse mesmo raciocínio (quão surpreendente uma evidência é, dado quão raro o evento já era) é o espírito por trás de calibração de modelo (mod. 14): um classificador bem calibrado "sabe" que um sinal positivo, num contexto de baixa prevalência, não deveria virar confiança de 95%.
>
> **Intuição — MLE e cross-entropy**: MLE pergunta "quais parâmetros tornam os dados que eu observei mais prováveis?". Para classificação, minimizar cross-entropy loss é *matematicamente equivalente* a maximizar a verossimilhança dos rótulos corretos sob a distribuição prevista pelo modelo — são a mesma otimização, só escrita de formas diferentes (cross-entropy é o negativo do log-likelihood). KL-divergência mede o quanto duas distribuições diferem; cross-entropy é KL-divergência mais um termo constante (a entropia da distribuição real) — por isso minimizar uma equivale a minimizar a outra na prática de treino. Essa é a mesma KL-divergência que aparece no `beta` do `DPOTrainer` (Projeto 9.2), controlando o quanto a política pode se afastar da referência.
>
> **Aplicação real**: todo treinamento de LLM por next-token prediction que você já fez (`F.cross_entropy` desde o Projeto 8.3) é MLE disfarçado de cross-entropy loss — o modelo está literalmente aprendendo os parâmetros que tornam o texto de treino mais provável sob a distribuição que ele produz. Sampling com temperature e top-k (mod. 10) manipula diretamente essa distribuição de probabilidade aprendida — o mesmo `F.softmax(logits / T, dim=-1)` que você já escreveu.
>
> **Checkpoint**: sem olhar o texto, explique por que testar positivo num exame raro não significa necessariamente que você tem a doença — o que o prior tem a ver com isso? Depois, explique em uma frase por que "minimizar cross-entropy" e "maximizar verossimilhança" são a mesma coisa.

---

## 1.4 Otimização

### Conceitos obrigatórios

- Função convexa vs não-convexa.
- Mínimo local vs global.
- **Gradient Descent** (vanilla).
- Stochastic Gradient Descent (SGD), mini-batch.
- Momentum, Nesterov.
- Adaptativos: AdaGrad, RMSProp, **Adam**, AdamW — o mesmo `torch.optim.AdamW` que você configurou em praticamente todo projeto de treino desde o Projeto 8.3.
- Learning rate schedules: warmup, cosine, linear decay — o mesmo `warmup`+`cosine` que você configurou explicitamente desde o mod. 09.

> **Intuição**: uma função **convexa** tem um único "vale" — qualquer caminho de descida chega ao mesmo mínimo global, como uma bacia. Uma função **não-convexa** (o caso de praticamente toda rede neural que você já treinou) tem múltiplos vales, platôs e selas — não há garantia de que gradient descent encontre o "melhor" mínimo, só *um* mínimo razoável. Isso é menos assustador na prática do que parece: em redes grandes, a maioria dos mínimos locais encontrados por SGD tem qualidade similar — é por isso que o `MiniLlama` do Projeto 8.3 converge de forma confiável, mesmo sem nenhuma garantia formal de otimalidade.
>
> Você já viu, no mod. 05, um exemplo numérico resolvido passo a passo de gradient descent convergindo (`L(w) = (w-3)²`) e a intuição completa de Momentum/Adam como "inércia" e "passo adaptativo" — essa é a mesma matemática, e este módulo é onde ela nasce formalmente, não onde ela é revisada.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre mínimo local e mínimo global — e por que essa diferença preocupa menos na prática de deep learning do que a definição sugere à primeira vista.

---

## Projetos práticos (curtos, obrigatórios)

Cada projeto deve ser implementado **sem usar bibliotecas que escondam a matemática**. Use NumPy puro. O ponto é "ver" a matemática rodar — a mesma disciplina do Projeto 5.1, aplicada aqui a fundamentos ainda mais básicos.

### Projeto 1.1 — PCA from scratch

Você vai implementar PCA via SVD, sem `sklearn.decomposition.PCA`, e usá-lo para comprimir e reconstruir imagens do MNIST.

**Pré-requisitos**: `pip install numpy matplotlib scikit-learn` (scikit-learn só para carregar o dataset MNIST).

```python
import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data / 255.0  # normaliza pixels para [0, 1]
y = mnist.target.astype(int)

def pca_via_svd(X, n_componentes):
    X_centralizado = X - X.mean(axis=0)  # PCA exige dados centralizados na origem
    U, S, Vt = np.linalg.svd(X_centralizado, full_matrices=False)
    componentes = Vt[:n_componentes]  # as n_componentes direções de maior variância
    X_reduzido = X_centralizado @ componentes.T
    return X_reduzido, componentes, X.mean(axis=0)

def reconstruir(X_reduzido, componentes, media):
    return X_reduzido @ componentes + media
```

`np.linalg.svd` decompõe `X_centralizado` em `U @ diag(S) @ Vt` — `Vt` contém, em cada linha, uma direção do espaço original ordenada por quanto ela captura da variância dos dados (a primeira linha é a direção de maior variância, a Intuição da seção 1.1). Pegar só as primeiras `n_componentes` linhas de `Vt` e projetar os dados nelas (`X_centralizado @ componentes.T`) é exatamente PCA — sem nenhuma chamada a `sklearn`.

**Primeiro, reduza para 2 dimensões e plote**, colorido por dígito (`plt.scatter(X_reduzido[:, 0], X_reduzido[:, 1], c=y)`) — uma verificação visual rápida: dígitos iguais devem formar aglomerados aproximados, mesmo com só 2 das 784 dimensões originais preservadas. Só depois de confirmar isso visualmente, **reduza para 50 dimensões**, reconstrua (`reconstruir(...)`), e compare visualmente uma imagem original com sua versão reconstruída (`plt.imshow(imagem.reshape(28, 28))`) — a reconstrução deve ser reconhecível como o mesmo dígito, mesmo perdendo os detalhes mais finos, prova de que 50 das 784 dimensões já carregam a maior parte da informação relevante.

---

### Projeto 1.2 — Regressão linear from scratch (3 vias)

Você vai resolver o mesmo problema de regressão linear por três caminhos diferentes, e confirmar que os três convergem para (aproximadamente) a mesma resposta.

**Pré-requisitos**: `pip install numpy`.

```python
import numpy as np

np.random.seed(0)
X = np.random.randn(200, 3)
w_verdadeiro = np.array([2.0, -1.0, 0.5])
y = X @ w_verdadeiro + np.random.randn(200) * 0.1  # y = Xw + ruído pequeno

# 1. Solução fechada (equação normal): w = (XᵀX)⁻¹Xᵀy — a resposta exata, de referência
w_fechada = np.linalg.inv(X.T @ X) @ X.T @ y
print("Solução fechada:", w_fechada)

# 2. Gradient descent manual (full-batch, usa todos os dados a cada passo)
w_gd = np.zeros(3)
lr = 0.1
for passo in range(500):
    y_previsto = X @ w_gd
    gradiente = X.T @ (y_previsto - y) / len(y)   # gradiente do MSE em relação a w
    w_gd -= lr * gradiente
print("Gradient descent:", w_gd)

# 3. SGD em mini-batches (usa só um subconjunto dos dados a cada passo)
w_sgd = np.zeros(3)
batch_size = 20
for epoca in range(50):
    indices = np.random.permutation(len(y))
    for inicio in range(0, len(y), batch_size):
        idx_batch = indices[inicio:inicio + batch_size]
        y_previsto = X[idx_batch] @ w_sgd
        gradiente = X[idx_batch].T @ (y_previsto - y[idx_batch]) / batch_size
        w_sgd -= lr * gradiente
print("SGD:", w_sgd)
```

A solução fechada resolve `∂MSE/∂w = 0` analiticamente — não itera, calcula a resposta exata numa única operação matricial, mas exige inverter `XᵀX`, o que fica caro (e às vezes numericamente instável) para muitas features. Gradient descent e SGD chegam à mesma vizinhança por iteração: a diferença entre eles é só quantos exemplos usam para estimar o gradiente a cada passo (todos vs um mini-batch) — SGD é mais barato por passo, mais ruidoso, mas costuma convergir em tempo de relógio comparável ou menor em datasets grandes, o mesmo motivo pelo qual `get_batch` no Projeto 8.3 usa mini-batches, não o corpus inteiro a cada passo.

**Compare os três `w`** — devem ser próximos entre si e próximos de `w_verdadeiro`, já que este é um problema convexo (a Intuição da seção 1.4): não importa o caminho, gradient descent aqui converge para o único mínimo global.

---

### Projeto 1.3 — Autograd minimalista

Você vai implementar um motor de autodiferenciação reversa genérico — não backprop para uma arquitetura específica (que você já fez no Projeto 5.1), mas o mecanismo geral que faz `.backward()` funcionar para *qualquer* composição de operações, o mesmo princípio por trás do `autograd` do PyTorch que você usa desde o Projeto 8.3.

**Pré-requisitos**: nenhuma biblioteca além da padrão.

```python
class Valor:
    def __init__(self, dado, filhos=(), operacao=""):
        self.dado = dado
        self.grad = 0.0
        self._backward = lambda: None   # como propagar o gradiente para os filhos — definido por operação
        self._filhos = filhos
        self._operacao = operacao

    def __add__(self, outro):
        outro = outro if isinstance(outro, Valor) else Valor(outro)
        saida = Valor(self.dado + outro.dado, (self, outro), "+")
        def _backward():
            self.grad += saida.grad     # d(a+b)/da = 1
            outro.grad += saida.grad    # d(a+b)/db = 1
        saida._backward = _backward
        return saida

    def __mul__(self, outro):
        outro = outro if isinstance(outro, Valor) else Valor(outro)
        saida = Valor(self.dado * outro.dado, (self, outro), "*")
        def _backward():
            self.grad += outro.dado * saida.grad    # d(a*b)/da = b
            outro.grad += self.dado * saida.grad    # d(a*b)/db = a
        saida._backward = _backward
        return saida

    def relu(self):
        saida = Valor(max(0, self.dado), (self,), "ReLU")
        def _backward():
            self.grad += (saida.dado > 0) * saida.grad   # derivada de ReLU: 1 se ativou, 0 senão
        saida._backward = _backward
        return saida

    def backward(self):
        ordem_topologica = []
        visitados = set()
        def construir_ordem(no):
            if no not in visitados:
                visitados.add(no)
                for filho in no._filhos:
                    construir_ordem(filho)
                ordem_topologica.append(no)
        construir_ordem(self)

        self.grad = 1.0   # gradiente da saída em relação a si mesma é 1
        for no in reversed(ordem_topologica):
            no._backward()
```

Cada operação (`__add__`, `__mul__`, `relu`) faz duas coisas: calcula o resultado (`self.dado + outro.dado`) e define `_backward`, uma função que sabe como propagar o gradiente da saída de volta para as entradas daquela operação específica — literalmente a regra da cadeia da seção 1.2, uma operação de cada vez. `backward()` primeiro constrói uma ordem topológica (garantindo que cada nó só processe `_backward` depois de todos os nós que dependem dele já terem processado o deles — senão o gradiente ainda não estaria completo), depois percorre essa ordem de trás para frente, chamando `_backward()` em cada nó — exatamente o que o `autograd` do PyTorch faz, em C++, de forma otimizada, toda vez que você chama `.backward()`.

**Comece validando só a multiplicação** antes de tudo o mais: `a = Valor(3.0); b = Valor(4.0); c = a * b; c.backward()` deveria dar `a.grad = 4.0` (`b`) e `b.grad = 3.0` (`a`) — confirme isso manualmente antes de confiar em composições mais longas. Depois, **treine uma rede de 1 camada oculta para resolver XOR** (4 exemplos: `(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0`, um problema clássico que uma única camada linear não resolve, mas uma rede com não-linearidade resolve) usando só `Valor`, sem PyTorch — construindo os pesos como uma lista de `Valor`, o forward pass como composições de `+`/`*`/`relu`, e o loop de treino chamando `.backward()` e atualizando `peso.dado -= lr * peso.grad` manualmente (lembrando de zerar `peso.grad = 0` a cada passo — o mesmo motivo de `optimizer.zero_grad()`).

---

### Projeto 1.4 — Comparativo de otimizadores

Você vai implementar SGD, Momentum e Adam do zero, e comparar como cada um navega uma superfície de loss difícil.

**Pré-requisitos**: `pip install numpy matplotlib`.

```python
import numpy as np

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def gradiente_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def sgd(ponto_inicial, n_passos=1000, lr=0.001):
    ponto = np.array(ponto_inicial, dtype=float)
    trajetoria = [ponto.copy()]
    for _ in range(n_passos):
        grad = gradiente_rosenbrock(*ponto)
        ponto -= lr * grad
        trajetoria.append(ponto.copy())
    return np.array(trajetoria)

def momentum(ponto_inicial, n_passos=1000, lr=0.001, beta=0.9):
    ponto = np.array(ponto_inicial, dtype=float)
    velocidade = np.zeros(2)
    trajetoria = [ponto.copy()]
    for _ in range(n_passos):
        grad = gradiente_rosenbrock(*ponto)
        velocidade = beta * velocidade - lr * grad   # acumula uma média móvel da direção do gradiente
        ponto += velocidade
        trajetoria.append(ponto.copy())
    return np.array(trajetoria)

def adam(ponto_inicial, n_passos=1000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    ponto = np.array(ponto_inicial, dtype=float)
    m, v = np.zeros(2), np.zeros(2)   # primeiro e segundo momento do gradiente
    trajetoria = [ponto.copy()]
    for t in range(1, n_passos + 1):
        grad = gradiente_rosenbrock(*ponto)
        m = beta1 * m + (1 - beta1) * grad          # média móvel do gradiente (momentum)
        v = beta2 * v + (1 - beta2) * grad**2        # média móvel do gradiente ao quadrado (escala adaptativa)
        m_corrigido = m / (1 - beta1**t)             # correção de viés (m/v começam em 0, enviesados no início)
        v_corrigido = v / (1 - beta2**t)
        ponto -= lr * m_corrigido / (np.sqrt(v_corrigido) + eps)
        trajetoria.append(ponto.copy())
    return np.array(trajetoria)
```

`rosenbrock` é uma função clássica de teste para otimizadores, com um "vale" estreito e curvo em forma de banana — fácil de entrar no vale, difícil de convergir rápido até o mínimo dentro dele, porque o gradiente muda de direção abruptamente ao longo do vale. `momentum` acumula uma média móvel (`velocidade`) da direção do gradiente, suavizando o zigue-zague que SGD puro sofre em vales estreitos (a Intuição do mod. 05, revisitada aqui na implementação). `adam` combina isso (`m`, o mesmo papel de `velocidade`) com uma escala de passo adaptativa por dimensão (`v`, que cresce em dimensões onde o gradiente tem sido consistentemente grande, reduzindo o passo ali) — `m_corrigido`/`v_corrigido` compensam o fato de `m`/`v` começarem em zero e ficarem artificialmente pequenos nos primeiros passos.

**Antes de plotar, formule uma hipótese**: qual otimizador você espera que convirja mais rápido no vale estreito da Rosenbrock, baseado na Intuição de Momentum/Adam do mod. 05? Depois, rode os três a partir do mesmo ponto inicial (ex.: `(-1.5, 2.0)`), plote as três trajetórias sobre um contour plot da função (`plt.contour`), e confira se sua previsão bateu.

**Repita o experimento numa rede neural simples** (o MLP do Projeto 5.1, trocando a atualização manual `W -= lr * dW` por cada uma das três regras acima, aplicadas a cada peso) — confirme que a mesma vantagem relativa observada na Rosenbrock (ou a ausência dela) aparece, ou não, no treino de uma rede de verdade.

---

## Erros comuns nesta fase

- **Ler sem implementar.** Matemática não-implementada não vira intuição — mas você já passou dessa fase, tendo implementado tudo antes de chegar formalmente aqui.
- **Achar que, por ter feito tudo na prática primeiro, a formalização é dispensável.** A prática constrói intuição rápido; a formalização é o que permite generalizar essa intuição para problemas que você ainda não viu, e ler pesquisa nova sem depender de outra pessoa "traduzir" a matemática pra você.
- **Achar que `numpy` é trivial.** Não é. Broadcasting, vetorização e *shapes* são onde a maioria dos bugs nasce — você já sentiu isso, provavelmente, em algum projeto anterior.

---

## O que esta matemática já sustentou, retrospectivamente

| Conceito daqui | Onde você já usou, antes de saber o nome formal |
|---|---|
| Produto matricial, transformações lineares | Todo `nn.Linear`, toda projeção Q/K/V desde o Projeto 8.3 |
| Autovalores, SVD, rank baixo | LoRA (mod. 09), PCA (Projeto 1.1) |
| Regra da cadeia | Todo `.backward()`, backprop manual no Projeto 5.1 |
| MLE / Cross-entropy | Toda loss de treino desde o Projeto 8.3 |
| Teorema de Bayes | Calibração e priors (mod. 14) |
| Gradient descent, Adam, warmup | Todo `optimizer.step()` desde o Projeto 8.3 |
| Distribuições, softmax | Sampling em LLMs (temperature, top-k — mod. 10) |
| KL-divergência | O `beta` do DPO (Projeto 9.2) |

---

## Checklist de saída

- [ ] Explico o que é o gradiente de uma função multivariada e por que apontamos *contra* ele em GD (se não, revise a seção 1.2).
- [ ] Derivei à mão a backprop de uma rede com 1 camada oculta, e implementei um motor de autodiferenciação genérico (se não, revise os Projetos 5.1 e 1.3).
- [ ] Explico por que cross-entropy loss é equivalente a MLE para classificação (se não, revise a seção 1.3).
- [ ] Implementei PCA, regressão linear (3 vias) e otimizadores (SGD/Momentum/Adam) em NumPy sem consultar (se não, revise os Projetos 1.1, 1.2 e 1.4).
- [ ] Leio um paper que tenha equações sem travar nas notações ∇, ∂, 𝔼, ∑.

---

## Fim do tronco e dos fundamentos — o que fazer agora

Você completou os 20 módulos desta trilha Acelerado, na ordem 08→20 e depois 07→01: primeiro construindo (um LLM do zero, RAG, agentes, avaliação, produção, visão, RL, multimodal, tópicos de fronteira, projetos integradores), depois formalizando a base que sustentava tudo isso (Transformers, NLP clássico, Deep Learning, ML moderno, ML clássico, ferramentas, matemática). Poucas trilhas de estudo em qualquer área cobrem esse ciclo completo — a maioria para na prática sem a formalização, ou para na teoria sem nunca ter construído nada real.

Você não "termina" IA — a área se move rápido. Mantenha o hábito de acompanhar arXiv (cs.CL, cs.LG) e o Hugging Face Daily Papers como sanity check periódico, ler os blogs de engenharia da Anthropic, OpenAI, Mistral, DeepMind e Google AI, e seguir vozes que traduzem pesquisa em prática com qualidade (Sebastian Raschka, Andrej Karpathy, Lilian Weng, Jay Alammar, Simon Willison). E, acima de tudo: continue construindo. Toda nova técnica ganha solidez quando passa pelas suas próprias mãos — foi assim que esta trilha inteira funcionou, e é assim que o que vem depois dela também vai funcionar.

Se você quiser a outra forma de percorrer este mesmo material — teoria explicada em profundidade antes de qualquer prática, na ordem 01→20 — ela existe nesta mesma trilha, na versão Clássico, acessível pelo seletor de versão no topo do site.

Bom trabalho.
