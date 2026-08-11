---
id: 05_deep_learning
title: "Módulo 05 — Deep Learning (Fundamentos e Arquiteturas)"
sidebar_position: 16
---

# Módulo 05 — Deep Learning (Fundamentos e Arquiteturas)

> **Objetivo**: dominar redes neurais a ponto de implementar do zero (inclusive backprop manual, sem autograd), treinar com confiança, e entender CNNs, RNNs/LSTMs e suas limitações — formalizando o que você já pratica com PyTorch desde o Projeto 8.3.
>
> **Pré-requisitos**: toda a trilha até aqui (módulos [08](08_llms_arquiteturas.md)–[20](20_projetos_integradores.md), [07](07_transformers.mdx), [06](06_nlp_classico.md)) — em particular, AdamW/warmup/cosine decay (mod. 09), RMSNorm (Projeto 8.3), e a conexão residual do bloco Transformer, que aqui você conhece pela origem (ResNet).
>
> **Tempo de referência**: 6–10 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar por que um neurônio precisa de não-linearidade, e derivar isso a partir da definição de combinação linear.
- Explicar por que uma rede mal inicializada não treina (vanishing/exploding), não só citar o termo.
- Justificar cada escolha de regularização (dropout, batchnorm, augmentation) numa CNN, não só aplicá-las por hábito.
- Explicar, com uma frase, por que skip connections permitem redes muito mais profundas — e por que isso é o mesmo princípio por trás da conexão residual que você já usa desde o Projeto 8.3.
- Diagnosticar (e não só nomear) overfitting, gradientes explodindo e loss virando NaN durante um treino real.
- Implementar backpropagation manualmente, sem `autograd`, e entender exatamente o que `.backward()` faz por você todas as vezes que você chamou isso desde o Projeto 8.3.

---

## Por que isso importa

Você já treinou várias redes neurais nesta trilha — sempre com `loss.backward()` e `optimizer.step()` fazendo o trabalho pesado por baixo dos panos. Este módulo abre essa caixa: implementa backpropagation manualmente (Projeto 5.1), e cobre as arquiteturas que vieram antes do Transformer (CNN, RNN/LSTM) e que explicam, por contraste, por que a arquitetura que você já domina na prática venceu. Skip connections, normalização, regularização — tudo que você já usou em RMSNorm, nas conexões residuais do `LlamaStyleBlock`, no AdamW com warmup — tem origem aqui, e entender essa origem é o que transforma "eu sei usar" em "eu sei por que funciona".

---

## 5.1 A rede neural mínima

### Conceitos
- **Neurônio artificial**: combinação linear + não-linearidade.
- **Funções de ativação**: sigmoid, tanh, **ReLU** (e variantes: Leaky ReLU, GELU, Swish/SiLU — o mesmo SiLU que você usou dentro do `SwiGLU` no Projeto 8.3), softmax.
- **Forward pass** vs **backward pass**.
- **Loss functions**: MSE (regressão), Cross-Entropy (classificação — o mesmo `F.cross_entropy` de todo treino que você já fez), BCE (binária), Hinge (SVM-like).
- **Backpropagation** como aplicação da regra da cadeia.

> **Intuição**: um neurônio artificial é um "juiz" que pesa evidências e decide se "ativa" ou não. Cada entrada é uma evidência, cada peso é o quanto aquela evidência importa pra esse juiz específico, e o bias é o quão fácil ou difícil é convencê-lo. A não-linearidade (ReLU, sigmoid...) é o que faz o juiz efetivamente *decidir* algo em vez de só somar números — sem ela, empilhar camadas equivaleria a uma única camada linear, por mais profunda que a rede pareça: a composição de duas transformações lineares ainda é uma transformação linear, então sem não-linearidade no meio, 100 camadas colapsam matematicamente em 1.
>
> **Exemplo resolvido**: entradas `x = [2, 3]`, pesos `w = [1, -0.5]`, bias `b = 0`.
> `z = (2×1) + (3×-0.5) + 0 = 2 - 1.5 = 0.5`
> Aplicando ReLU: `ReLU(0.5) = max(0, 0.5) = 0.5` → o neurônio "ativa" com força 0.5.
> Se a segunda entrada fosse `x = [1, 5]` com os mesmos pesos: `z = 1 - 2.5 = -1.5` → `ReLU(-1.5) = 0` → o neurônio não ativa. Backprop, mais adiante, é só a regra da cadeia aplicada repetidamente pra descobrir *quanto cada peso contribuiu pro erro final* — nada além disso, e é exatamente o que `loss.backward()` fez, escondido, em todo treino que você já rodou desde o Projeto 8.3. A escolha da função de ativação também molda o comportamento do gradiente: sigmoid e tanh "saturam" (derivada perto de zero) para `|z|` grande, o que é a raiz do vanishing gradient discutido na seção 5.2; ReLU tem derivada constante (1) para `z > 0`, por isso é o padrão em redes profundas.
>
> **Aplicação real**: toda camada de todo Transformer que você já implementou é, no fundo, essa mesma combinação linear + não-linearidade, repetida em escala — o `SwiGLU` do Projeto 8.3 é literalmente isso, com uma ativação (SiLU) um pouco mais sofisticada que ReLU. Entender isso aqui, pequeno, é entender o átomo de tudo que você já construiu.
>
> **Checkpoint**: sem olhar o texto acima, explique em duas frases por que uma rede *sem* função de ativação não-linear, não importa quantas camadas tenha, equivale a uma única camada linear. Depois, explique por que sigmoid tende a ter gradientes menores que ReLU.

---

## 5.2 Multi-Layer Perceptron (MLP)

### Tópicos
- Topologia: camadas ocultas, largura, profundidade.
- **Universal Approximation Theorem** — o que ele garante e o que não garante.
- Inicialização de pesos: Xavier/Glorot, He.
- **Vanishing/exploding gradients** — o problema histórico de redes profundas.

> **Intuição**: o Universal Approximation Theorem garante que um MLP largo o suficiente pode aproximar qualquer função contínua — mas "pode aproximar" é diferente de "vai aprender a aproximar" com gradient descent em tempo razoável. É como dizer que qualquer texto pode ser escrito com 26 letras: verdade, mas não te ensina a escrever um romance. A inicialização de pesos é o que decide se o treino sequer sai do lugar.
>
> **Exemplo (ilustrativo)**: imagine uma rede de 50 camadas onde cada camada, por causa da inicialização, multiplica a variância do sinal por ~1.5 ao passar adiante. Depois de 50 camadas, isso é `1.5^50` — um número absurdamente grande, ativações explodindo. Se em vez disso cada camada multiplicasse por ~0.7, seria `0.7^50` — praticamente zero, sinal morto antes de chegar à saída. É exatamente esse efeito multiplicativo, camada após camada, que Xavier/He init tentam neutralizar, escolhendo a escala inicial dos pesos (proporcional a `1/√fan_in` no caso de Xavier, `√(2/fan_in)` no caso de He) para manter a variância do sinal ~estável ao longo da rede, em vez de deixá-la ao acaso.
>
> **Aplicação real**: má inicialização é uma das causas mais comuns de "meu modelo não aprende nada e eu não sei por quê" — antes de suspeitar de bug complexo, verifique a inicialização. É também por isso que frameworks modernos (PyTorch, JAX) já usam He/Kaiming init por padrão em camadas com ReLU — a comunidade aprendeu a lição.
>
> **Checkpoint**: sem consultar o texto, explique por que "a rede pode teoricamente aproximar qualquer função" (Universal Approximation) não é a mesma coisa que "a rede vai aprender essa função na prática". Depois, explique por que He init usa uma escala diferente de Xavier init (dica: tem a ver com o que ReLU faz com metade dos valores).

---

## 5.3 Otimização e regularização para DL

### Otimização
- **SGD**, **SGD com Momentum**, **Nesterov**.
- **Adam, AdamW** — o otimizador que você já usa desde o Projeto 8.3.
- **Learning rate schedules**: step decay, cosine annealing, **warmup** — o mesmo `warmup`+`cosine` que você já configurou no mod. 09.
- **Gradient clipping** (essencial em RNNs e LLMs).

### Regularização
- **L2 weight decay**.
- **Dropout** (e variantes: DropConnect, Spatial Dropout).
- **Data augmentation**.
- **Early stopping**.
- **Label smoothing**.
- **Mixup, CutMix** (técnicas mais modernas, já usadas em prosa no mod. 16).

### Normalização
- **Batch Normalization** (cuidado: comportamento diferente train/eval).
- **Layer Normalization** (padrão no Transformer original — Projeto 7.2).
- **Group Normalization**, **RMSNorm** — o que você já implementou no Projeto 8.3.

> **Intuição**: pense em minimizar a loss como descer uma montanha no escuro, só sentindo a inclinação sob os pés. SGD puro dá um passo na direção da descida mais íngreme e reavalia — pode ficar "zigue-zagueando" em vales estreitos. Momentum é como ter inércia: se você vem descendo numa direção, tende a continuar nela mesmo que o terreno local sugira um desvio pequeno, o que suaviza o zigue-zague. Adam vai além: mantém uma "velocidade média" (momentum) *e* ajusta o tamanho do passo por parâmetro, dando passos maiores em direções onde o gradiente tem sido consistentemente pequeno e menores onde ele oscila muito — exatamente o que `torch.optim.AdamW` faz por você em toda linha `optimizer.step()` que você já escreveu.
>
> **Exemplo resolvido**: loss `L(w) = (w - 3)²`, mínimo em `w = 3`. Derivada: `dL/dw = 2(w - 3)`.
> Começando em `w = 0`, gradiente = `2×(0-3) = -6`. Com learning rate `0.1`: `w_novo = 0 - 0.1×(-6) = 0.6`.
> Próximo passo: gradiente em `w=0.6` é `2×(0.6-3) = -4.8`, `w_novo = 0.6 - 0.1×(-4.8) = 1.08`.
> Note que o passo fica menor conforme `w` se aproxima de 3 (gradiente encolhe) — é assim, mecanicamente, que gradient descent converge. Momentum acumularia parte desses `-6, -4.8, ...` numa média móvel, acelerando a convergência ao longo dessa mesma direção consistente. Regularização (dropout, weight decay, augmentation) ataca um problema diferente: não é sobre convergir mais rápido, é sobre convergir para uma solução que generalize — dropout força a rede a não depender de nenhum neurônio específico (desligando aleatoriamente uma fração deles a cada passo), weight decay penaliza pesos grandes (soluções "mais simples" tendem a generalizar melhor), augmentation multiplica artificialmente a variedade de exemplos vistos.
>
> **Aplicação real**: o pré-treinamento de qualquer LLM moderno (mod. 09, que você já passou) usa AdamW com warmup + cosine decay — exatamente os itens desta lista, só que rodando em milhares de GPUs por semanas. Você já entende por que warmup existe (evitar um passo grande demais antes do otimizador "aquecer" suas estimativas de momento) na prática; aqui você vê a origem formal dessa escolha.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Batch Normalization e Layer Normalization em uma frase cada — e diga qual delas o Transformer original usa. Depois, explique em uma frase por que dropout e weight decay resolvem problemas diferentes (velocidade de convergência vs. generalização) mesmo sendo os dois chamados de "regularização".

---

## 5.4 Convolutional Neural Networks (CNNs)

### Conceitos
- **Convolução discreta**: kernel, stride, padding, receptive field — você já viu isso em prosa e em código NumPy simples no mod. 16 (seção 16.1); aqui é onde a arquitetura completa é construída.
- **Pooling**: max, average, global.
- **Translation invariance** e **parameter sharing** — por que CNN funciona bem em imagens.

### Arquiteturas históricas
- **LeNet-5** (1998) — onde tudo começou.
- **AlexNet** (2012) — virada de chave do DL moderno.
- **VGG** (2014) — simplicidade.
- **GoogLeNet/Inception** (2014) — módulos.
- **ResNet** (2015) — **skip connections**, fundamentais para ir além de 100 camadas, detalhado abaixo.
- **DenseNet**, **EfficientNet**.

> **Intuição**: um kernel de convolução é um "detector de padrão" pequeno (tipo, uma bordinha diagonal) que desliza pela imagem inteira procurando aquele padrão em qualquer posição — é por isso que se chama *parameter sharing*: o mesmo detector é reusado em toda a imagem, em vez de aprender um detector diferente para cada posição de pixel. Isso é o que dá à CNN a propriedade de *translation invariance*: um gato no canto superior esquerdo ativa o mesmo detector que um gato no centro. `stride` controla de quanto em quanto o detector "pula" pela imagem (stride maior = saída menor, menos overlap); `padding` controla o que acontece nas bordas (sem padding, a imagem encolhe a cada convolução); `receptive field` é quanto da imagem original uma unidade da saída "enxerga" — cresce a cada camada empilhada, o que é parte de por que empilhar convoluções ajuda a capturar padrões cada vez mais globais.

### Por que skip connections importam para LLMs
Transformers usam residual connections derivadas das ResNets — o mesmo `x = x + self.attn(...)` que você escreveu no `LlamaStyleBlock` do Projeto 8.3.

> **Intuição**: sem skip connection, o gradiente que volta da última camada até a primeira precisa passar, multiplicando, por *todos* os pesos intermediários — exatamente o mesmo efeito multiplicativo do exemplo de inicialização da seção 5.2, só que agora acontecendo durante o treino inteiro, não só no início. Com poucas dezenas de camadas isso já é suficiente pra o gradiente vanishing antes de chegar às primeiras camadas. A skip connection (`output = F(x) + x`) cria um "atalho" onde o gradiente pode voltar direto pela soma, sem precisar sobreviver a todas as multiplicações — é literalmente uma rota alternativa que ignora o gargalo, e é exatamente essa soma que você já escreve toda vez que faz `x = x + self.attn(self.norm1(x), cos, sin)`.
>
> **Aplicação real**: ResNet vs uma CNN "plain" com o mesmo número de parâmetros é o experimento clássico que prova isso — a plain, a partir de certa profundidade, treina *pior* que uma versão mais rasa dela mesma (não é overfitting, é dificuldade de otimização). Esse é exatamente o resultado que você vai reproduzir no Projeto 5.3, abaixo. Skip connections viraram tão fundamentais que hoje são parte do bloco básico de praticamente toda arquitetura profunda moderna — inclusive o bloco Transformer inteiro tem duas skip connections por camada, uma ao redor da attention e outra ao redor do feed-forward, exatamente como no `LlamaStyleBlock`.
>
> **Checkpoint**: sem olhar o texto, explique por que uma CNN "plain" de 100 camadas costuma treinar pior que uma de 20 camadas — e por que isso não é overfitting.

---

## 5.5 Redes Recorrentes (RNN, LSTM, GRU)

### Por que estudá-las (apesar de obsoletas para grandes textos)
- Foram a forma de modelar sequências antes dos Transformers.
- Permitem entender por que os Transformers ganharam (paralelização, dependências de longo alcance).
- **State Space Models** (Mamba — Projeto 19.2) recuperam ideias recorrentes.

### Conceitos
- **RNN vanilla**: hidden state, BPTT (backprop through time), problema de gradientes.
- **LSTM**: cell state, gates (forget, input, output).
- **GRU**: simplificação do LSTM.
- **Bidirecional RNN**.
- **Encoder-decoder com RNN** (precursor do Seq2Seq que você já implementou com Transformer no Projeto 7.6).
- **Attention sobre RNN** (Bahdanau, Luong) — antecessor direto do self-attention que você já implementou.

> **Intuição**: uma RNN é como ler uma frase palavra por palavra enquanto anota tudo num único caderninho (o hidden state) — cada palavra nova atualiza o caderninho, mas o que foi anotado há 50 palavras já foi bastante reescrito por cima. LSTM/GRU adicionam "portões" (gates) que decidem o que manter, o que esquecer e o que escrever no caderninho, dando mais controle sobre o que sobrevive por muitas palavras — mas o gargalo estrutural continua: informação de uma palavra distante ainda precisa sobreviver passando sequencialmente por *todas* as palavras entre ela e a atual. O "forget gate" do LSTM, por exemplo, aprende quando é seguro "esquecer" parte do cell state; sem ele (RNN vanilla), tudo se mistura sempre.
>
> **Exemplo (ilustrativo)**: em BPTT, o gradiente numa RNN vanilla é multiplicado repetidamente pela mesma matriz de pesos recorrente a cada passo de tempo — o mesmo mecanismo multiplicativo das seções 5.2 e 5.4, agora ao longo do *tempo* em vez das *camadas*. Numa sequência de 100 tokens, isso significa ~100 multiplicações em cadeia: se cada uma encolhe o gradiente um pouco, depois de 100 passos ele já pode ter praticamente zerado. É por isso que RNN vanilla "esquece" contexto distante — e por que mesmo LSTM, que mitiga bastante o problema, ainda degrada em sequências muito longas (centenas ou milhares de tokens).
>
> **Aplicação real**: self-attention (o que você implementou no Projeto 8.3) resolve esse gargalo de um jeito direto: qualquer token pode "olhar" pra qualquer outro token em um único passo, sem depender de uma cadeia sequencial de estados intermediários. Attention sobre RNN (Bahdanau, Luong) foi o primeiro passo nessa direção — deixava o decoder "espiar" diretamente os estados do encoder em vez de confiar só no último hidden state — e é literalmente o antecessor conceitual da cross-attention que você implementou no Projeto 7.6.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase por que um LSTM ainda tem dificuldade com dependências muito longas, mesmo tendo "portões" para controlar memória.

---

## 5.6 Autoencoders e modelos generativos clássicos

- **Autoencoder vanilla**: encoder + decoder, bottleneck.
- **Denoising Autoencoder**.
- **Variational Autoencoder (VAE)**: introdução ao learning variacional.
- **Generative Adversarial Networks (GANs)**: gerador + discriminador.

> **Intuição**: um autoencoder aprende comprimindo e depois reconstruindo — o "gargalo" (bottleneck) no meio da rede é menor que a entrada, então a única forma de reconstruir bem é aprender uma representação compacta que capture o que é essencial e descarte o que é ruído/redundância. É o mesmo princípio de fazer um resumo de um livro: se o resumo cabe em uma página e ainda permite recontar a história, ele capturou a estrutura essencial. Um VAE vai além do autoencoder vanilla ao forçar essa representação compacta a seguir uma distribuição conhecida (normalmente gaussiana) — é isso que permite *gerar* amostras novas depois: basta amostrar um ponto aleatório dessa distribuição e passar pelo decoder. Uma GAN ataca o mesmo problema de geração de outro jeito: um gerador tenta produzir amostras convincentes, um discriminador tenta distinguir real de gerado, e os dois treinam um contra o outro até o gerador ficar bom o suficiente para enganar o discriminador.

### Por que importa
- Embeddings em LLMs são, conceitualmente, "encoders" treinados — o mesmo princípio por trás do `SentenceTransformer` que você usa desde o Projeto 8.5.
- Difusão (Projeto 19.3) é a evolução desses paradigmas.

> **Checkpoint**: sem olhar o texto, explique por que forçar um "gargalo" (bottleneck menor que a entrada) no meio da rede é o que faz o autoencoder aprender algo útil, em vez de só copiar a entrada para a saída.

---

## 5.7 Frameworks: PyTorch a fundo

### Tópicos
- `Tensor`, `autograd`, `nn.Module` — você já usa isso desde o Projeto 8.3; o Projeto 5.1 mostra o que `autograd` faz por baixo, implementando o equivalente manualmente.
- `DataLoader`, `Dataset`, custom datasets.
- Training loop manual: forward → loss → backward → optimizer.step() — o mesmo padrão de todo projeto desde o mod. 08.
- **`torch.compile`** (PyTorch 2.x, JIT).
- **`torch.distributed`**: DDP, FSDP (mod. 09, já visto).
- **PyTorch Lightning** (abstração sobre PyTorch para cortar boilerplate).
- **Hugging Face Accelerate** (alternativa, já mencionada no mod. 09).

> **Visão de mercado**: `autograd` é um dos temas mais comuns em entrevista técnica de ML — saber explicar que `.backward()` percorre o grafo computacional construído dinamicamente durante o forward, acumulando gradientes em `.grad`, e por que `optimizer.zero_grad()` é necessário (porque gradientes *acumulam* por padrão, não substituem, o mesmo detalhe que você já viu comentado em `optimizer.zero_grad()` desde o Projeto 8.3) é o tipo de pergunta que separa quem só usou PyTorch de quem entende o que está fazendo. DDP replica o modelo inteiro em cada GPU e sincroniza gradientes; FSDP particiona o próprio modelo entre GPUs — a diferença entre os dois é exatamente o tipo de trade-off que aparece em decisões reais de infraestrutura de treino.

---

## 5.8 Visualização e debugging de redes neurais

- **TensorBoard** ou **W&B** para acompanhar loss, métricas, distribuições de pesos — usado no Projeto 5.6.
- **Saliency maps**, **Grad-CAM** (interpretação visual).
- **Hooks** em PyTorch para inspecionar ativações.
- **Verificações de sanidade**: overfitar um único batch, gradient checking numérico.

> **Intuição**: depurar uma rede neural é mais parecido com diagnóstico médico do que com debugging de software tradicional — não dá pra colocar um breakpoint na "razão pela qual o modelo não aprende". Em vez disso, você observa sintomas (loss não desce, loss vira NaN, accuracy de treino boa mas validação ruim) e isola causas prováveis por eliminação. `Hooks` em PyTorch são a ferramenta que te dá visibilidade *dentro* da rede durante o forward/backward — sem eles, você só vê a loss final, um sintoma agregado que esconde onde o problema realmente está.
>
> **Cenário hipotético**: imagine que você está treinando e, depois de algumas centenas de passos, a loss vira `NaN`. Causas prováveis, em ordem de frequência: learning rate alto demais (gradiente explode), divisão por zero em alguma loss customizada, `log(0)` numa cross-entropy sem estabilização numérica, ou overflow em precisão mista (`fp16`) sem `loss scaling`. O primeiro passo de diagnóstico costuma ser reduzir o learning rate em 10× e ver se o problema some — se sim, era isso.
>
> **Checkpoint**: sem olhar o texto, liste 2 causas prováveis de um loss virar `NaN` durante o treino.

---

## Projetos práticos

### Projeto 5.1 — MLP from scratch em NumPy, com backprop manual

Você vai implementar forward pass, backward pass e treino inteiramente em NumPy, sem `autograd` — exatamente o que `.backward()` fez por você em todo projeto anterior, agora explícito.

**Pré-requisitos**: `pip install numpy` (sem PyTorch nesta parte, de propósito).

**1. Forward pass** (2 camadas: entrada → oculta com ReLU → saída com softmax):

```python
import numpy as np

def inicializar_pesos(dim_entrada, dim_oculta, dim_saida):
    # He init, como discutido na seção 5.2 — escala proporcional a sqrt(2/fan_in)
    W1 = np.random.randn(dim_entrada, dim_oculta) * np.sqrt(2.0 / dim_entrada)
    b1 = np.zeros(dim_oculta)
    W2 = np.random.randn(dim_oculta, dim_saida) * np.sqrt(2.0 / dim_oculta)
    b2 = np.zeros(dim_saida)
    return W1, b1, W2, b2

def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    cache = (X, z1, a1, z2, a2)  # guarda tudo que o backward vai precisar
    return a2, cache
```

`cache` guarda todos os valores intermediários do forward — o backward precisa deles para calcular os gradientes, e é exatamente o que `autograd` faz automaticamente ao construir o grafo computacional durante o forward, que você usa sem ver desde o Projeto 8.3. **Verifique antes de prosseguir**: a soma de cada linha de `a2` (a saída) deve ser 1 (é uma distribuição de probabilidade válida, propriedade do softmax) — confirme com `a2.sum(axis=-1)`.

**2. Backward pass** (a regra da cadeia, aplicada camada por camada, de trás para frente):

```python
def backward(y_verdadeiro, cache, W2):
    X, z1, a1, z2, a2 = cache
    n = X.shape[0]

    dz2 = a2.copy()
    dz2[range(n), y_verdadeiro] -= 1   # gradiente de cross-entropy + softmax combinados (simplificação clássica)
    dz2 /= n

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)   # derivada de ReLU: 1 onde z1 > 0, senão 0

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dW1, db1, dW2, db2
```

`dz2 = a2.copy(); dz2[range(n), y_verdadeiro] -= 1` é a derivada combinada de cross-entropy+softmax — uma simplificação algébrica clássica que evita calcular as duas derivadas separadamente (o cálculo isolado é mais longo, mas se simplifica exatamente nisso). Cada linha seguinte é a regra da cadeia, uma camada de cada vez, de trás para frente: `dW2`/`db2` vêm de como a saída da camada 2 depende de `W2`/`b2`; `da1` propaga o gradiente de volta para a camada oculta; `dz1 = da1 * (z1 > 0)` aplica a derivada do ReLU (1 onde a entrada era positiva, 0 onde era negativa — o mesmo "portão" binário mencionado na seção 5.1); `dW1`/`db1` fecham o ciclo.

**3. Valide com gradient checking numérico antes de treinar** — compare o gradiente analítico (calculado acima) com uma aproximação numérica (perturbar um peso levemente e medir a mudança na loss):

```python
def gradient_check(W, dW, X, y, forward_fn, loss_fn, epsilon=1e-5):
    i, j = np.random.randint(W.shape[0]), np.random.randint(W.shape[1])
    W_mais = W.copy(); W_mais[i, j] += epsilon
    W_menos = W.copy(); W_menos[i, j] -= epsilon
    grad_numerico = (loss_fn(forward_fn(X, W_mais)) - loss_fn(forward_fn(X, W_menos))) / (2 * epsilon)
    diferenca_relativa = abs(grad_numerico - dW[i, j]) / (abs(grad_numerico) + abs(dW[i, j]) + 1e-8)
    print(f"Gradiente analítico: {dW[i, j]:.6f}, numérico: {grad_numerico:.6f}, diferença relativa: {diferenca_relativa:.6f}")
```

Uma diferença relativa maior que ~1e-4 indica um bug no backward analítico — rode isso **antes** de confiar em qualquer treino longo.

**4. Loop de treino completo** (o mesmo padrão de sempre, com `W -= lr * dW` no lugar de `optimizer.step()`):

```python
W1, b1, W2, b2 = inicializar_pesos(784, 128, 10)  # MNIST: 784 pixels de entrada, 10 classes
lr = 0.1

for epoca in range(20):
    a2, cache = forward(X_treino, W1, b1, W2, b2)
    loss = -np.mean(np.log(a2[range(len(y_treino)), y_treino] + 1e-8))
    dW1, db1, dW2, db2 = backward(y_treino, cache, W2)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoca % 5 == 0:
        acc = (a2.argmax(axis=-1) == y_treino).mean()
        print(f"época {epoca}: loss={loss:.4f} acc={acc:.4f}")
```

Treine no MNIST (baixe via `sklearn.datasets.fetch_openml("mnist_784")` ou `torchvision.datasets.MNIST` só para os dados, sem usar `torch.nn`) até ultrapassar 97% de accuracy — inteiramente sem PyTorch. Compare o tempo de treino com uma versão equivalente em PyTorch (reaproveitando o padrão de treino do Projeto 8.3, com `nn.Linear`+`nn.CrossEntropyLoss`) — a versão PyTorch deve ser mais rápida (usa BLAS otimizado e, se disponível, GPU), mesmo fazendo matematicamente a mesma coisa.

---

### Projeto 5.2 — CNN no CIFAR-10

Você vai construir uma CNN em PyTorch, adicionando técnicas de regularização uma de cada vez, medindo o ganho isolado de cada uma.

**Pré-requisitos**: `pip install torch torchvision`.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

transformacoes_treino = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transformacoes_teste = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

treino = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transformacoes_treino)
teste = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transformacoes_teste)
loader_treino = torch.utils.data.DataLoader(treino, batch_size=128, shuffle=True)
loader_teste = torch.utils.data.DataLoader(teste, batch_size=256)

class CNNSimples(nn.Module):
    def __init__(self, usar_batchnorm=False, dropout=0.0):
        super().__init__()
        camadas = []
        canais = [3, 32, 64, 128]
        for i in range(3):
            camadas.append(nn.Conv2d(canais[i], canais[i + 1], kernel_size=3, padding=1))
            if usar_batchnorm:
                camadas.append(nn.BatchNorm2d(canais[i + 1]))
            camadas.append(nn.ReLU())
            camadas.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*camadas)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
```

`RandomCrop`+`RandomHorizontalFlip` (data augmentation) só se aplicam ao conjunto de treino, nunca ao de teste — aumentar artificialmente a variedade de exemplos vistos no treino, mas avaliar sempre nos dados reais, sem distorção. `nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1)` é a camada convolucional da seção 5.4: `padding=1` mantém o tamanho espacial constante (sem isso, a imagem encolheria a cada camada); `nn.MaxPool2d(2)` reduz a resolução espacial pela metade a cada bloco, aumentando o receptive field das camadas seguintes.

**Adicione cada técnica uma de cada vez, medindo o ganho isolado**: treine `CNNSimples(usar_batchnorm=False, dropout=0.0)` primeiro (baseline), depois só `usar_batchnorm=True`, depois `usar_batchnorm=True, dropout=0.3`, depois adicione o learning rate schedule (`torch.optim.lr_scheduler.CosineAnnealingLR`, o mesmo princípio de cosine decay do mod. 09) — reaproveitando o loop de treino padrão de todo projeto desde o mod. 08 (`forward` → `loss` → `zero_grad` → `backward` → `step`). Registre a accuracy final de cada configuração numa tabela, para poder atribuir o ganho a cada técnica especificamente, em vez de uma "sopa" de mudanças simultâneas. Meta: >85% accuracy no conjunto de teste.

---

### Projeto 5.3 — Implementar ResNet pequena, e provar por que skip connections importam

Você vai implementar blocos residuais à mão e comparar diretamente contra uma CNN "plain" do mesmo tamanho, isolando o efeito da profundidade.

**Pré-requisitos**: os mesmos do Projeto 5.2.

```python
class BlocoResidual(nn.Module):
    def __init__(self, canais):
        super().__init__()
        self.conv1 = nn.Conv2d(canais, canais, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(canais)
        self.conv2 = nn.Conv2d(canais, canais, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(canais)

    def forward(self, x):
        identidade = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identidade   # a skip connection — o "+x" da seção 5.4
        return torch.relu(out)

class ResNetPequena(nn.Module):
    def __init__(self, n_blocos=10, canais=32):
        super().__init__()
        self.entrada = nn.Conv2d(3, canais, kernel_size=3, padding=1)
        self.blocos = nn.Sequential(*[BlocoResidual(canais) for _ in range(n_blocos)])
        self.fc = nn.Linear(canais * 32 * 32, 10)

    def forward(self, x):
        x = torch.relu(self.entrada(x))
        x = self.blocos(x)
        return self.fc(x.flatten(1))

class CNNPlana(nn.Module):
    def __init__(self, n_blocos=10, canais=32):
        super().__init__()
        self.entrada = nn.Conv2d(3, canais, kernel_size=3, padding=1)
        camadas = []
        for _ in range(n_blocos):
            camadas += [nn.Conv2d(canais, canais, kernel_size=3, padding=1), nn.BatchNorm2d(canais), nn.ReLU(),
                        nn.Conv2d(canais, canais, kernel_size=3, padding=1), nn.BatchNorm2d(canais), nn.ReLU()]
        self.blocos = nn.Sequential(*camadas)
        self.fc = nn.Linear(canais * 32 * 32, 10)

    def forward(self, x):
        x = torch.relu(self.entrada(x))
        x = self.blocos(x)
        return self.fc(x.flatten(1))
```

`out = out + identidade` em `BlocoResidual.forward` é a skip connection — a única diferença estrutural entre `ResNetPequena` e `CNNPlana`, que por lo demais têm exatamente o mesmo número de camadas convolucionais e parâmetros.

**Confirme primeiro numa profundidade rasa** (`n_blocos=3`), onde ambas deveriam treinar bem — se uma delas já falha aqui, o bug está em outro lugar, não na hipótese sobre profundidade. Só depois, treine as duas em `n_blocos=15` ou mais no CIFAR-10, e compare a curva de loss de treino (não só accuracy final): a expectativa, confirmando a Intuição da seção 5.4, é que `CNNPlana` treine visivelmente pior nessa profundidade maior — não porque está *overfitando* (a loss de treino dela também fica pior, não só a de validação), mas porque o gradiente tem dificuldade de propagar até as primeiras camadas sem a rota de atalho.

---

### Projeto 5.4 — char-RNN (gerador de texto caractere a caractere) com LSTM

Você vai implementar um LSTM em PyTorch para geração de texto, e comparar diretamente com o `MiniLlama` (Transformer) do Projeto 8.3 no mesmo corpus.

**Pré-requisitos**: os mesmos do Projeto 8.3 (reaproveita `encode`/`decode`/`train_data` do Tiny Shakespeare).

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, dim_embedding=128, dim_hidden=256, n_camadas=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.lstm = nn.LSTM(dim_embedding, dim_hidden, n_camadas, batch_first=True)
        self.head = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, estado_oculto=None):
        x = self.embedding(x)
        saida, estado_oculto = self.lstm(x, estado_oculto)
        logits = self.head(saida)
        return logits, estado_oculto
```

`nn.LSTM` já implementa internamente os gates (forget, input, output) da seção 5.5 — você não precisa escrevê-los à mão para este projeto (diferente do Projeto 5.1, o foco aqui é comparar arquiteturas, não reimplementar LSTM do zero). `estado_oculto` (o par hidden state + cell state) é passado explicitamente entre chamadas, o que o distingue estruturalmente da self-attention: um LSTM processa a sequência *sequencialmente*, carregando estado; o Transformer processa a sequência inteira em paralelo, sem estado carregado entre posições.

Treine com o mesmo loop e o mesmo Tiny Shakespeare do Projeto 8.3 (`get_batch`, `AdamW`, `cross_entropy`), gerando texto a cada poucas centenas de passos para acompanhar qualitativamente o progresso. **Compare com o `MiniLlama`** treinado no mesmo corpus, pelo mesmo número de passos: além da qualidade do texto gerado, meça o tempo de treino por passo (LSTM processa sequencialmente, então tende a ser mais lento para treinar em paralelo que o Transformer, que processa a sequência inteira de uma vez) — uma demonstração direta e pessoal do argumento de paralelização da seção 5.5.

---

### Projeto 5.5 — VAE no MNIST

Você vai implementar um Variational Autoencoder completo, incluindo o reparameterization trick, e visualizar o espaço latente aprendido.

**Pré-requisitos**: `pip install torch torchvision matplotlib`.

**Antes de implementar, entenda por que o reparameterization trick é necessário**: o encoder de um VAE produz os parâmetros (média `μ`, desvio padrão `σ`) de uma distribuição, e o decoder recebe uma *amostra* dessa distribuição — mas amostragem estocástica direta (`z = amostra_de(Normal(μ, σ))`) não é uma operação diferenciável, então o gradiente não conseguiria fluir de volta através dela até `μ`/`σ` durante o backward. O truque: reescrever `z = μ + σ * ε`, onde `ε` é amostrado de uma `Normal(0, 1)` *fixa* (sem parâmetros aprendidos) — agora a aleatoriedade está isolada em `ε`, e `z` é uma função diferenciável de `μ` e `σ`.

```python
class VAE(nn.Module):
    def __init__(self, dim_entrada=784, dim_latente=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_entrada, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, dim_latente)
        self.fc_logvar = nn.Linear(400, dim_latente)
        self.decoder = nn.Sequential(
            nn.Linear(dim_latente, 400), nn.ReLU(),
            nn.Linear(400, dim_entrada), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps   # o reparameterization trick
        return self.decoder(z), mu, logvar

def loss_vae(x_reconstruido, x_original, mu, logvar):
    reconstrucao = nn.functional.binary_cross_entropy(x_reconstruido, x_original, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstrucao + kl_divergence
```

A loss tem duas partes: `reconstrucao` mede quão bem o decoder reconstrói a entrada original a partir de `z` (o mesmo princípio de qualquer autoencoder, seção 5.6); `kl_divergence` penaliza a distribuição aprendida (`μ`, `σ` por exemplo) por se afastar de uma `Normal(0, 1)` padrão — é essa penalidade que força o espaço latente a ficar organizado de um jeito que permite amostrar pontos novos e obter reconstruções coerentes, não só memorizar cada exemplo de treino num canto isolado do espaço.

Treine no MNIST (reaproveitando o `DataLoader` do Projeto 5.2, adaptado — imagens achatadas em vetores de 784, não mantidas como grade 2D, já que este VAE usa camadas lineares). **Visualize o espaço latente em 2D**: treine uma segunda versão com `dim_latente=2` especificamente para poder plotar (com `matplotlib`, o padrão desde o Projeto 8.4) cada dígito do conjunto de teste como um ponto `(z[0], z[1])`, colorido por classe — dígitos parecidos (como 4 e 9) devem ficar visualmente próximos no espaço. **Gere amostras interpolando**: escolha dois pontos `z_a` e `z_b` do espaço latente (de dois dígitos diferentes, digamos um "3" e um "8"), gere pontos intermediários (`z_a + t*(z_b - z_a)` para `t` de 0 a 1) e passe cada um pelo decoder — a sequência de imagens geradas deve mostrar uma transição suave de um dígito para o outro.

---

### Projeto 5.6 — Pipeline de experimentação completo com W&B

Você vai rodar uma grade sistemática de experimentos (3 arquiteturas × 3 learning rates × 2 batch sizes) e usar o Weights & Biases para registrar e comparar todos os resultados — a mesma disciplina de observability do mod. 15, aplicada a treino em vez de produção.

**Pré-requisitos**: `pip install wandb`, conta gratuita em wandb.ai.

```python
import wandb

arquiteturas = {
    "cnn_simples": lambda: CNNSimples(usar_batchnorm=True, dropout=0.3),   # do Projeto 5.2
    "resnet_pequena": lambda: ResNetPequena(n_blocos=5),                    # do Projeto 5.3
    "resnet_funda": lambda: ResNetPequena(n_blocos=15),
}
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [64, 128]

for nome_arq, construtor in arquiteturas.items():
    for lr in learning_rates:
        for bs in batch_sizes:
            wandb.init(project="cifar10-grid", config={"arquitetura": nome_arq, "lr": lr, "batch_size": bs}, reinit=True)

            modelo = construtor()
            loader = torch.utils.data.DataLoader(treino, batch_size=bs, shuffle=True)
            optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr)

            for epoca in range(10):
                for x, y in loader:
                    logits = modelo(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                wandb.log({"epoca": epoca, "loss": loss.item()})

            acc_final = avaliar(modelo, loader_teste)  # reaproveita a lógica de avaliação do Projeto 5.2
            wandb.log({"accuracy_final": acc_final})
            wandb.finish()
```

`wandb.init(..., config={...})` registra os hiperparâmetros daquela execução específica; `wandb.log(...)` registra métricas ao longo do treino (visualizáveis como gráficos no painel web do W&B, sem você precisar plotar nada manualmente); `reinit=True` permite iniciar uma nova run dentro do mesmo loop Python, uma por combinação de hiperparâmetros. Ao final das 18 combinações, o painel do W&B permite ordenar e filtrar por `accuracy_final`, comparando visualmente o efeito de cada hiperparâmetro.

**Antes de olhar a tabela final, documente uma hipótese para cada resultado**: qual arquitetura você espera que vença? Qual learning rate é grande demais para a ResNet funda (dado que redes mais profundas costumam ser mais sensíveis a learning rate alto)? Comparar sua hipótese com o resultado real constrói intuição de hiperparâmetro mais rápido do que só rodar e observar depois.

---

## Erros comuns

- **Não fazer o "overfitar 1 batch"** — primeira sanity check de qualquer treinamento: se o modelo não consegue nem memorizar um único batch pequeno, há um bug de configuração, não de escala.
- **Misturar `model.train()` e `model.eval()`** — BatchNorm e Dropout dependem disso (BatchNorm usa estatísticas de batch em train, estatísticas acumuladas em eval).
- **Esquecer `optimizer.zero_grad()`** — gradientes acumulam (o mesmo erro já visto na seção 5.7, e desde o Projeto 8.3).
- **Comparar runs com seeds diferentes** sem rodar várias seeds.
- **Acreditar que "mais profundo é sempre melhor"** — sem skip connections, não é (Projeto 5.3).
- **Treinar sem learning rate scheduler** em qualquer treinamento longo.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Backpropagation | Treinamento de Transformers (já usado desde o mod. 08) |
| Skip connections | Transformers (já usado no Projeto 8.3) |
| Layer Normalization | Transformers (Projeto 7.2) |
| Adam, warmup | Treinamento de LLMs (mod. 09, já visto) |
| RNN encoder-decoder | Conceito que Transformers substituem (Projeto 7.6) |
| Attention (Bahdanau) | Self-attention (já implementada no Projeto 8.3) |
| Embeddings em autoencoders | Word embeddings (mod. 06) |

---

## Checklist de saída

- [ ] Implementei MLP e backprop em NumPy puro, validado com gradient checking numérico (se não, revise o Projeto 5.1).
- [ ] Treinei pelo menos uma CNN com >85% no CIFAR-10, medindo o ganho isolado de cada técnica de regularização (se não, revise o Projeto 5.2).
- [ ] Implementei ResNet e CNN plana com o mesmo tamanho, e observei o gap de treino em profundidade alta, não só em teoria (se não, revise o Projeto 5.3).
- [ ] Implementei e treinei um LSTM numa tarefa de sequência, comparando com Transformer no mesmo corpus (se não, revise o Projeto 5.4).
- [ ] Implementei um VAE completo, incluindo o reparameterization trick, e visualizei o espaço latente (se não, revise o Projeto 5.5).
- [ ] Sei diagnosticar overfitting, vanishing gradients, loss NaN (se não, revise a seção 5.8).
- [ ] Rodei uma grade de experimentos com W&B e comparei hipóteses com resultados reais (se não, revise o Projeto 5.6).
- [ ] Sei explicar por que skip connections permitem redes muito profundas (se não, revise a seção 5.4).
