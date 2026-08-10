---
id: 05_deep_learning
title: "Módulo 05 — Deep Learning (Fundamentos e Arquiteturas)"
sidebar_position: 5
---

# Módulo 05 — Deep Learning (Fundamentos e Arquiteturas)

> **Objetivo**: dominar redes neurais a ponto de implementar do zero, treinar com confiança, e entender CNNs, RNNs/LSTMs e suas limitações — preparando o terreno conceitual para Transformers (mod. [07](07_transformers.mdx)).
>
> **Pré-requisitos**: Módulos [01](01_matematica.md)–[04](04_ml_moderno.md).
>
> **Tempo de referência**: 6–10 semanas.

> **Tempo por caminho**: **Clássico** ~8–10 semanas (teoria completa + prática, nessa ordem) · **Acelerado** ~2–3 semanas (só prática guiada, com código, sem derivação teórica).

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Implementar forward e backward pass de um MLP em NumPy puro, sem framework.
- Explicar por que uma rede mal inicializada não treina (vanishing/exploding), não só citar o termo.
- Treinar uma CNN competitiva no CIFAR-10 e justificar cada escolha de regularização usada.
- Explicar, com uma frase, por que skip connections permitem redes muito mais profundas — e por que isso importa para Transformers.
- Diagnosticar (e não só nomear) overfitting, gradientes explodindo e loss virando NaN durante um treino real.
- Navegar o ciclo completo dataset → DataLoader → modelo → loss → optimizer → log → checkpoint sem copiar um template.

## Diagnóstico rápido

Responda sim/não pra cada afirmação — te ajuda a decidir o ritmo, não é teste eliminatório:

- Já implementei backpropagation à mão (com papel ou código), sem usar `autograd`.
- Sei explicar, sem consultar nada, por que ReLU tem "vanishing gradient" menor que sigmoid.
- Já treinei uma rede que divergiu (loss virou NaN) e sei ao menos 2 causas prováveis.
- Sei a diferença prática entre Batch Norm e Layer Norm — não só a fórmula, o *quando usar cada uma*.
- Já usei `torch.distributed` ou `DataParallel` em algum treino, mesmo que pequeno.

Poucos "sim" → siga o caminho **Clássico** (teoria completa e depois prática). Maioria "sim" → vá direto pelos blocos **Prática guiada — Acelerado**, que assumem que você já manja da teoria e só quer construir — mas ainda te dão código, não só uma lista de tarefas.

Este módulo tem dois trilhos em paralelo: um de teoria (intuição, exemplo resolvido com código, aplicação real, checkpoint) pra quem está construindo a base, e um de prática guiada — passos com esqueleto de código pronto pra rodar, faltando só as linhas-chave que forçam você a aplicar o conceito — pra quem já tem bagagem e quer ir direto pro código sem reler teoria.

---

## Por que isso importa

Pular DL e ir direto para LLMs é como tentar entender Relatividade sem Mecânica Newtoniana. Transformers são uma arquitetura *neural*, com gradient descent, backprop, regularização — tudo mora aqui. Sem isso, você não tem intuição.

---

## 5.1 A rede neural mínima

### Conceitos
- **Neurônio artificial**: combinação linear + não-linearidade.
- **Funções de ativação**: sigmoid, tanh, **ReLU** (e variantes: Leaky ReLU, GELU, Swish/SiLU), softmax.
- **Forward pass** vs **backward pass**.
- **Loss functions**: MSE (regressão), Cross-Entropy (classificação), BCE (binária), Hinge (SVM-like).
- **Backpropagation** como aplicação da regra da cadeia.

> **Intuição**: um neurônio artificial é um "juiz" que pesa evidências e decide se "ativa" ou não. Cada entrada é uma evidência, cada peso é o quanto aquela evidência importa pra esse juiz específico, e o bias é o quão fácil ou difícil é convencê-lo. A não-linearidade (ReLU, sigmoid...) é o que faz o juiz efetivamente *decidir* algo em vez de só somar números — sem ela, empilhar camadas equivaleria a uma única camada linear, por mais profunda que a rede pareça.
>
> **Exemplo resolvido**: entradas `x = [2, 3]`, pesos `w = [1, -0.5]`, bias `b = 0`.
> `z = (2×1) + (3×-0.5) + 0 = 2 - 1.5 = 0.5`
> Aplicando ReLU: `ReLU(0.5) = max(0, 0.5) = 0.5` → o neurônio "ativa" com força 0.5.
>
> Em código (você pode rodar isso exatamente como está):
>
> ```python
> def relu(z):
>     return max(0, z)
>
> def neuron(x, w, b, activation=relu):
>     z = sum(xi * wi for xi, wi in zip(x, w)) + b
>     return activation(z)
>
> x = [2, 3]
> w = [1, -0.5]
> b = 0
>
> print(neuron(x, w, b))        # 0.5
> print(neuron([1, 5], w, b))   # 0  (a segunda entrada "não ativa" o neurônio)
> ```
>
> Backprop, mais adiante, é só a regra da cadeia aplicada repetidamente pra descobrir *quanto cada peso contribuiu pro erro final* — nada além disso.
>
> **Aplicação real**: toda camada de todo Transformer (inclusive os LLMs que você vai estudar a partir do mod. [07](07_transformers.mdx)) é, no fundo, essa mesma combinação linear + não-linearidade, repetida em escala. Entender isso aqui, pequeno, é entender o átomo de tudo que vem depois.
>
> **Checkpoint**: sem olhar o texto acima, explique em duas frases por que uma rede *sem* função de ativação não-linear, não importa quantas camadas tenha, equivale a uma única camada linear.

> **Prática guiada — Acelerado**
>
> Rode o código do bloco de teoria acima primeiro (`neuron`), depois generalize:
>
> ```python
> import math
>
> def sigmoid(z):
>     return 1 / (1 + math.exp(-z))
>
> def tanh(z):
>     return math.tanh(z)
>
> def relu(z):
>     return max(0, z)
>
> def neuron(x, w, b, activation):
>     z = sum(xi * wi for xi, wi in zip(x, w)) + b
>     return activation(z)
>
> x = [1, 5]
> w = [1, -0.5]
> b = 0
>
> for act in (sigmoid, tanh, relu):
>     print(act.__name__, neuron(x, w, b, act))
> ```
>
> 1. Rode o código acima e confira que as três ativações dão saídas diferentes para o mesmo `z`.
> 2. Usando `matplotlib`, plote as três funções (`sigmoid`, `tanh`, `relu`) para `z` de -5 a 5 num mesmo gráfico:
>    ```python
>    import matplotlib.pyplot as plt
>
>    zs = [z / 10 for z in range(-50, 51)]
>    plt.plot(zs, [sigmoid(z) for z in zs], label="sigmoid")
>    plt.plot(zs, [tanh(z) for z in zs], label="tanh")
>    plt.plot(zs, [relu(z) for z in zs], label="relu")
>    plt.legend()
>    plt.show()
>    ```
> 3. Observe no gráfico: onde sigmoid e tanh "achatam" (derivada perto de zero) — é ali que o vanishing gradient começa.

### Implementação from scratch
Implemente em NumPy puro: forward, backward, treinamento, validação.

### Referências
- `Livro` **Deep Learning** (Goodfellow et al.), cap. 6. https://www.deeplearningbook.org/
- `Curso` **Karpathy — Neural Networks: Zero to Hero** (série completa, valor inestimável). https://karpathy.ai/zero-to-hero.html
- `Curso` **3Blue1Brown — Neural Networks** (4 vídeos clássicos). https://www.3blue1brown.com/topics/neural-networks
- `Paper` **Learning representations by back-propagating errors** — Rumelhart, Hinton, Williams (1986). Paper histórico, vale ler para perspectiva.

---

## 5.2 Multi-Layer Perceptron (MLP)

### Tópicos
- Topologia: camadas ocultas, largura, profundidade.
- **Universal Approximation Theorem** — o que ele garante e o que não garante.
- Inicialização de pesos: Xavier/Glorot, He.
- **Vanishing/exploding gradients** — o problema histórico de redes profundas.

> **Intuição**: o Universal Approximation Theorem garante que um MLP largo o suficiente pode aproximar qualquer função contínua — mas "pode aproximar" é diferente de "vai aprender a aproximar" com gradient descent em tempo razoável. É como dizer que qualquer texto pode ser escrito com 26 letras: verdade, mas não te ensina a escrever um romance. A inicialização de pesos é o que decide se o treino sequer sai do lugar.
>
> **Exemplo (código real)**: o snippet abaixo empilha 30 camadas lineares sem ativação e mostra a norma do sinal explodindo, murchando, ou ficando estável, dependendo só da escala inicial dos pesos.
>
> ```python
> import numpy as np
>
> def run_stack(std, n_layers=30, dim=100, seed=0):
>     rng = np.random.default_rng(seed)
>     x = rng.standard_normal(dim)
>     norms = [np.linalg.norm(x)]
>     for _ in range(n_layers):
>         W = rng.standard_normal((dim, dim)) * std
>         x = W @ x
>         norms.append(np.linalg.norm(x))
>     return norms
>
> print("std=2.0 (grande):", run_stack(std=2.0)[-1])
> print("std=0.01 (pequeno):", run_stack(std=0.01)[-1])
> print("He init (sqrt(2/dim)):", run_stack(std=np.sqrt(2 / 100))[-1])
> ```
>
> Rode e compare os três números finais: o primeiro deve ser um número gigante, o segundo praticamente zero, e o terceiro (He init) deve ficar numa faixa razoável — é exatamente esse efeito multiplicativo, camada após camada, que Xavier/He init tentam neutralizar.
>
> **Aplicação real**: má inicialização é uma das causas mais comuns de "meu modelo não aprende nada e eu não sei por quê" — antes de suspeitar de bug complexo, verifique a inicialização.
>
> **Checkpoint**: sem consultar o texto, explique por que "a rede pode teoricamente aproximar qualquer função" (Universal Approximation) não é a mesma coisa que "a rede vai aprender essa função na prática".

> **Prática guiada — Acelerado**
>
> ```python
> import numpy as np
>
> def run_stack(std, n_layers=30, dim=100, seed=0):
>     rng = np.random.default_rng(seed)
>     x = rng.standard_normal(dim)
>     norms = [np.linalg.norm(x)]
>     for _ in range(n_layers):
>         W = rng.standard_normal((dim, dim)) * std
>         # TODO: aplique a transformação linear desta camada (x = W @ x)
>         x = ...
>         norms.append(np.linalg.norm(x))
>     return norms
> ```
>
> 1. Complete o `TODO` acima e rode com `std=2.0`, `std=0.01` e `std=np.sqrt(2/100)`.
> 2. Plote as três listas de normas (`norms`) num gráfico de linha (eixo x = camada, eixo y = norma) — visualize a explosão/vanish acontecendo camada a camada, não só no valor final.
> 3. Repita substituindo a multiplicação pura `W @ x` por `np.maximum(0, W @ x)` (ReLU) em cada camada e compare se o padrão muda.

### Referências
- `Paper` **Understanding the difficulty of training deep feedforward neural networks** — Glorot & Bengio (2010). https://proceedings.mlr.press/v9/glorot10a.html
- `Paper` **Delving Deep into Rectifiers (He init)** — He et al. (2015). https://arxiv.org/abs/1502.01852

---

## 5.3 Otimização e regularização para DL

### Otimização
- **SGD**, **SGD com Momentum**, **Nesterov**.
- **Adam, AdamW** (revisar do mod. [01](01_matematica.md) com mais profundidade).
- **Learning rate schedules**: step decay, cosine annealing, **warmup**.
- **Gradient clipping** (essencial em RNNs e LLMs).

### Regularização
- **L2 weight decay**.
- **Dropout** (e variantes: DropConnect, Spatial Dropout).
- **Data augmentation**.
- **Early stopping**.
- **Label smoothing**.
- **Mixup, CutMix** (técnicas mais modernas).

### Normalização
- **Batch Normalization** (cuidado: comportamento diferente train/eval).
- **Layer Normalization** (padrão em Transformers).
- **Group Normalization**, **RMSNorm** (usado em LLaMA).

> **Intuição**: pense em minimizar a loss como descer uma montanha no escuro, só sentindo a inclinação sob os pés. SGD puro dá um passo na direção da descida mais íngreme e reavalia — pode ficar "zigue-zagueando" em vales estreitos. Momentum é como ter inércia: se você vem descendo numa direção, tende a continuar nela mesmo que o terreno local sugira um desvio pequeno, o que suaviza o zigue-zague. Adam vai além: mantém uma "velocidade média" (momentum) *e* ajusta o tamanho do passo por parâmetro.
>
> **Exemplo resolvido (código)**: loss `L(w) = (w - 3)²`, mínimo em `w = 3`. Derivada: `dL/dw = 2(w - 3)`.
>
> ```python
> def loss_grad(w):
>     return 2 * (w - 3)
>
> def sgd_step(w, grad, lr):
>     return w - lr * grad
>
> w = 0.0
> for step in range(10):
>     g = loss_grad(w)
>     w = sgd_step(w, g, lr=0.1)
>     print(f"passo {step}: w={w:.4f}, gradiente={g:.4f}")
> ```
>
> Rode e observe: o gradiente encolhe conforme `w` se aproxima de 3, e o passo encolhe junto — é assim, mecanicamente, que gradient descent converge.
>
> **Aplicação real**: o pré-treinamento de qualquer LLM moderno (mod. [09](09_treinamento_e_alinhamento.mdx)) usa AdamW com warmup + cosine decay — exatamente os itens desta lista, só que rodando em milhares de GPUs por semanas.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Batch Normalization e Layer Normalization em uma frase cada — e diga qual delas os Transformers usam.

> **Prática guiada — Acelerado**
>
> ```python
> def loss_grad(w):
>     return 2 * (w - 3)
>
> def momentum_step(w, grad_val, v, lr=0.1, beta=0.9):
>     v = beta * v + (1 - beta) * grad_val
>     # TODO: atualize w usando 'v' (a média móvel do gradiente) em vez do gradiente cru
>     w_new = ...
>     return w_new, v
>
> w, v = 0.0, 0.0
> for step in range(10):
>     g = loss_grad(w)
>     w, v = momentum_step(w, g, v)
>     print(f"passo {step}: w={w:.4f}")
> ```
>
> 1. Complete o `TODO` e rode — compare quantos passos o Momentum leva até `|w - 3| < 0.01`, versus o SGD puro do bloco de teoria (rode os dois e conte).
> 2. Confirme com PyTorch, resolvendo o mesmo problema com o otimizador pronto:
>    ```python
>    import torch
>
>    w = torch.tensor(0.0, requires_grad=True)
>    optimizer = torch.optim.SGD([w], lr=0.1)
>    for step in range(10):
>        optimizer.zero_grad()
>        loss = (w - 3) ** 2
>        loss.backward()
>        optimizer.step()
>    print(w.item())  # deve bater com o resultado da sua implementação manual
>    ```
> 3. Troque `torch.optim.SGD` por `torch.optim.Adam` no mesmo código e compare a velocidade de convergência.

### Referências
- `Paper` **Dropout: A Simple Way to Prevent NN from Overfitting** — Srivastava et al. (2014). https://jmlr.org/papers/v15/srivastava14a.html
- `Paper` **Batch Normalization** — Ioffe & Szegedy (2015). https://arxiv.org/abs/1502.03167
- `Paper` **Layer Normalization** — Ba, Kiros, Hinton (2016). https://arxiv.org/abs/1607.06450
- `Paper` **Root Mean Square Layer Normalization (RMSNorm)** — Zhang & Sennrich (2019). https://arxiv.org/abs/1910.07467

---

## 5.4 Convolutional Neural Networks (CNNs)

### Conceitos
- **Convolução discreta**: kernel, stride, padding, receptive field.
- **Pooling**: max, average, global.
- **Translation invariance** e **parameter sharing** — por que CNN funciona bem em imagens.

### Arquiteturas históricas
- **LeNet-5** (1998) — onde tudo começou.
- **AlexNet** (2012) — virada de chave do DL moderno.
- **VGG** (2014) — simplicidade.
- **GoogLeNet/Inception** (2014) — módulos.
- **ResNet** (2015) — **skip connections**, fundamentais para ir além de 100 camadas.
- **DenseNet**, **EfficientNet**.
- **Vision Transformer (ViT)** — preview do mod. [16](16_visao_computacional.md).

> **Intuição**: um kernel de convolução é um "detector de padrão" pequeno (tipo, uma bordinha diagonal) que desliza pela imagem inteira procurando aquele padrão em qualquer posição — é por isso que se chama *parameter sharing*: o mesmo detector é reusado em toda a imagem. Isso é o que dá à CNN a propriedade de *translation invariance*: um gato no canto superior esquerdo ativa o mesmo detector que um gato no centro.
>
> **Exemplo resolvido (código)**: convolução 2D manual, aplicando um kernel de detecção de borda vertical numa imagem pequena e sintética (metade escura, metade clara — uma borda no meio):
>
> ```python
> import numpy as np
>
> def conv2d(image, kernel):
>     kh, kw = kernel.shape
>     ih, iw = image.shape
>     oh, ow = ih - kh + 1, iw - kw + 1
>     output = np.zeros((oh, ow))
>     for i in range(oh):
>         for j in range(ow):
>             region = image[i:i + kh, j:j + kw]
>             output[i, j] = np.sum(region * kernel)
>     return output
>
> edge_kernel = np.array([
>     [-1, 0, 1],
>     [-1, 0, 1],
>     [-1, 0, 1],
> ])
>
> image = np.zeros((8, 8))
> image[:, 4:] = 1  # metade direita "clara", esquerda "escura" -> borda vertical no meio
>
> result = conv2d(image, edge_kernel)
> print(result)
> ```
>
> Rode e observe: a coluna onde fica a borda (no meio da imagem) tem valores bem diferentes de zero — o kernel "detectou" a transição escuro→claro. Nas colunas uniformes (totalmente escuras ou totalmente claras), o resultado é ~0.

### Por que skip connections importam para LLMs
Transformers usam residual connections derivadas das ResNets. Entender o problema que ResNet resolve é entender por que Transformers profundos treinam.

> **Intuição**: sem skip connection, o gradiente que volta da última camada até a primeira precisa passar, multiplicando, por *todos* os pesos intermediários — o mesmo efeito multiplicativo da seção 5.2. A skip connection (`output = F(x) + x`) cria um "atalho" onde o gradiente pode voltar direto pela soma, sem precisar sobreviver a todas as multiplicações.
>
> **Aplicação real**: ResNet vs uma CNN "plain" com o mesmo número de parâmetros é o experimento clássico que prova isso — a plain, a partir de certa profundidade, treina *pior* que uma versão mais rasa dela mesma (não é overfitting, é dificuldade de otimização). É exatamente o resultado do Projeto 5.3, abaixo.
>
> ```python
> import torch
> import torch.nn as nn
>
> class ResBlock(nn.Module):
>     def __init__(self, channels):
>         super().__init__()
>         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>
>     def forward(self, x):
>         out = torch.relu(self.conv1(x))
>         out = self.conv2(out)
>         return torch.relu(out + x)  # <- a skip connection: soma 'x' antes do relu final
>
> block = ResBlock(channels=16)
> x = torch.randn(1, 16, 8, 8)
> print(block(x).shape)  # torch.Size([1, 16, 8, 8]) — mesma forma da entrada
> ```
>
> **Checkpoint**: sem olhar o texto, explique por que uma CNN "plain" de 100 camadas costuma treinar pior que uma de 20 camadas — e por que isso não é overfitting.

> **Prática guiada — Acelerado**
>
> ```python
> import numpy as np
>
> def conv2d(image, kernel):
>     kh, kw = kernel.shape
>     ih, iw = image.shape
>     oh, ow = ih - kh + 1, iw - kw + 1
>     output = np.zeros((oh, ow))
>     for i in range(oh):
>         for j in range(ow):
>             region = image[i:i + kh, j:j + kw]
>             # TODO: multiplique 'region' pelo 'kernel' elemento-a-elemento e some tudo
>             output[i, j] = ...
>     return output
> ```
>
> 1. Complete o `TODO` acima e rode com o `edge_kernel` e a `image` do bloco de teoria — confirme que a saída bate com a do exemplo (borda destacada no meio).
> 2. Em PyTorch, implemente as duas versões do bloco:
>    ```python
>    import torch
>    import torch.nn as nn
>
>    class ResBlock(nn.Module):
>        def __init__(self, channels):
>            super().__init__()
>            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>
>        def forward(self, x):
>            out = torch.relu(self.conv1(x))
>            out = self.conv2(out)
>            # TODO: some a entrada original 'x' (o skip) antes do relu final
>            return torch.relu(...)
>
>    class PlainBlock(nn.Module):
>        def __init__(self, channels):
>            super().__init__()
>            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
>
>        def forward(self, x):
>            out = torch.relu(self.conv1(x))
>            out = self.conv2(out)
>            return torch.relu(out)  # sem skip, de propósito
>    ```
> 3. Complete o `TODO` do `ResBlock`, depois empilhe 20 de cada e confirme que ambos rodam sem erro de dimensão:
>    ```python
>    res_stack = nn.Sequential(*[ResBlock(16) for _ in range(20)])
>    plain_stack = nn.Sequential(*[PlainBlock(16) for _ in range(20)])
>
>    x = torch.randn(1, 16, 8, 8)
>    print(res_stack(x).shape, plain_stack(x).shape)
>    ```
> 4. Isso te deixa pronto pro Projeto 5.3 — treinar as duas versões de verdade e ver o gap.

### Referências
- `Paper` **ImageNet Classification with Deep CNN (AlexNet)** — Krizhevsky et al. (2012). https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- `Paper` **Deep Residual Learning for Image Recognition (ResNet)** — He et al. (2015). https://arxiv.org/abs/1512.03385
- `Curso` **Stanford CS231N**. http://cs231n.stanford.edu/

---

## 5.5 Redes Recorrentes (RNN, LSTM, GRU)

### Por que estudá-las (apesar de obsoletas para grandes textos)
- Foram a forma de modelar sequências antes dos Transformers.
- Permitem entender por que os Transformers ganharam (paralelização, dependências de longo alcance).
- **State Space Models** (Mamba — mod. [19](19_topicos_avancados.md)) recuperam ideias recorrentes.

### Conceitos
- **RNN vanilla**: hidden state, BPTT (backprop through time), problema de gradientes.
- **LSTM**: cell state, gates (forget, input, output).
- **GRU**: simplificação do LSTM.
- **Bidirecional RNN**.
- **Encoder-decoder com RNN** (precursor de Seq2Seq).
- **Attention sobre RNN** (Bahdanau, Luong) — antecessor direto do self-attention.

> **Intuição**: uma RNN é como ler uma frase palavra por palavra enquanto anota tudo num único caderninho (o hidden state) — cada palavra nova atualiza o caderninho, mas o que foi anotado há 50 palavras já foi bastante reescrito por cima. LSTM/GRU adicionam "portões" que decidem o que manter, o que esquecer e o que escrever no caderninho — mas o gargalo estrutural continua: informação de uma palavra distante ainda precisa sobreviver passando sequencialmente por *todas* as palavras entre ela e a atual.
>
> **Exemplo (código)**: uma célula RNN vanilla, rodada em 5 passos, mostrando a norma do estado oculto ao longo do tempo:
>
> ```python
> import torch
>
> def rnn_cell(x_t, h_prev, Wx, Wh, b):
>     return torch.tanh(Wx @ x_t + Wh @ h_prev + b)
>
> dim = 4
> torch.manual_seed(0)
> Wx = torch.randn(dim, dim) * 0.5
> Wh = torch.randn(dim, dim) * 0.5
> b = torch.zeros(dim)
>
> h = torch.zeros(dim)
> for t in range(5):
>     x_t = torch.randn(dim)
>     h = rnn_cell(x_t, h, Wx, Wh, b)
>     print(f"t={t}: norma(h)={h.norm().item():.4f}")
> ```
>
> **Aplicação real**: self-attention (mod. [07](07_transformers.mdx)) resolve o gargalo sequencial de um jeito direto — qualquer token pode "olhar" pra qualquer outro token em um único passo. Attention sobre RNN (Bahdanau, Luong) foi o primeiro passo nessa direção e é o antecessor conceitual do self-attention.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase por que um LSTM ainda tem dificuldade com dependências muito longas, mesmo tendo "portões" para controlar memória.

> **Prática guiada — Acelerado**
>
> ```python
> import torch
>
> def rnn_cell(x_t, h_prev, Wx, Wh, b):
>     # TODO: combine x_t e h_prev (via Wx e Wh) e aplique tanh
>     return torch.tanh(...)
>
> dim = 4
> torch.manual_seed(0)
> Wx = torch.randn(dim, dim) * 1.5  # pesos "grandes" de propósito
> Wh = torch.randn(dim, dim) * 1.5
> b = torch.zeros(dim)
>
> h = torch.zeros(dim)
> for t in range(5):
>     x_t = torch.randn(dim)
>     h = rnn_cell(x_t, h, Wx, Wh, b)
>     print(f"t={t}: norma(h)={h.norm().item():.4f}")
> ```
>
> 1. Complete o `TODO` e rode — com pesos "grandes" (`*1.5`), observe se a norma cresce ou satura ao longo dos passos (dica: `tanh` satura em ±1, então a explosão vai aparecer mais em redes profundas/sequências mais longas do que aqui; troque `torch.tanh` por identidade — sem ativação — pra ver a explosão de forma mais clara).
> 2. Rode a mesma sequência com `nn.LSTM` do PyTorch e compare a estabilidade:
>    ```python
>    lstm = torch.nn.LSTM(input_size=dim, hidden_size=dim, batch_first=True)
>    seq = torch.randn(1, 5, dim)
>    out, (hn, cn) = lstm(seq)
>    print(out.norm(dim=-1))
>    ```
> 3. Siga pro Projeto 5.4 pra ver esse efeito em escala real, num corpus de texto.

### Referências
- `Paper` **Long Short-Term Memory** — Hochreiter & Schmidhuber (1997). https://www.bioinf.jku.at/publications/older/2604.pdf
- `Paper` **Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)** — Bahdanau, Cho, Bengio (2014). https://arxiv.org/abs/1409.0473
- `Paper` **Sequence to Sequence Learning with Neural Networks** — Sutskever, Vinyals, Le (2014). https://arxiv.org/abs/1409.3215
- `Curso` **Karpathy — The Unreasonable Effectiveness of RNNs** (post + repo `char-rnn`). https://karpathy.github.io/2015/05/21/rnn-effectiveness/

---

## 5.6 Autoencoders e modelos generativos clássicos

- **Autoencoder vanilla**: encoder + decoder, bottleneck.
- **Denoising Autoencoder**.
- **Variational Autoencoder (VAE)**: introdução ao learning variacional.
- **Generative Adversarial Networks (GANs)**: gerador + discriminador.

> **Intuição**: um autoencoder aprende comprimindo e depois reconstruindo — o "gargalo" (bottleneck) no meio da rede é menor que a entrada, então a única forma de reconstruir bem é aprender uma representação compacta que capture o que é essencial. É o mesmo princípio de fazer um resumo de um livro: se o resumo cabe em uma página e ainda permite recontar a história, ele capturou a estrutura essencial.
>
> **Exemplo (código)**:
>
> ```python
> import torch
> import torch.nn as nn
>
> class Autoencoder(nn.Module):
>     def __init__(self, input_dim=784, latent_dim=32):
>         super().__init__()
>         self.encoder = nn.Linear(input_dim, latent_dim)
>         self.decoder = nn.Linear(latent_dim, input_dim)
>
>     def forward(self, x):
>         z = self.encoder(x)
>         reconstruction = self.decoder(z)
>         return reconstruction, z
>
> model = Autoencoder()
> x = torch.randn(1, 784)  # imagem "achatada" tipo MNIST (28x28=784)
> reconstruction, z = model(x)
> print(z.shape, reconstruction.shape)  # torch.Size([1, 32]) torch.Size([1, 784])
> ```

### Por que importa
- Embeddings em LLMs são, conceitualmente, "encoders" treinados.
- Difusão (mod. [19](19_topicos_avancados.md)) é a evolução desses paradigmas.

> **Checkpoint**: sem olhar o texto, explique por que forçar um "gargalo" (bottleneck menor que a entrada) no meio da rede é o que faz o autoencoder aprender algo útil, em vez de só copiar a entrada para a saída.

> **Prática guiada — Acelerado**
>
> ```python
> import torch
> import torch.nn as nn
>
> class Autoencoder(nn.Module):
>     def __init__(self, input_dim=784, latent_dim=32):
>         super().__init__()
>         self.encoder = nn.Linear(input_dim, latent_dim)
>         self.decoder = nn.Linear(latent_dim, input_dim)
>
>     def forward(self, x):
>         z = self.encoder(x)
>         # TODO: reconstrua a partir de 'z' usando self.decoder
>         reconstruction = ...
>         return reconstruction, z
> ```
>
> 1. Complete o `TODO` e treine no MNIST por poucas épocas (loss = `nn.MSELoss()` entre `reconstruction` e `x`).
> 2. Compare visualmente uma imagem original com sua reconstrução (`plt.imshow`).
> 3. Reduza `latent_dim` para 2, treine de novo, e plote o espaço latente (`z[:, 0]` vs `z[:, 1]`) colorido por dígito.

### Referências
- `Paper` **Auto-Encoding Variational Bayes** — Kingma & Welling (2013). https://arxiv.org/abs/1312.6114
- `Paper` **Generative Adversarial Networks** — Goodfellow et al. (2014). https://arxiv.org/abs/1406.2661

---

## 5.7 Frameworks: PyTorch a fundo

### Tópicos
- `Tensor`, `autograd`, `nn.Module`.
- `DataLoader`, `Dataset`, custom datasets.
- Training loop manual: forward → loss → backward → optimizer.step().
- **`torch.compile`** (PyTorch 2.x, JIT).
- **`torch.distributed`**: DDP, FSDP (preview do mod. [09](09_treinamento_e_alinhamento.mdx)).
- **PyTorch Lightning** (abstração sobre PyTorch para cortar boilerplate).
- **Hugging Face Accelerate** (alternativa).

> **Visão de mercado**: `autograd` é um dos temas mais comuns em entrevista técnica de ML — saber explicar que `.backward()` percorre o grafo computacional construído dinamicamente durante o forward, acumulando gradientes em `.grad`, e por que `optimizer.zero_grad()` é necessário (porque gradientes *acumulam* por padrão, não substituem) é o tipo de pergunta que separa quem só usou PyTorch de quem entende o que está fazendo.
>
> **Exemplo (código)**: o esqueleto de um training loop completo, o padrão que você vai reusar em praticamente todo projeto daqui em diante:
>
> ```python
> for epoch in range(num_epochs):
>     for x_batch, y_batch in dataloader:
>         optimizer.zero_grad()
>         preds = model(x_batch)
>         loss = loss_fn(preds, y_batch)
>         loss.backward()
>         optimizer.step()
>     print(f"epoch {epoch}: loss={loss.item():.4f}")
> ```

> **Prática guiada — Acelerado**
>
> ```python
> for epoch in range(num_epochs):
>     for x_batch, y_batch in dataloader:
>         # TODO: zere os gradientes acumulados do passo anterior
>         ...
>         preds = model(x_batch)
>         loss = loss_fn(preds, y_batch)
>         # TODO: calcule os gradientes (backward) e dê um passo do otimizador
>         ...
>     print(f"epoch {epoch}: loss={loss.item():.4f}")
> ```
>
> 1. Complete os dois `TODO`s acima com um modelo/dataset simples seus (pode ser sintético).
> 2. Comente a linha do `zero_grad()` de propósito, rode por algumas épocas, e observe a loss se comportar de forma anômala — depois descomente e confirme que volta ao normal.
> 3. Escreva um `Dataset` e `DataLoader` customizados (`torch.utils.data.Dataset`, implementando `__len__` e `__getitem__`) para seus dados.
> 4. Rode a mesma arquitetura com e sem `torch.compile(model)` e compare o tempo de uma época.

### Referências
- `Curso` **Deep Learning with PyTorch** (livro oficial gratuito). https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf
- `Ferramenta` **PyTorch Tutorials**. https://pytorch.org/tutorials/
- `Ferramenta` **PyTorch Lightning**. https://lightning.ai/docs/pytorch/stable/
- `Livro` **Dive into Deep Learning (D2L)** — todo em PyTorch. https://d2l.ai/

---

## 5.8 Visualização e debugging de redes neurais

- **TensorBoard** ou **W&B** para acompanhar loss, métricas, distribuições de pesos.
- **Saliency maps**, **Grad-CAM** (interpretação visual).
- **Hooks** em PyTorch para inspecionar ativações.
- **Verificações de sanidade**: overfitar um único batch, gradient checking numérico.

> **Intuição**: depurar uma rede neural é mais parecido com diagnóstico médico do que com debugging de software tradicional — não dá pra colocar um breakpoint na "razão pela qual o modelo não aprende". Você observa sintomas e isola causas prováveis por eliminação.
>
> **Cenário hipotético**: imagine que você está treinando e, depois de algumas centenas de passos, a loss vira `NaN`. Causas prováveis, em ordem de frequência: learning rate alto demais (gradiente explode), divisão por zero numa loss customizada, `log(0)` numa cross-entropy sem estabilização numérica, ou overflow em precisão mista (`fp16`).
>
> **Exemplo (código)**: hooks pra inspecionar a norma da ativação de cada camada, a ferramenta que você usaria pra achar *onde* uma explosão de ativação começa:
>
> ```python
> def make_hook(name):
>     def hook(module, input, output):
>         print(f"{name}: norma={output.norm().item():.4f}")
>     return hook
>
> for name, layer in model.named_modules():
>     layer.register_forward_hook(make_hook(name))
> ```
>
> **Checkpoint**: sem olhar o texto, liste 2 causas prováveis de um loss virar `NaN` durante o treino.

> **Prática guiada — Acelerado**
>
> ```python
> def make_hook(name):
>     def hook(module, input, output):
>         # TODO: imprima o nome da camada e a norma da saída (output.norm())
>         ...
>     return hook
>
> for name, layer in model.named_modules():
>     layer.register_forward_hook(make_hook(name))
> ```
>
> 1. Complete o `TODO` e registre os hooks no modelo de um projeto anterior (ex.: o MLP do Projeto 5.1 ou a CNN do 5.2).
> 2. Force um `NaN` de propósito (aumente o learning rate em 100×) e identifique, pelos prints dos hooks, em qual camada a norma explode primeiro.
> 3. Implemente a verificação "overfitar 1 batch": treine só em 8 exemplos por 200 passos e confirme que a loss cai para perto de 0 — se não cair, tem bug antes de escalar pro dataset todo.

### Referências
- `Paper` **A Recipe for Training Neural Networks** — Karpathy (post). http://karpathy.github.io/2019/04/25/recipe/

---

## Projetos práticos

### Projeto 5.1 — MLP from scratch em NumPy
- Forward, backward, SGD, tudo manual.
- Treine no MNIST. Atinja >97% accuracy.
- Compare velocidade vs PyTorch.

> **Clássico — variante guiada**: implemente em 3 checkpoints intermediários — (1) só forward pass, verifique que as dimensões batem e que a saída é uma distribuição de probabilidade válida (soma 1); (2) backward pass, verifique com gradient checking numérico antes de treinar; (3) loop de treino completo. Não avance pro próximo checkpoint sem o anterior funcionando.

> **Acelerado — checklist de implementação**
>
> Esqueleto para começar (arquitetura `784 → 128 (ReLU) → 10 (softmax)`):
>
> ```python
> import numpy as np
>
> def init_params(input_dim=784, hidden_dim=128, output_dim=10, seed=0):
>     rng = np.random.default_rng(seed)
>     W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2 / input_dim)
>     b1 = np.zeros(hidden_dim)
>     W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2 / hidden_dim)
>     b2 = np.zeros(output_dim)
>     return W1, b1, W2, b2
>
> def softmax(logits):
>     exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
>     return exp / exp.sum(axis=-1, keepdims=True)
>
> def forward(x, W1, b1, W2, b2):
>     z1 = x @ W1 + b1
>     a1 = np.maximum(0, z1)  # ReLU
>     z2 = a1 @ W2 + b2
>     # TODO: aplique softmax em z2 para obter as probabilidades finais
>     probs = ...
>     return probs, (z1, a1, z2)
> ```
>
> 1. Complete o `TODO` e confirme que `probs.sum(axis=-1)` dá `1.0` para qualquer entrada.
> 2. Implemente o backward pass camada por camada (comece pela última: gradiente de cross-entropy + softmax combinado é `probs - one_hot(y)`).
> 3. Rode um passo de treino num único batch pequeno e confirme que a loss cai.
> 4. Escale para o MNIST completo e meça a accuracy de validação.
> 5. Cronometre 1 época e compare com a mesma arquitetura implementada em PyTorch.

### Projeto 5.2 — CNN no CIFAR-10
- Em PyTorch.
- Comece com arquitetura simples, depois adicione: BatchNorm, Dropout, augmentation.
- Implemente learning rate schedule + warmup.
- Acompanhe com W&B ou TensorBoard.
- **Meta**: >85% accuracy.

> **Clássico — variante guiada**: adicione cada técnica (BatchNorm, Dropout, augmentation, schedule) uma de cada vez, registrando o ganho de accuracy isolado de cada uma antes de adicionar a próxima.

> **Acelerado — checklist de implementação**
> 1. Defina uma CNN simples (3 blocos conv+pool) e treine um baseline sem regularização.
> 2. Adicione BatchNorm + Dropout + augmentation de uma vez.
> 3. Adicione cosine schedule + warmup (`torch.optim.lr_scheduler.CosineAnnealingLR`).
> 4. Ajuste hiperparâmetros até bater a meta de 85%.

### Projeto 5.3 — Implementar ResNet pequena
- Implemente blocos residuais à mão (não use `torchvision.models.resnet`).
- Treine no CIFAR-10.
- Compare com CNN sem skip connections (com mesmo nº de parâmetros): observe o gap.

> **Clássico — variante guiada**: antes de treinar a versão funda, treine as duas versões (com e sem skip connection) numa profundidade rasa onde ambas devem treinar bem — confirme que o gap só aparece quando você aumenta a profundidade. Isso isola a variável certa.

> **Acelerado — checklist de implementação**
> 1. Reuse o `ResBlock`/`PlainBlock` implementados na seção 5.4.
> 2. Empilhe blocos até uma profundidade alta (20+ blocos), com uma camada final `nn.Linear` para classificação.
> 3. Treine as duas versões no CIFAR-10 com o mesmo orçamento de épocas.
> 4. Plote as duas curvas de loss de treino no mesmo gráfico.
> 5. Anote a diferença observada em 1 linha.

### Projeto 5.4 — char-RNN (gerador de texto caractere a caractere)
- Implemente LSTM em PyTorch.
- Treine em corpus de texto à sua escolha (Shakespeare, código, letras de música).
- Gere texto novo. Compare com versão GPT mínima (preview de mod. [07](07_transformers.mdx)).

> **Clássico — variante guiada**: treine e valide o LSTM sozinho, com checkpoints de geração intermediários (a cada X épocas, gere uma amostra e leia), antes de comparar com a versão GPT mínima.

> **Acelerado — checklist de implementação**
> 1. Prepare o corpus: tokenização por caractere, vocabulário (`set` dos caracteres únicos), encoding para índices inteiros.
> 2. Implemente `nn.LSTM(input_size=vocab_size, hidden_size=256, batch_first=True)` + `nn.Linear(256, vocab_size)` de saída.
> 3. Treine com teacher forcing (a sequência de entrada é o texto, a de saída é o texto deslocado 1 caractere).
> 4. Implemente geração autoregressiva: a cada passo, amostre o próximo caractere da distribuição de saída e realimente o modelo.
> 5. Repita com o GPT mínimo do mod. [07](07_transformers.mdx) no mesmo corpus e compare velocidade de treino e qualidade do texto gerado.

### Projeto 5.5 — VAE no MNIST
- Implemente encoder, decoder, reparameterization trick.
- Visualize espaço latente em 2D.
- Gere amostras interpolando no espaço latente.

> **Clássico — variante guiada**: antes de implementar, derive por que o reparameterization trick é necessário (gradiente não passa por uma amostragem estocástica direta) — só depois implemente.

> **Acelerado — checklist de implementação**
>
> ```python
> import torch
>
> def reparameterize(mu, log_var):
>     std = torch.exp(0.5 * log_var)
>     eps = torch.randn_like(std)
>     # TODO: combine mu, std e eps para amostrar z (a fórmula do reparameterization trick)
>     z = ...
>     return z
> ```
>
> 1. Implemente o encoder (saída: `mu` e `log_var`, dois vetores) e o decoder.
> 2. Complete o `TODO` acima.
> 3. Treine com loss = reconstrução (MSE ou BCE) + KL divergence (`-0.5 * sum(1 + log_var - mu**2 - log_var.exp())`).
> 4. Reduza a dimensão latente para 2 e plote o espaço latente.
> 5. Gere amostras interpolando linearmente entre dois pontos do espaço latente.

### Projeto 5.6 (cross-link com mod. [14](14_avaliacao_e_seguranca.md)) — Pipeline de experimentação completo
- Use W&B para log.
- Faça grid de experimentos: 3 arquiteturas × 3 learning rates × 2 batch sizes.
- Analise a tabela de resultados.

> **Clássico — variante guiada**: rode o grid completo (18 combinações) e documente, para cada resultado, uma hipótese de por que ele saiu melhor ou pior antes de olhar a tabela final.

> **Acelerado — checklist de implementação**
> 1. Escreva um script que recebe arquitetura/learning rate/batch size como argumentos de linha de comando (`argparse`).
> 2. Rode o grid completo via loop (ou `wandb sweep`, se já usa W&B).
> 3. Puxe os resultados numa tabela via API do W&B.
> 4. Identifique a melhor combinação e justifique em 1 linha.

---

## Erros comuns

- **Não fazer o "overfitar 1 batch"** — primeira sanity check de qualquer treinamento.
- **Misturar `model.train()` e `model.eval()`** — BatchNorm e Dropout dependem disso.
- **Esquecer `optimizer.zero_grad()`** — gradientes acumulam.
- **Comparar runs com seeds diferentes** sem rodar várias seeds.
- **Acreditar que "mais profundo é sempre melhor"** — sem skip connections, não é.
- **Treinar sem learning rate scheduler** em qualquer treinamento longo.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Backpropagation | Treinamento de Transformers |
| Skip connections | Transformers (mod. [07](07_transformers.mdx)) |
| Layer Normalization | Transformers (mod. [07](07_transformers.mdx)) |
| Adam, warmup | Treinamento de LLMs (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| RNN encoder-decoder | Conceito que Transformers substituem |
| Attention (Bahdanau) | Self-attention (mod. [07](07_transformers.mdx)) |
| Embeddings em autoencoders | Word embeddings (mod. [06](06_nlp_classico.md)) |

---

## Checklist de saída

- [ ] Implementei MLP e backprop em NumPy puro.
- [ ] Treinei pelo menos uma CNN com >85% no CIFAR-10.
- [ ] Implementei e treinei uma RNN/LSTM em uma tarefa de sequência.
- [ ] Sei diagnosticar overfitting, vanishing gradients, loss NaN.
- [ ] Domino o ciclo: dataset → DataLoader → modelo → loss → optimizer → log → checkpoint.
- [ ] Sei explicar por que skip connections permitem redes muito profundas.
