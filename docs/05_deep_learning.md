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

> **Tempo por caminho**: **Clássico** ~8–10 semanas (teoria completa + prática, nessa ordem) · **Acelerado** ~2–3 semanas (só prática guiada, sem derivação teórica).

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

Poucos "sim" → siga o caminho **Clássico** (teoria completa e depois prática). Maioria "sim" → vá direto pelos blocos **Prática guiada — Acelerado**, que assumem que você já manja da teoria e só quer construir.

Este módulo tem dois trilhos em paralelo: um de teoria (intuição, exemplo resolvido, aplicação real, checkpoint) pra quem está construindo a base, e um de prática guiada — só passos de implementação, sem prosa — pra quem já tem bagagem e quer ir direto pro código.

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
> Se a segunda entrada fosse `x = [1, 5]` com os mesmos pesos: `z = 1 - 2.5 = -1.5` → `ReLU(-1.5) = 0` → o neurônio não ativa. Backprop, mais adiante, é só a regra da cadeia aplicada repetidamente pra descobrir *quanto cada peso contribuiu pro erro final* — nada além disso.
>
> **Aplicação real**: toda camada de todo Transformer (inclusive os LLMs que você vai estudar a partir do mod. [07](07_transformers.mdx)) é, no fundo, essa mesma combinação linear + não-linearidade, repetida em escala. Entender isso aqui, pequeno, é entender o átomo de tudo que vem depois.
>
> **Checkpoint**: sem olhar o texto acima, explique em duas frases por que uma rede *sem* função de ativação não-linear, não importa quantas camadas tenha, equivale a uma única camada linear.

> **Prática guiada — Acelerado**
> 1. Crie `x = [2, 3]`, `w = [1, -0.5]`, `b = 0` em Python puro (sem NumPy).
> 2. Calcule `z = x[0]*w[0] + x[1]*w[1] + b` na mão, em código.
> 3. Implemente `relu = lambda z: max(0, z)` e aplique sobre `z`.
> 4. Rode com `x = [1, 5]` e confirme que a saída é `0`.
> 5. Generalize: escreva `neuron(x, w, b, activation)` aceitando entrada de tamanho arbitrário (use um loop ou `sum(xi*wi for xi, wi in zip(x, w))`).
> 6. Implemente `sigmoid`, `tanh` e `relu` como três funções separadas e plote as três num mesmo gráfico (`matplotlib`) para `z` de -5 a 5.

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
> **Exemplo (ilustrativo)**: imagine uma rede de 50 camadas onde cada camada, por causa da inicialização, multiplica a variância do sinal por ~1.5 ao passar adiante. Depois de 50 camadas, isso é `1.5^50` — um número absurdamente grande, ativações explodindo. Se em vez disso cada camada multiplicasse por ~0.7, seria `0.7^50` — praticamente zero, sinal morto antes de chegar à saída. É exatamente esse efeito multiplicativo, camada após camada, que Xavier/He init tentam neutralizar, escolhendo a escala inicial dos pesos para manter a variância ~estável ao longo da rede.
>
> **Aplicação real**: má inicialização é uma das causas mais comuns de "meu modelo não aprende nada e eu não sei por quê" — antes de suspeitar de bug complexo, verifique a inicialização.
>
> **Checkpoint**: sem consultar o texto, explique por que "a rede pode teoricamente aproximar qualquer função" (Universal Approximation) não é a mesma coisa que "a rede vai aprender essa função na prática".

> **Prática guiada — Acelerado**
> 1. Implemente uma pilha de 30 camadas lineares (`y = W @ x`, sem ativação) em NumPy.
> 2. Inicialize todos os pesos com `std=2.0` e passe um vetor de entrada aleatório; imprima a norma da ativação a cada camada.
> 3. Repita com `std=0.01` e observe o padrão oposto.
> 4. Repita usando He init (`std = sqrt(2/fan_in)`) e confirme que a norma se mantém estável ao longo das 30 camadas.
> 5. Anote em 1 linha o que mudou entre as três rodadas.

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

> **Intuição**: pense em minimizar a loss como descer uma montanha no escuro, só sentindo a inclinação sob os pés. SGD puro dá um passo na direção da descida mais íngreme e reavalia — pode ficar "zigue-zagueando" em vales estreitos. Momentum é como ter inércia: se você vem descendo numa direção, tende a continuar nela mesmo que o terreno local sugira um desvio pequeno, o que suaviza o zigue-zague. Adam vai além: mantém uma "velocidade média" (momentum) *e* ajusta o tamanho do passo por parâmetro, dando passos maiores em direções onde o gradiente tem sido consistentemente pequeno e menores onde ele oscila muito.
>
> **Exemplo resolvido**: loss `L(w) = (w - 3)²`, mínimo em `w = 3`. Derivada: `dL/dw = 2(w - 3)`.
> Começando em `w = 0`, gradiente = `2×(0-3) = -6`. Com learning rate `0.1`: `w_novo = 0 - 0.1×(-6) = 0.6`.
> Próximo passo: gradiente em `w=0.6` é `2×(0.6-3) = -4.8`, `w_novo = 0.6 - 0.1×(-4.8) = 1.08`.
> Note que o passo fica menor conforme `w` se aproxima de 3 (gradiente encolhe) — é assim, mecanicamente, que gradient descent converge. Momentum acumularia parte desses `-6, -4.8, ...` numa média móvel, acelerando a convergência ao longo dessa mesma direção consistente.
>
> **Aplicação real**: o pré-treinamento de qualquer LLM moderno (mod. [09](09_treinamento_e_alinhamento.mdx)) usa AdamW com warmup + cosine decay — exatamente os itens desta lista, só que rodando em milhares de GPUs por semanas. Se você entende por que warmup existe aqui (evitar um passo grande demais antes do otimizador "aquecer" suas estimativas), você entende por que ele é não-negociável em treino de LLM.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Batch Normalization e Layer Normalization em uma frase cada — e diga qual delas os Transformers usam.

> **Prática guiada — Acelerado**
> 1. Implemente `sgd_step(w, grad, lr)` retornando `w - lr*grad`.
> 2. Rode 10 iterações em `L(w) = (w-3)**2` a partir de `w=0`, `lr=0.1`, imprimindo `w` a cada passo.
> 3. Implemente `momentum_step(w, grad, v, lr, beta)` mantendo uma variável `v` (velocidade) e repita o mesmo experimento.
> 4. Compare quantos passos cada versão leva até `|w - 3| < 0.01`.
> 5. Refaça o mesmo problema com `torch.tensor(0.0, requires_grad=True)` e `torch.optim.SGD`/`torch.optim.Adam`, e confirme que os valores batem com sua implementação manual.

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

> **Intuição**: um kernel de convolução é um "detector de padrão" pequeno (tipo, uma bordinha diagonal) que desliza pela imagem inteira procurando aquele padrão em qualquer posição — é por isso que se chama *parameter sharing*: o mesmo detector é reusado em toda a imagem, em vez de aprender um detector diferente para cada posição de pixel. Isso é o que dá à CNN a propriedade de *translation invariance*: um gato no canto superior esquerdo ativa o mesmo detector que um gato no centro.

### Por que skip connections importam para LLMs
Transformers usam residual connections derivadas das ResNets. Entender o problema que ResNet resolve é entender por que Transformers profundos treinam.

> **Intuição**: sem skip connection, o gradiente que volta da última camada até a primeira precisa passar, multiplicando, por *todos* os pesos intermediários — exatamente o mesmo efeito multiplicativo do exemplo de inicialização da seção 5.2, só que agora acontecendo durante o treino inteiro, não só no início. Com poucas dezenas de camadas isso já é suficiente pra o gradiente vanishing antes de chegar às primeiras camadas. A skip connection (`output = F(x) + x`) cria um "atalho" onde o gradiente pode voltar direto pela soma, sem precisar sobreviver a todas as multiplicações — é literalmente uma rota alternativa que ignora o gargalo.
>
> **Aplicação real**: ResNet vs uma CNN "plain" com o mesmo número de parâmetros é o experimento clássico que prova isso — a plain, a partir de certa profundidade, treina *pior* que uma versão mais rasa dela mesma (não é overfitting, é dificuldade de otimização). Esse é exatamente o resultado que você vai reproduzir no Projeto 5.3, abaixo.
>
> **Checkpoint**: sem olhar o texto, explique por que uma CNN "plain" de 100 camadas costuma treinar pior que uma de 20 camadas — e por que isso não é overfitting.

> **Prática guiada — Acelerado**
> 1. Implemente uma convolução 2D em NumPy (sem `nn.Conv2d`) aplicando um kernel 3×3 de detecção de borda numa imagem pequena (8×8).
> 2. Confirme numericamente que os valores nas bordas ficam destacados.
> 3. Em PyTorch, implemente `ResBlock`: `forward(x) = relu(conv2(relu(conv1(x))) + x)`.
> 4. Implemente a versão "plain" do mesmo bloco, removendo só a soma do skip.
> 5. Empilhe 20 blocos de cada versão e confirme que ambos rodam sem erro de dimensão antes de ir pro Projeto 5.3.

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

> **Intuição**: uma RNN é como ler uma frase palavra por palavra enquanto anota tudo num único caderninho (o hidden state) — cada palavra nova atualiza o caderninho, mas o que foi anotado há 50 palavras já foi bastante reescrito por cima. LSTM/GRU adicionam "portões" que decidem o que manter, o que esquecer e o que escrever no caderninho, dando mais controle sobre o que sobrevive por muitas palavras — mas o gargalo estrutural continua: informação de uma palavra distante ainda precisa sobreviver passando sequencialmente por *todas* as palavras entre ela e a atual.
>
> **Exemplo (ilustrativo)**: em BPTT, o gradiente numa RNN vanilla é multiplicado repetidamente pela mesma matriz de pesos recorrente a cada passo de tempo — o mesmo mecanismo multiplicativo das seções 5.2 e 5.4, agora ao longo do *tempo* em vez das *camadas*. Numa sequência de 100 tokens, isso significa ~100 multiplicações em cadeia: se cada uma encolhe o gradiente um pouco, depois de 100 passos ele já pode ter praticamente zerado. É por isso que RNN vanilla "esquece" contexto distante.
>
> **Aplicação real**: self-attention (mod. [07](07_transformers.mdx)) resolve esse gargalo de um jeito direto: qualquer token pode "olhar" pra qualquer outro token em um único passo, sem depender de uma cadeia sequencial de estados intermediários. Attention sobre RNN (Bahdanau, Luong) foi o primeiro passo nessa direção — deixava o decoder "espiar" diretamente os estados do encoder em vez de confiar só no último hidden state — e é literalmente o antecessor conceitual do self-attention.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase por que um LSTM ainda tem dificuldade com dependências muito longas, mesmo tendo "portões" para controlar memória.

> **Prática guiada — Acelerado**
> 1. Implemente uma célula RNN vanilla manualmente em PyTorch: `h_t = tanh(Wx @ x_t + Wh @ h_prev + b)`.
> 2. Rode numa sequência de 5 passos com valores aleatórios e imprima a norma de `h_t` a cada passo.
> 3. Troque `tanh` por identidade e use pesos com norma > 1 — observe a explosão da norma.
> 4. Rode a mesma sequência com `nn.LSTM` do PyTorch e compare a estabilidade da norma do estado ao longo dos passos.
> 5. Siga pro Projeto 5.4 pra ver esse efeito em escala real, num corpus de texto.

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

> **Intuição**: um autoencoder aprende comprimindo e depois reconstruindo — o "gargalo" (bottleneck) no meio da rede é menor que a entrada, então a única forma de reconstruir bem é aprender uma representação compacta que capture o que é essencial e descarte o que é ruído/redundância. É o mesmo princípio de fazer um resumo de um livro: se o resumo cabe em uma página e ainda permite recontar a história, ele capturou a estrutura essencial.

### Por que importa
- Embeddings em LLMs são, conceitualmente, "encoders" treinados.
- Difusão (mod. [19](19_topicos_avancados.md)) é a evolução desses paradigmas.

> **Checkpoint**: sem olhar o texto, explique por que forçar um "gargalo" (bottleneck menor que a entrada) no meio da rede é o que faz o autoencoder aprender algo útil, em vez de só copiar a entrada para a saída.

> **Prática guiada — Acelerado**
> 1. Implemente um autoencoder simples em PyTorch: `Linear(784→32) → Linear(32→784)`.
> 2. Treine no MNIST por poucas épocas (loss = MSE de reconstrução).
> 3. Compare visualmente uma imagem original com sua reconstrução.
> 4. Reduza o bottleneck para 2 dimensões e plote o espaço latente colorido por dígito.

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

> **Visão de mercado**: `autograd` é um dos temas mais comuns em entrevista técnica de ML — saber explicar que `.backward()` percorre o grafo computacional construído dinamicamente durante o forward, acumulando gradientes em `.grad`, e por que `optimizer.zero_grad()` é necessário (porque gradientes *acumulam* por padrão, não substituem) é o tipo de pergunta que separa quem só usou PyTorch de quem entende o que está fazendo. DDP replica o modelo inteiro em cada GPU e sincroniza gradientes; FSDP (preview do mod. [09](09_treinamento_e_alinhamento.mdx)) particiona o próprio modelo entre GPUs — a diferença entre os dois é exatamente o tipo de trade-off que aparece em decisões reais de infraestrutura de treino.

> **Prática guiada — Acelerado**
> 1. Escreva um training loop mínimo sem Lightning/Accelerate: forward → loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`.
> 2. Rode o mesmo loop esquecendo `zero_grad()` de propósito e observe a loss se comportar de forma anômala — depois corrija.
> 3. Escreva um `Dataset` e `DataLoader` customizados para um conjunto de dados seu (pode ser sintético).
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

> **Intuição**: depurar uma rede neural é mais parecido com diagnóstico médico do que com debugging de software tradicional — não dá pra colocar um breakpoint na "razão pela qual o modelo não aprende". Em vez disso, você observa sintomas (loss não desce, loss vira NaN, accuracy de treino boa mas validação ruim) e isola causas prováveis por eliminação.
>
> **Cenário hipotético**: imagine que você está treinando e, depois de algumas centenas de passos, a loss vira `NaN`. Causas prováveis, em ordem de frequência: learning rate alto demais (gradiente explode), divisão por zero em alguma loss customizada, `log(0)` numa cross-entropy sem estabilização numérica, ou overflow em precisão mista (`fp16`) sem `loss scaling`. O primeiro passo de diagnóstico costuma ser reduzir o learning rate em 10× e ver se o problema some — se sim, era isso.
>
> **Checkpoint**: sem olhar o texto, liste 2 causas prováveis de um loss virar `NaN` durante o treino.

> **Prática guiada — Acelerado**
> 1. Pegue o training loop da seção 5.7 e adicione um `register_forward_hook` em cada camada que imprime a norma da ativação.
> 2. Force um `NaN` de propósito (aumente o learning rate em 100×) e identifique em qual camada a norma explode primeiro.
> 3. Implemente a verificação "overfitar 1 batch": treine só em 8 exemplos por 200 passos e confirme que a loss cai para perto de 0.
> 4. Implemente gradient checking numérico para uma única camada linear pequena (compare gradiente analítico vs diferença finita).

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
> 1. Defina a arquitetura: `784 → 128 (ReLU) → 10 (softmax)`.
> 2. Inicialize os pesos com He init.
> 3. Implemente o forward pass retornando os logits.
> 4. Implemente cross-entropy + softmax combinados (estável numericamente, sem `log(0)`).
> 5. Implemente o backward pass camada por camada, começando pela última.
> 6. Rode um passo de treino num único batch pequeno e confirme que a loss cai.
> 7. Escale para o MNIST completo e meça a accuracy de validação.
> 8. Cronometre 1 época e compare com a mesma arquitetura implementada em PyTorch.

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
> 3. Adicione cosine schedule + warmup.
> 4. Ajuste hiperparâmetros até bater a meta de 85%.

### Projeto 5.3 — Implementar ResNet pequena
- Implemente blocos residuais à mão (não use `torchvision.models.resnet`).
- Treine no CIFAR-10.
- Compare com CNN sem skip connections (com mesmo nº de parâmetros): observe o gap.

> **Clássico — variante guiada**: antes de treinar a versão funda, treine as duas versões (com e sem skip connection) numa profundidade rasa onde ambas devem treinar bem — confirme que o gap só aparece quando você aumenta a profundidade. Isso isola a variável certa.

> **Acelerado — checklist de implementação**
> 1. Implemente `ResBlock` (2 convs + skip).
> 2. Empilhe blocos até uma profundidade alta (20+ blocos).
> 3. Implemente a versão "plain" removendo só a soma do skip (mesmo nº de parâmetros).
> 4. Treine as duas no CIFAR-10 com o mesmo orçamento de épocas.
> 5. Plote as duas curvas de loss de treino no mesmo gráfico.
> 6. Anote a diferença observada em 1 linha.

### Projeto 5.4 — char-RNN (gerador de texto caractere a caractere)
- Implemente LSTM em PyTorch.
- Treine em corpus de texto à sua escolha (Shakespeare, código, letras de música).
- Gere texto novo. Compare com versão GPT mínima (preview de mod. [07](07_transformers.mdx)).

> **Clássico — variante guiada**: treine e valide o LSTM sozinho, com checkpoints de geração intermediários (a cada X épocas, gere uma amostra e leia), antes de comparar com a versão GPT mínima.

> **Acelerado — checklist de implementação**
> 1. Prepare o corpus: tokenização por caractere, vocabulário, encoding (one-hot ou embedding).
> 2. Implemente `nn.LSTM` + camada linear de saída.
> 3. Treine com teacher forcing.
> 4. Implemente geração autoregressiva (sample token a token).
> 5. Repita com o GPT mínimo do mod. [07](07_transformers.mdx) no mesmo corpus e compare velocidade de treino e qualidade do texto gerado.

### Projeto 5.5 — VAE no MNIST
- Implemente encoder, decoder, reparameterization trick.
- Visualize espaço latente em 2D.
- Gere amostras interpolando no espaço latente.

> **Clássico — variante guiada**: antes de implementar, derive por que o reparameterization trick é necessário (gradiente não passa por uma amostragem estocástica direta) — só depois implemente.

> **Acelerado — checklist de implementação**
> 1. Implemente o encoder (saída: média e log-variância) e o decoder.
> 2. Implemente o reparameterization trick: `z = mu + std * eps`, com `eps` amostrado de uma normal padrão.
> 3. Treine com loss = reconstrução + KL divergence.
> 4. Reduza a dimensão latente para 2 e plote o espaço latente.
> 5. Gere amostras interpolando linearmente entre dois pontos do espaço latente.

### Projeto 5.6 (cross-link com mod. [14](14_avaliacao_e_seguranca.md)) — Pipeline de experimentação completo
- Use W&B para log.
- Faça grid de experimentos: 3 arquiteturas × 3 learning rates × 2 batch sizes.
- Analise a tabela de resultados.

> **Clássico — variante guiada**: rode o grid completo (18 combinações) e documente, para cada resultado, uma hipótese de por que ele saiu melhor ou pior antes de olhar a tabela final.

> **Acelerado — checklist de implementação**
> 1. Escreva um script que recebe arquitetura/learning rate/batch size como argumentos.
> 2. Rode o grid completo via loop (ou `wandb sweep`).
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
