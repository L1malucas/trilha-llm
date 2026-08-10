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

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar por que um neurônio precisa de não-linearidade, e derivar isso a partir da definição de combinação linear.
- Explicar por que uma rede mal inicializada não treina (vanishing/exploding), não só citar o termo.
- Justificar cada escolha de regularização (dropout, batchnorm, augmentation) numa CNN, não só aplicá-las por hábito.
- Explicar, com uma frase, por que skip connections permitem redes muito mais profundas — e por que isso importa para Transformers.
- Diagnosticar (e não só nomear) overfitting, gradientes explodindo e loss virando NaN durante um treino real.
- Navegar o ciclo completo dataset → DataLoader → modelo → loss → optimizer → log → checkpoint sem copiar um template.

Este módulo constrói cada ideia nova a partir de uma analogia ou de um exemplo numérico pequeno, antes de qualquer notação formal — e fecha cada seção com um checkpoint pra você verificar se realmente entendeu, não só leu.

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

> **Intuição**: um neurônio artificial é um "juiz" que pesa evidências e decide se "ativa" ou não. Cada entrada é uma evidência, cada peso é o quanto aquela evidência importa pra esse juiz específico, e o bias é o quão fácil ou difícil é convencê-lo. A não-linearidade (ReLU, sigmoid...) é o que faz o juiz efetivamente *decidir* algo em vez de só somar números — sem ela, empilhar camadas equivaleria a uma única camada linear, por mais profunda que a rede pareça: a composição de duas transformações lineares ainda é uma transformação linear, então sem não-linearidade no meio, 100 camadas colapsam matematicamente em 1.
>
> **Exemplo resolvido**: entradas `x = [2, 3]`, pesos `w = [1, -0.5]`, bias `b = 0`.
> `z = (2×1) + (3×-0.5) + 0 = 2 - 1.5 = 0.5`
> Aplicando ReLU: `ReLU(0.5) = max(0, 0.5) = 0.5` → o neurônio "ativa" com força 0.5.
> Se a segunda entrada fosse `x = [1, 5]` com os mesmos pesos: `z = 1 - 2.5 = -1.5` → `ReLU(-1.5) = 0` → o neurônio não ativa. Backprop, mais adiante, é só a regra da cadeia aplicada repetidamente pra descobrir *quanto cada peso contribuiu pro erro final* — nada além disso. A escolha da função de ativação também molda o comportamento do gradiente: sigmoid e tanh "saturam" (derivada perto de zero) para `|z|` grande, o que é a raiz do vanishing gradient discutido na seção 5.2; ReLU tem derivada constante (1) para `z > 0`, por isso é o padrão em redes profundas.
>
> **Aplicação real**: toda camada de todo Transformer (inclusive os LLMs que você vai estudar a partir do mod. [07](07_transformers.mdx)) é, no fundo, essa mesma combinação linear + não-linearidade, repetida em escala. Entender isso aqui, pequeno, é entender o átomo de tudo que vem depois.
>
> **Checkpoint**: sem olhar o texto acima, explique em duas frases por que uma rede *sem* função de ativação não-linear, não importa quantas camadas tenha, equivale a uma única camada linear. Depois, explique por que sigmoid tende a ter gradientes menores que ReLU.

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
> **Exemplo (ilustrativo)**: imagine uma rede de 50 camadas onde cada camada, por causa da inicialização, multiplica a variância do sinal por ~1.5 ao passar adiante. Depois de 50 camadas, isso é `1.5^50` — um número absurdamente grande, ativações explodindo. Se em vez disso cada camada multiplicasse por ~0.7, seria `0.7^50` — praticamente zero, sinal morto antes de chegar à saída. É exatamente esse efeito multiplicativo, camada após camada, que Xavier/He init tentam neutralizar, escolhendo a escala inicial dos pesos (proporcional a `1/√fan_in` no caso de Xavier, `√(2/fan_in)` no caso de He) para manter a variância do sinal ~estável ao longo da rede, em vez de deixá-la ao acaso.
>
> **Aplicação real**: má inicialização é uma das causas mais comuns de "meu modelo não aprende nada e eu não sei por quê" — antes de suspeitar de bug complexo, verifique a inicialização. É também por isso que frameworks modernos (PyTorch, JAX) já usam He/Kaiming init por padrão em camadas com ReLU — a comunidade aprendeu a lição.
>
> **Checkpoint**: sem consultar o texto, explique por que "a rede pode teoricamente aproximar qualquer função" (Universal Approximation) não é a mesma coisa que "a rede vai aprender essa função na prática". Depois, explique por que He init usa uma escala diferente de Xavier init (dica: tem a ver com o que ReLU faz com metade dos valores).

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
> Note que o passo fica menor conforme `w` se aproxima de 3 (gradiente encolhe) — é assim, mecanicamente, que gradient descent converge. Momentum acumularia parte desses `-6, -4.8, ...` numa média móvel, acelerando a convergência ao longo dessa mesma direção consistente. Regularização (dropout, weight decay, augmentation) ataca um problema diferente: não é sobre convergir mais rápido, é sobre convergir para uma solução que generalize — dropout força a rede a não depender de nenhum neurônio específico (desligando aleatoriamente uma fração deles a cada passo), weight decay penaliza pesos grandes (soluções "mais simples" tendem a generalizar melhor), augmentation multiplica artificialmente a variedade de exemplos vistos.
>
> **Aplicação real**: o pré-treinamento de qualquer LLM moderno (mod. [09](09_treinamento_e_alinhamento.mdx)) usa AdamW com warmup + cosine decay — exatamente os itens desta lista, só que rodando em milhares de GPUs por semanas. Se você entende por que warmup existe aqui (evitar um passo grande demais antes do otimizador "aquecer" suas estimativas de momento), você entende por que ele é não-negociável em treino de LLM.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Batch Normalization e Layer Normalization em uma frase cada — e diga qual delas os Transformers usam. Depois, explique em uma frase por que dropout e weight decay resolvem problemas diferentes (velocidade de convergência vs. generalização) mesmo sendo os dois chamados de "regularização".

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

> **Intuição**: um kernel de convolução é um "detector de padrão" pequeno (tipo, uma bordinha diagonal) que desliza pela imagem inteira procurando aquele padrão em qualquer posição — é por isso que se chama *parameter sharing*: o mesmo detector é reusado em toda a imagem, em vez de aprender um detector diferente para cada posição de pixel. Isso é o que dá à CNN a propriedade de *translation invariance*: um gato no canto superior esquerdo ativa o mesmo detector que um gato no centro. `stride` controla de quanto em quanto o detector "pula" pela imagem (stride maior = saída menor, menos overlap); `padding` controla o que acontece nas bordas (sem padding, a imagem encolhe a cada convolução); `receptive field` é quanto da imagem original uma unidade da saída "enxerga" — cresce a cada camada empilhada, o que é parte de por que empilhar convoluções ajuda a capturar padrões cada vez mais globais.

### Por que skip connections importam para LLMs
Transformers usam residual connections derivadas das ResNets. Entender o problema que ResNet resolve é entender por que Transformers profundos treinam.

> **Intuição**: sem skip connection, o gradiente que volta da última camada até a primeira precisa passar, multiplicando, por *todos* os pesos intermediários — exatamente o mesmo efeito multiplicativo do exemplo de inicialização da seção 5.2, só que agora acontecendo durante o treino inteiro, não só no início. Com poucas dezenas de camadas isso já é suficiente pra o gradiente vanishing antes de chegar às primeiras camadas. A skip connection (`output = F(x) + x`) cria um "atalho" onde o gradiente pode voltar direto pela soma, sem precisar sobreviver a todas as multiplicações — é literalmente uma rota alternativa que ignora o gargalo.
>
> **Aplicação real**: ResNet vs uma CNN "plain" com o mesmo número de parâmetros é o experimento clássico que prova isso — a plain, a partir de certa profundidade, treina *pior* que uma versão mais rasa dela mesma (não é overfitting, é dificuldade de otimização). Esse é exatamente o resultado que você vai reproduzir no Projeto 5.3, abaixo. Skip connections viraram tão fundamentais que hoje são parte do bloco básico de praticamente toda arquitetura profunda moderna — inclusive o bloco Transformer inteiro (mod. [07](07_transformers.mdx)) tem duas skip connections por camada, uma ao redor da attention e outra ao redor do feed-forward.
>
> **Checkpoint**: sem olhar o texto, explique por que uma CNN "plain" de 100 camadas costuma treinar pior que uma de 20 camadas — e por que isso não é overfitting.

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

> **Intuição**: uma RNN é como ler uma frase palavra por palavra enquanto anota tudo num único caderninho (o hidden state) — cada palavra nova atualiza o caderninho, mas o que foi anotado há 50 palavras já foi bastante reescrito por cima. LSTM/GRU adicionam "portões" (gates) que decidem o que manter, o que esquecer e o que escrever no caderninho, dando mais controle sobre o que sobrevive por muitas palavras — mas o gargalo estrutural continua: informação de uma palavra distante ainda precisa sobreviver passando sequencialmente por *todas* as palavras entre ela e a atual. O "forget gate" do LSTM, por exemplo, aprende quando é seguro "esquecer" parte do cell state; sem ele (RNN vanilla), tudo se mistura sempre.
>
> **Exemplo (ilustrativo)**: em BPTT, o gradiente numa RNN vanilla é multiplicado repetidamente pela mesma matriz de pesos recorrente a cada passo de tempo — o mesmo mecanismo multiplicativo das seções 5.2 e 5.4, agora ao longo do *tempo* em vez das *camadas*. Numa sequência de 100 tokens, isso significa ~100 multiplicações em cadeia: se cada uma encolhe o gradiente um pouco, depois de 100 passos ele já pode ter praticamente zerado. É por isso que RNN vanilla "esquece" contexto distante — e por que mesmo LSTM, que mitiga bastante o problema, ainda degrada em sequências muito longas (centenas ou milhares de tokens).
>
> **Aplicação real**: self-attention (mod. [07](07_transformers.mdx)) resolve esse gargalo de um jeito direto: qualquer token pode "olhar" pra qualquer outro token em um único passo, sem depender de uma cadeia sequencial de estados intermediários. Attention sobre RNN (Bahdanau, Luong) foi o primeiro passo nessa direção — deixava o decoder "espiar" diretamente os estados do encoder em vez de confiar só no último hidden state — e é literalmente o antecessor conceitual do self-attention.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase por que um LSTM ainda tem dificuldade com dependências muito longas, mesmo tendo "portões" para controlar memória.

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

> **Intuição**: um autoencoder aprende comprimindo e depois reconstruindo — o "gargalo" (bottleneck) no meio da rede é menor que a entrada, então a única forma de reconstruir bem é aprender uma representação compacta que capture o que é essencial e descarte o que é ruído/redundância. É o mesmo princípio de fazer um resumo de um livro: se o resumo cabe em uma página e ainda permite recontar a história, ele capturou a estrutura essencial. Um VAE vai além do autoencoder vanilla ao forçar essa representação compacta a seguir uma distribuição conhecida (normalmente gaussiana) — é isso que permite *gerar* amostras novas depois: basta amostrar um ponto aleatório dessa distribuição e passar pelo decoder. Uma GAN ataca o mesmo problema de geração de outro jeito: um gerador tenta produzir amostras convincentes, um discriminador tenta distinguir real de gerado, e os dois treinam um contra o outro até o gerador ficar bom o suficiente para enganar o discriminador.

### Por que importa
- Embeddings em LLMs são, conceitualmente, "encoders" treinados.
- Difusão (mod. [19](19_topicos_avancados.md)) é a evolução desses paradigmas.

> **Checkpoint**: sem olhar o texto, explique por que forçar um "gargalo" (bottleneck menor que a entrada) no meio da rede é o que faz o autoencoder aprender algo útil, em vez de só copiar a entrada para a saída.

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

> **Visão de mercado**: `autograd` é um dos temas mais comuns em entrevista técnica de ML — saber explicar que `.backward()` percorre o grafo computacional construído dinamicamente durante o forward, acumulando gradientes em `.grad`, e por que `optimizer.zero_grad()` é necessário (porque gradientes *acumulam* por padrão, não substituem) é o tipo de pergunta que separa quem só usou PyTorch de quem entende o que está fazendo. DDP replica o modelo inteiro em cada GPU e sincroniza gradientes; FSDP (preview do mod. [09](09_treinamento_e_alinhamento.mdx)) particiona o próprio modelo entre GPUs — a diferença entre os dois é exatamente o tipo de trade-off que aparece em decisões reais de infraestrutura de treino. Para a implementação passo a passo de um training loop completo, use o `Curso` de PyTorch listado nas referências abaixo — é o material certo para acompanhar linha a linha.

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

> **Intuição**: depurar uma rede neural é mais parecido com diagnóstico médico do que com debugging de software tradicional — não dá pra colocar um breakpoint na "razão pela qual o modelo não aprende". Em vez disso, você observa sintomas (loss não desce, loss vira NaN, accuracy de treino boa mas validação ruim) e isola causas prováveis por eliminação. `Hooks` em PyTorch são a ferramenta que te dá visibilidade *dentro* da rede durante o forward/backward — sem eles, você só vê a loss final, um sintoma agregado que esconde onde o problema realmente está.
>
> **Cenário hipotético**: imagine que você está treinando e, depois de algumas centenas de passos, a loss vira `NaN`. Causas prováveis, em ordem de frequência: learning rate alto demais (gradiente explode), divisão por zero em alguma loss customizada, `log(0)` numa cross-entropy sem estabilização numérica, ou overflow em precisão mista (`fp16`) sem `loss scaling`. O primeiro passo de diagnóstico costuma ser reduzir o learning rate em 10× e ver se o problema some — se sim, era isso. Para implementar hooks e visualizações concretas (Grad-CAM, saliency maps), consulte os tutoriais oficiais do PyTorch e do W&B listados nas referências da seção 5.7 — a mecânica de registrar um hook (`module.register_forward_hook`) é simples, o valor está em saber *o que* observar.
>
> **Checkpoint**: sem olhar o texto, liste 2 causas prováveis de um loss virar `NaN` durante o treino.

### Referências
- `Paper` **A Recipe for Training Neural Networks** — Karpathy (post). http://karpathy.github.io/2019/04/25/recipe/

---

## Projetos práticos

### Projeto 5.1 — MLP from scratch em NumPy
- Forward, backward, SGD, tudo manual.
- Treine no MNIST. Atinja >97% accuracy.
- Compare velocidade vs PyTorch.

> **Variante guiada**: implemente em 3 checkpoints intermediários — (1) só forward pass, verifique que as dimensões batem e que a saída é uma distribuição de probabilidade válida (soma 1); (2) backward pass, verifique com gradient checking numérico antes de treinar; (3) loop de treino completo. Não avance pro próximo checkpoint sem o anterior funcionando. Para a implementação passo a passo, siga o `Curso` de Karpathy (Zero to Hero) listado nas referências da seção 5.1 — ele constrói exatamente isso, do zero, explicando cada linha.

### Projeto 5.2 — CNN no CIFAR-10
- Em PyTorch.
- Comece com arquitetura simples, depois adicione: BatchNorm, Dropout, augmentation.
- Implemente learning rate schedule + warmup.
- Acompanhe com W&B ou TensorBoard.
- **Meta**: >85% accuracy.

> **Variante guiada**: adicione cada técnica (BatchNorm, Dropout, augmentation, schedule) uma de cada vez, registrando o ganho de accuracy isolado de cada uma antes de adicionar a próxima — isso te dá intuição real de quanto cada peça contribui, em vez de uma "sopa" de técnicas cujo efeito individual você não sabe medir.

### Projeto 5.3 — Implementar ResNet pequena
- Implemente blocos residuais à mão (não use `torchvision.models.resnet`).
- Treine no CIFAR-10.
- Compare com CNN sem skip connections (com mesmo nº de parâmetros): observe o gap.

> **Variante guiada**: antes de treinar a versão funda, treine as duas versões (com e sem skip connection) numa profundidade rasa onde ambas devem treinar bem — confirme que o gap só aparece quando você aumenta a profundidade. Isso isola a variável certa e evita concluir errado (ex.: achar que é overfitting quando na verdade é dificuldade de otimização).

### Projeto 5.4 — char-RNN (gerador de texto caractere a caractere)
- Implemente LSTM em PyTorch.
- Treine em corpus de texto à sua escolha (Shakespeare, código, letras de música).
- Gere texto novo. Compare com versão GPT mínima (preview de mod. [07](07_transformers.mdx)).

> **Variante guiada**: treine e valide o LSTM sozinho, com checkpoints de geração intermediários (a cada X épocas, gere uma amostra e leia), antes de comparar com a versão GPT mínima do mod. 07. O repositório `char-rnn` do Karpathy (referências da seção 5.5) é a implementação de referência pra seguir.

### Projeto 5.5 — VAE no MNIST
- Implemente encoder, decoder, reparameterization trick.
- Visualize espaço latente em 2D.
- Gere amostras interpolando no espaço latente.

> **Variante guiada**: antes de implementar, derive por que o reparameterization trick é necessário (gradiente não passa por uma amostragem estocástica direta — sem o trick, não dá pra fazer backprop através de "amostrar de uma distribuição") — só depois implemente. O paper original (Kingma & Welling, referências da seção 5.6) explica essa motivação na íntegra.

### Projeto 5.6 (cross-link com mod. [14](14_avaliacao_e_seguranca.md)) — Pipeline de experimentação completo
- Use W&B para log.
- Faça grid de experimentos: 3 arquiteturas × 3 learning rates × 2 batch sizes.
- Analise a tabela de resultados.

> **Variante guiada**: rode o grid completo (18 combinações) e documente, para cada resultado, uma hipótese de por que ele saiu melhor ou pior antes de olhar a tabela final — comparar sua hipótese com o resultado real é o que constrói intuição de hiperparâmetro mais rápido do que só rodar e olhar.

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
