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

### Por que skip connections importam para LLMs
Transformers usam residual connections derivadas das ResNets. Entender o problema que ResNet resolve é entender por que Transformers profundos treinam.

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

### Por que importa
- Embeddings em LLMs são, conceitualmente, "encoders" treinados.
- Difusão (mod. [19](19_topicos_avancados.md)) é a evolução desses paradigmas.

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

### Referências
- `Paper` **A Recipe for Training Neural Networks** — Karpathy (post). http://karpathy.github.io/2019/04/25/recipe/

---

## Projetos práticos

### Projeto 5.1 — MLP from scratch em NumPy
- Forward, backward, SGD, tudo manual.
- Treine no MNIST. Atinja >97% accuracy.
- Compare velocidade vs PyTorch.

### Projeto 5.2 — CNN no CIFAR-10
- Em PyTorch.
- Comece com arquitetura simples, depois adicione: BatchNorm, Dropout, augmentation.
- Implemente learning rate schedule + warmup.
- Acompanhe com W&B ou TensorBoard.
- **Meta**: >85% accuracy.

### Projeto 5.3 — Implementar ResNet pequena
- Implemente blocos residuais à mão (não use `torchvision.models.resnet`).
- Treine no CIFAR-10.
- Compare com CNN sem skip connections (com mesmo nº de parâmetros): observe o gap.

### Projeto 5.4 — char-RNN (gerador de texto caractere a caractere)
- Implemente LSTM em PyTorch.
- Treine em corpus de texto à sua escolha (Shakespeare, código, letras de música).
- Gere texto novo. Compare com versão GPT mínima (preview de mod. [07](07_transformers.mdx)).

### Projeto 5.5 — VAE no MNIST
- Implemente encoder, decoder, reparameterization trick.
- Visualize espaço latente em 2D.
- Gere amostras interpolando no espaço latente.

### Projeto 5.6 (cross-link com mod. [14](14_avaliacao_e_seguranca.md)) — Pipeline de experimentação completo
- Use W&B para log.
- Faça grid de experimentos: 3 arquiteturas × 3 learning rates × 2 batch sizes.
- Analise a tabela de resultados.

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
