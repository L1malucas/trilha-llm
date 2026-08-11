---
id: 16_visao_computacional
title: "Módulo 16 — Visão Computacional"
sidebar_position: 9
---

# Módulo 16 — Visão Computacional

> **Objetivo**: dominar visão computacional moderna — das CNNs clássicas aos Vision Transformers, foundation models visuais (DINOv2, CLIP, SAM), e os modelos visão-linguagem que conectam com o módulo [18](18_multimodal.mdx).
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[15](15_engenharia_producao.mdx). Os módulos de fundamento (Deep Learning, Transformers) que aprofundam CNNs e attention formalmente só vêm mais adiante nesta trilha — este módulo ensina, de forma direta, o que é necessário sobre convolução e sobre a arquitetura Transformer aplicada a imagens para completar os projetos, reaproveitando o Transformer que você já implementou do zero no Projeto 8.3.
>
> **Tempo de referência**: 4–6 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar como uma imagem vira uma sequência de "tokens" para um Vision Transformer.
- Escolher entre CNN e ViT para um cenário dado (tamanho de dataset, restrições de edge).
- Explicar por que DINOv2 é chamado de "BERT da visão" — o que o self-supervised learning nele reaproveita do mesmo princípio de pré-treino sem rótulos.
- Explicar como CLIP alinha imagem e texto no mesmo espaço de embeddings.
- Escolher a ferramenta certa (SAM, YOLO, OCR, VLM) para uma tarefa de visão específica.

---

## Por que isso importa

Mesmo numa trilha centrada em LLMs, visão computacional é parte do tronco: modelos multimodais (GPT-4V, Claude, Gemini, LLaVA) integram visão; agentes com computer-use (mod. [13](13_agentes_tools_protocolos.md)) precisam parsear screenshots; OCR e extração de PDFs são essenciais em RAG profissional (mod. 12), onde boa parte do conhecimento de uma empresa vive em documentos escaneados, não em texto limpo. Muitos conceitos que você já domina (embeddings, transfer learning, attention) reaparecem aqui quase sem tradução.

---

## 16.1 Fundamentos de imagens digitais

### Conceitos
- Pixels, canais (RGB, RGBA, grayscale, HSV, LAB).
- Amostragem, quantização.
- Convolução clássica (filtros: Sobel, Gaussian, Laplacian).
- Operações morfológicas (erosão, dilatação).
- Detecção de bordas (Canny), keypoints (SIFT, ORB) — **clássicos pré-DL**, ainda úteis.

### O que é uma convolução, na prática
Uma convolução aplica um pequeno "filtro" (uma matriz de números, chamada kernel) a cada posição de uma imagem, deslizando-o pixel a pixel: em cada posição, multiplica os valores do kernel pelos pixels sob ele e soma o resultado, produzindo um único valor de saída — repetido em toda a imagem, isso produz um "mapa" que realça algum padrão específico (uma borda, um gradiente de cor).

```python
import numpy as np

def convolucao_2d(imagem, kernel):
    kh, kw = kernel.shape
    h, w = imagem.shape
    saida = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(saida.shape[0]):
        for j in range(saida.shape[1]):
            regiao = imagem[i:i + kh, j:j + kw]
            saida[i, j] = np.sum(regiao * kernel)
    return saida

sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
imagem_exemplo = np.random.rand(10, 10)
bordas = convolucao_2d(imagem_exemplo, sobel_horizontal)
print(bordas.shape)  # (8, 8) — menor que a imagem original, porque o kernel não desliza além da borda
```

O kernel Sobel acima é desenhado à mão para realçar mudanças bruscas de intensidade na horizontal (bordas verticais) — é exatamente esse tipo de kernel que uma camada convolucional de uma rede neural *aprende* automaticamente durante o treino, em vez de ter seus valores definidos manualmente por um humano. Uma CNN empilha várias dessas camadas: as primeiras aprendem a detectar padrões simples (bordas, cores), e camadas mais profundas combinam esses padrões em conceitos cada vez mais abstratos (texturas, partes de objetos, objetos inteiros) — o mod. [05](05_deep_learning.md#54-convolutional-neural-networks-cnns), mais adiante nesta trilha, aprofunda a matemática e a arquitetura completa. Para os projetos deste módulo, você não implementa uma CNN do zero — usa modelos já treinados (transfer learning), então essa intuição do que a convolução calcula é o suficiente para entender o que está acontecendo por dentro.

### Ferramentas
OpenCV (Python, com bindings TS limitados via `opencv.js`), Pillow e scikit-image são as bibliotecas padrão para manipulação e processamento clássico de imagem — usadas para pré/pós-processamento (cropping, augmentation, calibração de cor) mesmo em pipelines dominados por deep learning.

---

## 16.2 CNNs revisitadas (com profundidade)

### Arquiteturas
- **LeNet-5** (didática, a primeira CNN prática).
- **AlexNet** (2012, a virada de chave que popularizou deep learning em visão).
- **VGG** (simplicidade — blocos repetidos de convolução 3×3).
- **GoogLeNet / Inception**.
- **ResNet** — introduz conexões residuais (o mesmo princípio de "soma a entrada de volta" do bloco Transformer que você montou no Projeto 8.3, `x = x + self.attn(...)`, aplicado aqui a blocos convolucionais) para permitir redes muito mais profundas sem o treino degradar.
- **DenseNet**.
- **MobileNet / EfficientNet** (eficiência mobile).
- **ConvNeXt** — ResNet "modernizada" incorporando truques popularizados por Transformers (normalização diferente, ativações diferentes — o mesmo tipo de "atualização de detalhes arquiteturais" que você já viu acontecer com LLMs no mod. [08](08_llms_arquiteturas.md#84-detalhes-arquiteturais-modernos-20242025)).

### Truques de treinamento
- **Data augmentation**: random crop, flip, color jitter, RandAugment, AutoAugment.
- **Mixup**, **CutMix**, **CutOut**.
- **Label smoothing**.
- **EMA** (exponential moving average de pesos).
- **Cosine learning rate**.
- **Stochastic depth**.

O CS231N de Stanford é a referência mais citada para cobrir essas arquiteturas e técnicas em profundidade, caso você queira uma imersão completa além do necessário para os projetos deste módulo.

---

## 16.3 Vision Transformer (ViT) e variantes

### ViT
Aplica o Transformer que você já implementou no Projeto 8.3 — praticamente sem modificação estrutural — a patches de imagem em vez de tokens de texto: uma imagem 224×224 vira 196 patches de 16×16 pixels, cada patch é achatado e projetado num vetor (um "token"), e o Transformer encoder processa essa sequência de patches exatamente como processaria uma sequência de tokens.

> **Intuição**: o título do paper original que introduziu o ViT — "An Image is Worth 16×16 Words" — é literal: ViT trata uma imagem como se fosse uma "frase" de patches. Cada patch de 16×16 pixels vira um "token" (achatado num vetor e projetado linearmente, o mesmo tipo de projeção que `nn.Embedding` faz para tokens de texto no Projeto 8.3, só que aqui é uma projeção linear sobre pixels em vez de uma tabela de consulta sobre índices discretos), soma-se um positional encoding (o mesmo papel que RoPE cumpre no Projeto 8.3 — a ordem dos patches importa tanto quanto a ordem de palavras numa frase), e o Transformer encoder processa essa sequência de patches via self-attention, exatamente como o `GQACausalSelfAttention` que você escreveu, só que sem a máscara causal (uma imagem não tem uma ordem "temporal" a respeitar — cada patch pode legitimamente atender a qualquer outro, inclusive os que viriam "depois" numa leitura esquerda-direita). A vantagem sobre CNN: cada patch pode "atender" diretamente a qualquer outro patch da imagem desde a primeira camada (campo receptivo global imediato), enquanto uma CNN precisa empilhar várias camadas para que informação de cantos opostos da imagem se combine. A desvantagem: CNN tem um *inductive bias* embutido (proximidade espacial importa, parameter sharing) que ajuda muito com pouco dado; ViT precisa aprender esse bias do zero a partir de dados, por isso costuma precisar de datasets bem maiores para superar CNN.
>
> **Checkpoint**: sem olhar o texto, explique como uma imagem 224×224 vira uma sequência de tokens para o ViT. Depois, explique por que ViT costuma precisar de mais dados que CNN para atingir a mesma qualidade.

### Variantes importantes
- **DeiT** — ViT com truques de treino que reduzem a necessidade do dataset gigante (JFT-300M) usado no ViT original.
- **Swin Transformer** — hierárquico, com attention em janelas locais deslizantes (uma ideia parecida com a Sliding Window Attention do mod. 08, aplicada aqui por razões de eficiência espacial, não de contexto longo).
- **MLP-Mixer** — uma alternativa que substitui attention inteiramente por camadas MLP aplicadas alternadamente entre patches e entre canais.
- **MaxViT**, **CoAtNet** — híbridos que combinam convolução e attention no mesmo modelo.

### Quando preferir ViT vs CNN
- **Datasets grandes**: ViT escala melhor.
- **Datasets pequenos**: CNN com inductive bias pode ganhar — você testa isso diretamente no Projeto 16.1.
- **Resolução variável**: ViT lida bem.
- **Inferência em edge**: CNN mobile (MobileNet, EfficientNet) ainda dominam.

---

## 16.4 Self-Supervised Learning em VC

### Contrastive
SimCLR e MoCo v3 treinam o modelo a reconhecer que duas versões aumentadas (crop, cor alterada, etc.) da mesma imagem devem ter embeddings próximos, enquanto imagens diferentes devem ficar distantes — o mesmo princípio de contrastive learning por trás do CLIP (seção 16.6), aplicado dentro de uma única modalidade (só imagem, sem texto pareado).

### Cluster-based / Self-distillation
- **DINO** — self-distillation sem nenhum rótulo.
- **DINOv2** (Meta, 2023) — backbone universal, embeddings de alta qualidade, detalhado abaixo.

### Masked Image Modeling
- **MAE (Masked Autoencoders)** — esconde uma fração grande dos patches da imagem e treina o modelo a reconstruí-los — o análogo visual direto do masked language modeling (mascarar palavras e prever o que falta).
- **BEiT**, **iBOT** — variantes da mesma ideia.

### Por que DINOv2 importa
É o "BERT da visão" atual: um backbone pré-treinado que serve para classificação, segmentação, retrieval e estimativa de profundidade, sem fine-tuning ou com muito pouco.

> **Intuição**: o paralelo com BERT é direto — assim como BERT é pré-treinado numa tarefa de pretexto (masked language modeling) sem rótulos humanos e depois reutilizado (transfer learning) para dezenas de tarefas downstream, DINOv2 é pré-treinado via self-distillation (o modelo aprende comparando suas próprias previsões sob diferentes "views" aumentadas da mesma imagem, sem nenhum rótulo) e vira um backbone genérico — os embeddings que ele produz servem tanto para classificação quanto segmentação quanto busca de imagens similares, geralmente sem precisar de fine-tuning algum. Você extrai e usa esses embeddings diretamente no Projeto 16.3, com o mesmo padrão de `.encode(...)` + similaridade de cosseno que você já usa desde o Projeto 8.5, só que sobre imagens em vez de texto.
>
> **Checkpoint**: sem olhar o texto, explique o paralelo entre DINOv2 e BERT — o que cada um "aprende" sem rótulos, e por que isso generaliza para tarefas diferentes depois?

---

## 16.5 Tarefas clássicas de VC (e o estado da arte 2024–2025)

### Classificação
ImageNet é o benchmark histórico; modelos atuais como ConvNeXt v2, ViT-22B, EVA e InternImage disputam o estado da arte.

### Detecção de objetos
- **R-CNN, Fast/Faster R-CNN** (clássicos two-stage — primeiro propõem regiões candidatas, depois classificam cada uma).
- **YOLO** (one-stage, real-time — prevê caixas e classes numa única passada pela rede) — a família evolui rapidamente (YOLOv8 até YOLOv11 e além); usada no Projeto 16.2.
- **DETR** — trata detecção como um problema de "prever um conjunto de objetos" diretamente com um Transformer, eliminando várias etapas heurísticas dos métodos anteriores.
- **Grounding DINO** — detecção open-vocabulary: em vez de uma lista fixa de classes definida no treino, você descreve o que procurar em texto livre.

### Segmentação
- **U-Net** (clássica, ainda forte em imagem médica).
- **Mask R-CNN**.
- **DeepLabV3+**.
- **SAM (Segment Anything Model)** — Meta, um foundation model para segmentação: dado um ponto ou caixa clicado pelo usuário, segmenta o objeto correspondente, sem ter sido treinado especificamente naquela classe de objeto.
- **SAM 2** — estende SAM para vídeo, propagando a segmentação entre frames; usado no Projeto 16.4.
- **Mask2Former** — unifica segmentação panóptica, de instância e semântica num único modelo.

### Estimativa de profundidade (depth)
Depth Anything (e sua v2) e MiDaS estimam um mapa de profundidade a partir de uma única imagem 2D, sem câmera estéreo ou sensor de profundidade — um problema historicamente difícil que os foundation models recentes resolvem com qualidade surpreendente.

### Pose estimation
OpenPose, HRNet e o toolkit MMPose estimam a posição de articulações do corpo humano numa imagem ou vídeo.

### OCR
Tesseract é o clássico open-source; PaddleOCR, EasyOCR e TrOCR são alternativas mais modernas, frequentemente mais precisas em texto não-latino ou manuscrito. Modelos VLM modernos (LLaVA, Qwen2-VL, ambos disponíveis via Ollama) fazem OCR razoável end-to-end, sem um pipeline dedicado. Marker e Docling são ferramentas especializadas em extração estrutural de PDF (preservando tabelas, títulos, hierarquia) combinando várias dessas técnicas — usadas no Projeto 16.5.

---

## 16.6 Modelos visão-linguagem (preview do mod. [18](18_multimodal.mdx))

- **CLIP** — alinhamento visão-texto via contrastive learning (detalhado abaixo).
- **SigLIP** — versão melhorada do CLIP, trocando a loss softmax original por uma sigmoid loss mais estável em batches grandes.
- **BLIP / BLIP-2** — foco em bootstrapping de captioning (gerar legendas) a partir de dados ruidosos da web.
- **Florence-2** — modelo de propósito geral da Microsoft, cobrindo várias tarefas visão-linguagem com uma única arquitetura.
- **OpenCLIP** (LAION) — reproduções abertas do CLIP original, usadas no Projeto 16.3 e 16.6.

> **Intuição — CLIP**: a ideia central é a mesma de contrastive learning da seção 16.4 — pares positivos (uma imagem e sua legenda real) ficam próximos no espaço de embeddings, pares negativos (imagem com legenda de outra imagem aleatória) ficam distantes. A diferença é que agora os dois "lados" do par positivo vêm de *modalidades diferentes* (pixel e texto), forçando o modelo a aprender uma representação **compartilhada** onde a foto de um cachorro e a palavra "cachorro" acabam próximas no mesmo espaço vetorial, mesmo vindo de encoders separados (um encoder de imagem, um encoder de texto, treinados juntos). É esse espaço compartilhado que permite zero-shot classification (comparar o embedding de uma imagem nova contra embeddings de nomes de classe, sem nenhum treino específico naquelas classes) e é a base de como VLMs modernos (mod. [18](18_multimodal.mdx)) conectam visão e linguagem. Você usa exatamente esse mecanismo no Projeto 16.6.

### Aplicações
- Zero-shot classification.
- Image-text retrieval.
- Embeddings universais para imagens.
- Inicialização de VLMs (mod. [18](18_multimodal.mdx)).

---

## 16.7 Modelos generativos visuais (preview do mod. [19](19_topicos_avancados.md))

GANs (família StyleGAN), Diffusion Models (DDPM, Stable Diffusion, SDXL, FLUX.1 — detalhados a fundo no mod. [19](19_topicos_avancados.md)), modelos autoregressivos de imagem (Parti, MaskGIT), e geração de vídeo (Sora-style, Mochi, CogVideoX, LTX-Video) formam a fronteira de geração visual — fora do escopo prático deste módulo, aprofundados adiante.

---

## 16.8 Vision foundation models (2024–2025)

DINOv2 (backbone universal), SAM 2 (segmentação universal), Depth Anything v2 (depth universal), Grounding DINO 1.5 (detecção open-vocabulary) e CoTracker/TAPIR (point tracking, seguir um ponto específico ao longo de um vídeo) formam o conjunto de modelos "genéricos, prontos para usar" mais citado atualmente. A biblioteca `timm` (PyTorch Image Models) mantém um repositório enorme de pesos pré-treinados prontos para carregar; a biblioteca `transformers` (a mesma usada desde o mod. 09) também dá suporte à maioria dos modelos VLM/VC modernos.

---

## 16.9 Visão em produção

No browser, `transformers.js` (o mesmo do Projeto 10.5) roda modelos de visão via ONNX, com aceleração WebGL/WebGPU quando disponível; o MediaPipe do Google oferece modelos visuais otimizados para rodar localmente. Em mobile, CoreML (iOS) e TensorFlow Lite/MediaPipe (Android) cumprem papel equivalente, tipicamente com modelos quantizados em FP16/INT8 (o mesmo princípio de quantização do mod. 10, aplicado a visão). Em servidor, Triton Inference Server, ONNX Runtime e TorchServe são as opções de produção mais usadas.

---

## Projetos práticos

### Projeto 16.1 — Pipeline clássico vs CNN vs ViT

Você vai comparar 3 abordagens na mesma tarefa de classificação de imagens (5-10 classes, ex.: espécies de flores ou tipos de objetos comuns), incluindo uma variante com dataset reduzido para evidenciar a diferença de comportamento entre CNN e ViT.

**Pré-requisitos**: `pip install scikit-image scikit-learn torchvision transformers`, um dataset de imagens organizado em pastas por classe (datasets pequenos como o Oxford Flowers ou um subconjunto do CIFAR-10 funcionam bem).

**1. Pipeline clássico — features HOG + SVM**:

```python
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
import numpy as np

def extrair_hog(caminho_imagem):
    img = resize(imread(caminho_imagem, as_gray=True), (128, 128))
    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return features

X_treino = np.array([extrair_hog(caminho) for caminho in caminhos_treino])  # suas imagens de treino
y_treino = labels_treino

classificador = SVC(kernel="rbf")
classificador.fit(X_treino, y_treino)
```

HOG (Histogram of Oriented Gradients) é uma técnica clássica pré-deep-learning: divide a imagem em células pequenas, calcula a direção predominante dos gradientes de intensidade (bordas) em cada uma, e monta um vetor de features a partir desses histogramas — não é aprendido, é uma receita fixa desenhada por humanos (o mesmo espírito dos filtros Sobel da seção 16.1).

**2. Pipeline CNN — transfer learning com ResNet50**:

```python
import torch
import torchvision.models as models
import torch.nn as nn

resnet = models.resnet50(weights="IMAGENET1K_V2")
for param in resnet.parameters():
    param.requires_grad = False  # congela os pesos pré-treinados

n_classes = len(set(y_treino))
resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)  # substitui só a última camada, treinável

# o loop de treino é o mesmo padrão do Projeto 8.3: forward, loss, backward, step —
# só que aqui apenas os parâmetros de resnet.fc têm requires_grad=True, então só eles são atualizados
optimizer = torch.optim.AdamW(resnet.fc.parameters(), lr=1e-3)
```

`weights="IMAGENET1K_V2"` carrega pesos já treinados no ImageNet (1000 classes) — a hipótese de transfer learning é que os padrões visuais aprendidos ali (bordas, texturas, formas) são reaproveitáveis para a sua tarefa, mesmo que as classes finais sejam diferentes. Congelar os pesos originais (`requires_grad = False`) e treinar só a última camada é a forma mais simples de transfer learning — o `resnet.fc` original (que produzia 1000 saídas) é substituído por uma nova camada linear com `n_classes` saídas, ainda não treinada.

**3. Pipeline ViT — fine-tuning com Hugging Face**:

```python
from transformers import ViTForImageClassification, ViTImageProcessor

processador = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=n_classes, ignore_mismatched_sizes=True
)

# uso análogo ao AutoModelForCausalLM/AutoTokenizer do Projeto 9.1, só que para imagens em vez de texto
inputs = processador(images=[imread(c) for c in caminhos_treino[:4]], return_tensors="pt")
outputs = vit(**inputs)
```

`ViTImageProcessor` faz o pré-processamento (resize, normalização) que o modelo espera — o análogo do tokenizador, para imagens. `ignore_mismatched_sizes=True` permite substituir a camada de classificação final (originalmente para as classes do treino do ViT) pelas suas `n_classes`, do mesmo jeito que fizemos manualmente com `resnet.fc` acima.

**4. Compare os 3**: accuracy, tempo de treino, e tamanho do modelo salvo, no dataset completo e depois com apenas 20% dos dados de treino (mesmo split de teste nos dois casos). O resultado esperado, confirmando a intuição da seção 16.3: com dataset reduzido, a diferença entre CNN e ViT deve favorecer a CNN relativamente mais do que no dataset completo.

---

### Projeto 16.2 — Fine-tune YOLO em domínio próprio

Você vai treinar um detector de objetos customizado com Ultralytics YOLO, num domínio de sua escolha.

**Pré-requisitos**: `pip install ultralytics`, 200-500 imagens anotadas no formato YOLO (ou um dataset público já anotado, como os disponíveis no Roboflow Universe).

```python
from ultralytics import YOLO

modelo = YOLO("yolo11n.pt")  # versão "nano", pequena o suficiente para treinar em GPU modesta
resultados = modelo.train(data="meu_dataset/data.yaml", epochs=50, imgsz=640)

metricas = modelo.val()
print(f"mAP50-95: {metricas.box.map:.3f}")

modelo.export(format="onnx")  # reaproveita o conceito de exportar para inferência eficiente, mod. 10
```

`data.yaml` descreve o dataset (caminhos para imagens de treino/validação, número de classes, nomes das classes) no formato que o Ultralytics espera — cada imagem tem um arquivo `.txt` correspondente com uma linha por objeto anotado (`classe x_centro y_centro largura altura`, em coordenadas normalizadas 0-1). `mAP50-95` é a métrica padrão de detecção: mede a precisão da detecção em vários limiares de sobreposição entre a caixa prevista e a caixa real, calculada e reportada automaticamente por `modelo.val()`. `export(format="onnx")` gera um artefato pronto para inferência em produção, incluindo em browser (Projeto 16.7) ou edge.

---

### Projeto 16.3 — Embeddings com DINOv2 e comparação com CLIP

Você vai usar DINOv2 (sem fine-tuning) como extrator de features para busca de imagens similares na sua própria coleção, e comparar com embeddings do CLIP na mesma tarefa.

**Pré-requisitos**: `pip install transformers streamlit`, uma coleção de ~50-100 imagens pessoais.

**1. Extraia embeddings com DINOv2 e com CLIP**:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np

processor_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
modelo_dino = AutoModel.from_pretrained("facebook/dinov2-base")

def embed_dinov2(imagem_pil):
    inputs = processor_dino(images=imagem_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = modelo_dino(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # embedding do token [CLS]

from transformers import CLIPProcessor, CLIPModel

processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
modelo_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def embed_clip(imagem_pil):
    inputs = processor_clip(images=imagem_pil, return_tensors="pt")
    with torch.no_grad():
        embedding = modelo_clip.get_image_features(**inputs)
    return embedding.squeeze().numpy()
```

O token `[CLS]` em `last_hidden_state[:, 0, :]` é, por convenção arquitetural (a mesma de BERT), o token cujo vetor final é treinado para resumir a sequência inteira — em ViT/DINOv2, resume a imagem inteira, do mesmo jeito que resumiria uma frase inteira de texto num modelo BERT.

**2. Implemente busca por similaridade** (o mesmo padrão exato do Projeto 8.5, agora sobre embeddings de imagem):

```python
def buscar_similares(imagem_query, embeddings_colecao, imagens_colecao, embed_fn, k=5):
    query_emb = embed_fn(imagem_query)
    sims = [np.dot(query_emb, e) / (np.linalg.norm(query_emb) * np.linalg.norm(e)) for e in embeddings_colecao]
    top_k = np.argsort(sims)[-k:][::-1]
    return [imagens_colecao[i] for i in top_k]
```

**3. Compare DINOv2 vs CLIP** na mesma coleção: DINOv2 tende a agrupar por similaridade *visual* pura (composição, textura, forma), enquanto CLIP tende a agrupar por similaridade *semântica* (o que o objeto "é", influenciado pelo alinhamento com texto durante o treino) — teste com uma imagem query ambígua (por exemplo, um objeto com textura incomum) e veja se os top-5 de cada modelo refletem essa diferença.

**4. Frontend simples com Streamlit** (`pip install streamlit`, já instalado):

```python
import streamlit as st
from PIL import Image

st.title("Busca de imagens similares")
arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "png"])
if arquivo:
    imagem_query = Image.open(arquivo)
    st.image(imagem_query, caption="Query", width=200)
    resultados = buscar_similares(imagem_query, embeddings_dinov2, colecao, embed_dinov2)
    st.image(resultados, width=150)
```

Rode com `streamlit run app.py` — Streamlit transforma um script Python simples numa interface web interativa sem você escrever HTML/JS, útil para demos rápidas como esta.

---

### Projeto 16.4 — Segmentação com SAM 2

Você vai usar o SAM 2 para segmentar objetos a partir de um ponto clicado, numa interface mínima "click to segment".

**Pré-requisitos**: `pip install git+https://github.com/facebookresearch/sam2.git`, os pesos pré-treinados do SAM 2 (baixados conforme instruções do repositório oficial).

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

sam2_model = build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt")
predictor = SAM2ImagePredictor(sam2_model)

imagem = np.array(Image.open("minha_foto.jpg"))
predictor.set_image(imagem)

ponto_clicado = np.array([[320, 240]])  # coordenadas x, y de onde o usuário "clicou"
label_ponto = np.array([1])  # 1 = ponto positivo (pertence ao objeto a segmentar)

mascaras, scores, _ = predictor.predict(point_coords=ponto_clicado, point_labels=label_ponto, multimask_output=True)
melhor_mascara = mascaras[np.argmax(scores)]
```

`multimask_output=True` faz o SAM 2 retornar 3 hipóteses de segmentação diferentes para o mesmo ponto (útil quando o ponto clicado é ambíguo — por exemplo, clicar num botão de uma camisa pode significar "segmentar o botão" ou "segmentar a camisa inteira"), cada uma com um score de confiança; `melhor_mascara` pega a de maior score. Para uma UI de verdade, capture as coordenadas do clique do usuário (via Streamlit, reaproveitando o Projeto 16.3, com `st.image` e um componente de captura de clique) em vez de coordenadas fixas como no exemplo acima.

**Bônus — vídeo**: o SAM 2 (diferente do SAM original) suporta propagar uma máscara inicial ao longo dos frames de um vídeo (`SAM2VideoPredictor`), rastreando o objeto sem precisar clicar em cada frame — útil para tarefas como remover fundo de vídeo ou rastrear um objeto em movimento.

---

### Projeto 16.5 — OCR + extração estruturada de PDFs

Você vai extrair texto e tabelas de PDFs preservando estrutura, comparando uma abordagem clássica com um modelo visão-linguagem fazendo a extração end-to-end.

**Pré-requisitos**: `pip install docling`, Ollama com um modelo de visão (`ollama pull llava` ou `ollama pull qwen2.5vl`).

**1. Extração estrutural com Docling**:

```python
from docling.document_converter import DocumentConverter

conversor = DocumentConverter()
resultado = conversor.convert("relatorio.pdf")
markdown_extraido = resultado.document.export_to_markdown()

print(markdown_extraido[:2000])
```

Docling combina detecção de layout (identificar onde estão títulos, parágrafos, tabelas na página — um problema de visão computacional, usando um modelo de detecção de objetos como os da seção 16.5) com OCR e parsing de tabelas, produzindo Markdown estruturado em vez de um bloco de texto corrido — preservando, por exemplo, que uma tabela é uma tabela, não texto solto com espaços.

**2. Extração end-to-end com VLM via Ollama** — modelos de visão-linguagem locais aceitam imagens diretamente no mesmo padrão de API que você já usa desde o Projeto 8.1:

```python
import requests
import base64

def extrair_com_vlm(caminho_imagem_pagina):
    with open(caminho_imagem_pagina, "rb") as f:
        imagem_b64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5vl",
            "prompt": "Extraia todo o texto e tabelas desta página, em formato Markdown, preservando a estrutura.",
            "images": [imagem_b64],
            "stream": False,
        },
    )
    return response.json()["response"]
```

A única diferença estrutural em relação às chamadas ao Ollama que você já fez dezenas de vezes é o campo `images` — uma lista de imagens codificadas em base64. Compare a saída das duas abordagens na mesma página de PDF (convertida antes para imagem, com uma biblioteca como `pdf2image`): Docling tende a ser mais consistente estruturalmente (tabelas quase sempre corretas), enquanto o VLM pode ser mais flexível com layouts incomuns, mas com maior variância na formatação exata da saída.

---

### Projeto 16.6 — Zero-shot classification com CLIP

Você vai construir um classificador de imagens que funciona em categorias arbitrárias, definidas em texto, sem nenhum treino específico naquelas categorias.

**Pré-requisitos**: `pip install open_clip_torch`.

```python
import open_clip
import torch
from PIL import Image

modelo, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def classificar_zero_shot(caminho_imagem, categorias):
    imagem = preprocess(Image.open(caminho_imagem)).unsqueeze(0)
    textos = tokenizer([f"uma foto de um(a) {c}" for c in categorias])

    with torch.no_grad():
        emb_imagem = modelo.encode_image(imagem)
        emb_textos = modelo.encode_text(textos)
        emb_imagem /= emb_imagem.norm(dim=-1, keepdim=True)
        emb_textos /= emb_textos.norm(dim=-1, keepdim=True)
        similaridades = (emb_imagem @ emb_textos.T).squeeze(0)

    melhor = similaridades.argmax().item()
    return categorias[melhor], similaridades[melhor].item()

print(classificar_zero_shot("foto_teste.jpg", ["gato", "cachorro", "pássaro", "carro"]))
```

`emb_imagem @ emb_textos.T` é o mesmo produto interno usado como similaridade de cosseno em todo o resto da trilha, só que aqui comparando um único embedding de imagem contra vários embeddings de texto (um por categoria candidata) — a categoria com maior similaridade "vence", sem que o modelo tenha sido treinado especificamente para distinguir essas categorias.

**Teste a sensibilidade ao formato do prompt**: rode `classificar_zero_shot` trocando `"uma foto de um(a) {c}"` por só `"{c}"`, e compare a confiança (`similaridades[melhor]`) e a taxa de acerto nas mesmas imagens de teste — CLIP é sensível ao formato textual do prompt de um jeito estruturalmente parecido com o que você já viu em prompt engineering (mod. 11), porque no fundo é a mesma arquitetura de encoder de texto por trás.

**Compare com o modelo supervisionado** do Projeto 16.1 na mesma tarefa (se as classes coincidirem): o modelo supervisionado, treinado especificamente nessas classes, tende a ganhar em accuracy pura, mas exigiu dados rotulados; o zero-shot não exigiu nenhum.

---

### Projeto 16.7 — Inferência em browser

Você vai rodar um classificador de imagem inteiramente no navegador com `transformers.js`, reaproveitando o setup do Projeto 10.5.

**Pré-requisitos**: o setup do Projeto 10.5 (`@xenova/transformers` já instalado).

```javascript
import { pipeline } from "@xenova/transformers";

async function classificarImagem(urlImagem) {
  const t0 = performance.now();
  const classificador = await pipeline("image-classification", "Xenova/vit-base-patch16-224");
  const t1 = performance.now();

  const t2 = performance.now();
  const resultado = await classificador(urlImagem);
  const t3 = performance.now();

  console.log(`Carregamento do modelo: ${(t1 - t0).toFixed(0)}ms`);
  console.log(`Classificação: ${(t3 - t2).toFixed(0)}ms`);
  console.log(resultado);
}

classificarImagem("https://exemplo.com/foto.jpg");
```

Mesma estrutura do classificador de sentimento do Projeto 10.5, trocando `"sentiment-analysis"` por `"image-classification"` e o modelo por uma versão ONNX do ViT. Compare o tempo de classificação medido aqui (client-side, rodando no seu navegador) contra o tempo de uma chamada equivalente a um servidor rodando o mesmo modelo (o backend do Projeto 15.4, adaptado para aceitar uma imagem) — o ganho de latência de rede do client-side é real, mas o tempo de carregamento inicial do modelo (baixado uma vez, cacheado depois) é um custo que a versão servidor não tem.

---

## Erros comuns

- **Augmentation errado**: rotação livre num dataset onde orientação importa (texto, faces).
- **Normalização inconsistente** entre treino e inferência (mean/std diferentes).
- **Overfitting silencioso** em datasets pequenos sem augmentation.
- **Confundir mAP@50 com mAP@50:95** — métricas COCO têm convenções específicas.
- **Esquecer de fixar data leakage** quando há mesma "instância" em splits diferentes (ex: vários frames do mesmo vídeo).
- **Ignorar resolução** — a maioria dos modelos VC tem ponto-doce de input específico.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| ViT | Multimodal (mod. [18](18_multimodal.mdx)) |
| CLIP, SigLIP | Multimodal (mod. [18](18_multimodal.mdx)) |
| Diffusion preview | Tópicos avançados (mod. [19](19_topicos_avancados.md)) |
| DINOv2 embeddings | Agentes com visão (mod. [13](13_agentes_tools_protocolos.md) estendido) |

---

## Checklist de saída

- [ ] Comparei pipeline clássico, CNN e ViT na mesma tarefa, com dataset completo e reduzido (se não, revise o Projeto 16.1).
- [ ] Treinei YOLO em domínio customizado e avaliei em mAP (se não, revise o Projeto 16.2).
- [ ] Apliquei DINOv2 como feature extractor e comparei com CLIP na mesma tarefa de retrieval (se não, revise o Projeto 16.3).
- [ ] Usei SAM 2 para segmentação a partir de um ponto (se não, revise o Projeto 16.4).
- [ ] Tenho um pipeline de extração de PDF funcionando, comparado com extração via VLM (se não, revise o Projeto 16.5).
- [ ] Construí um classificador zero-shot com CLIP e testei sensibilidade ao formato do prompt (se não, revise o Projeto 16.6).
- [ ] Construí uma demo de inferência client-side no browser (se não, revise o Projeto 16.7).
