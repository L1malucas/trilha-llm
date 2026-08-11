---
id: 08_llms_arquiteturas
title: "Módulo 08 — Arquiteturas de LLMs"
sidebar_position: 1
---

# Módulo 08 — Arquiteturas de LLMs

> **Objetivo**: conhecer as famílias de LLMs, suas escolhas arquiteturais, scaling laws, e os modelos abertos de 2024–2025 que definem o estado da arte open-source — e sair com uma mini-LLaMA rodando, um comparativo de modelos reais, e um experimento próprio de scaling laws.
>
> **Pré-requisitos**: nenhum específico — este é o primeiro módulo desta trilha. Onde um conceito de Transformers (Q/K/V, attention, blocos) for necessário, ele é explicado aqui mesmo, na medida do que o projeto precisa.
>
> **Tempo de referência**: 3–4 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Escolher entre encoder-only, decoder-only e encoder-decoder para uma tarefa dada, com justificativa.
- Explicar o resultado de Chinchilla (razão tokens/parâmetro) e por que ele mudou a forma de treinar LLMs.
- Explicar, de fato, o que cada mudança arquitetural moderna (RMSNorm, SwiGLU, RoPE, GQA, Sliding Window Attention, MoE, multi-token prediction, long context) faz e por que existe — não só reconhecer o nome.
- Explicar por que modelos de raciocínio (o1, DeepSeek-R1) usam mais compute em inferência, não só em treino.
- Avaliar um modelo aberto por critérios técnicos verificáveis — licença, tamanho, idioma, resultado em benchmark — em vez de reputação.
- Ter implementado, do zero, um Transformer decoder-only com a arquitetura usada em LLaMA/Mistral.

---

## Por que isso importa

Saber que "GPT é decoder-only" não é suficiente para decidir qual modelo usar num projeto real. **LLaMA 3, Mistral, Qwen 2.5, Gemma 2 e Phi-3** são todos decoder-only e ainda assim fazem escolhas arquiteturais diferentes entre si — e cada uma dessas escolhas se traduz em diferenças concretas de performance, consumo de memória e custo de inferência, que este módulo detalha seção por seção.

---

## 8.1 Famílias arquiteturais

### Encoder-only
**Para quê**: classificação, extração, embeddings, NER.

Exemplos conhecidos: **BERT**, **RoBERTa** (BERT com treino melhorado), **DeBERTa** (attention "desentrelaçada", ainda muito competitivo), **ELECTRA** (mais eficiente de treinar que BERT), **ModernBERT** (2024, atualiza BERT com os avanços arquiteturais das seções seguintes).

### Decoder-only (autoregressivos)
**Para quê**: geração, chat, completion, raciocínio.

Exemplos conhecidos: a família **GPT** (GPT-2, GPT-3), **LLaMA** (1, 2, 3), **Mistral** e **Mixtral**, **Qwen 2/2.5**, **Gemma/Gemma 2**, **Phi-3**, **DeepSeek-V3** e **DeepSeek-R1**. É a família que você vai construir uma versão mínima neste módulo (Projeto 8.3).

### Encoder-Decoder
**Para quê**: text-to-text unificado (tradução, sumarização, QA).

Exemplos conhecidos: **T5**, **BART**, **mT5** (multilíngue), **Flan-T5** (instruction-tuned), **NLLB-200** (tradução em 200 idiomas).

> **Intuição**: encoder-only é um **revisor de texto** — lê o texto inteiro de uma vez (pode olhar pra frente e pra trás) antes de opinar sobre cada palavra, ótimo pra entender/classificar. Decoder-only é um **contador de histórias** que só sabe o que já foi dito até agora e precisa prever a próxima palavra sem espiar o futuro — ótimo pra gerar. Encoder-decoder é um **tradutor**: lê a frase inteira de origem antes de começar a escrever a tradução palavra por palavra, consultando o que leu a cada palavra que escreve.
>
> Se você ainda não construiu intuição sobre *como* attention funciona por dentro (Q/K/V, causal mask), o Projeto 8.3 abaixo te leva por isso na prática — não precisa entender a fundo antes de tentar.

---

## 8.2 Scaling Laws

### Conceito
Como performance varia com **parâmetros (N)**, **dados (D)**, e **compute (C)**? Existe um equilíbrio ótimo?

Dois resultados importantes moldaram como LLMs são treinados hoje: o trabalho de **Kaplan et al. (2020)**, que sugeriu escalar parâmetros mais rápido que dados, e o trabalho que ficou conhecido como **"Chinchilla"** (Hoffmann et al., 2022), que corrigiu essa recomendação — mostrando que a maioria dos modelos daquela época estava **subtreinada em dados** relativo ao seu tamanho, e que o ponto ótimo de compute fica perto de **20 tokens de treino por parâmetro do modelo**.

> **Intuição**: para um orçamento fixo de compute, você pode gastar em mais *parâmetros* (modelo maior) ou mais *dados* (mais tokens de treino). Um exemplo concreto: um modelo de 10B parâmetros, Chinchilla-ótimo, treinaria em ~200B tokens — não menos, não muito mais.
>
> **Por que "além de Chinchilla-ótimo" também faz sentido na prática**: Chinchilla otimiza o custo de *treino*, mas ignora o custo de *inferência*. Um modelo menor, treinado bem além do ponto Chinchilla-ótimo (mais tokens do que o "ideal" pra aquele tamanho), pode alcançar qualidade comparável a um modelo maior Chinchilla-ótimo, só que depois é mais barato de rodar em produção milhões de vezes. É por isso que LLaMA 3 8B foi treinado em ~15 trilhões de tokens — muito além do que Chinchilla recomendaria pra esse tamanho — porque o custo de inferência ao longo da vida do modelo supera de longe o custo extra de treino.
>
> **Checkpoint**: sem olhar o texto, explique em uma frase o que Chinchilla corrigiu em relação ao trabalho anterior de Kaplan. Depois, explique por que treinar "além do Chinchilla-ótimo" pode ser uma boa decisão de engenharia, mesmo sendo "sub-ótimo" em compute de treino.

Você vai reproduzir esse trade-off empiricamente, em miniatura, no Projeto 8.4.

### Implicação prática
- **Modelos pequenos bem treinados** (ex: Phi-3 mini, Llama 3.2 3B) podem competir com modelos grandes.
- **Tokens consumidos** é métrica tão importante quanto parâmetros.

---

## 8.3 Pré-treinamento: o que faz uma LLM "saber"

### Componentes
- **Dataset massivo** (trilhões de tokens): Common Crawl, Wikipedia, livros, código, papers, fóruns.
- **Curadoria e limpeza**: deduplicação, filtros de qualidade, balanceamento de domínios.
- **Tokenização**: quebrar o texto em unidades (tokens) que o modelo processa — cobrimos isso na prática no Projeto 8.3, quando você tokeniza um corpus caractere a caractere.
- **Objetivo**: next-token prediction (decoder) ou masked language modeling (encoder) — o modelo aprende prevendo pedaços de texto escondidos ou o próximo token, sem precisar de rótulo humano.
- **Treinamento distribuído**: quando o modelo não cabe (ou não treina rápido o bastante) numa única GPU, o trabalho é dividido entre várias — dados diferentes por GPU (data parallel), partes do modelo em GPUs diferentes (tensor/pipeline parallel), ou uma combinação (FSDP).

### Datasets abertos relevantes
- **The Pile** (EleutherAI).
- **C4** (Colossal Clean Crawled Corpus) — usado por T5.
- **RedPajama** (reprodução do dataset do LLaMA original). https://github.com/togethercomputer/RedPajama-Data
- **FineWeb / FineWeb-Edu** (HuggingFace, 2024). https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **Dolma** (Allen AI).
- **Common Corpus** (atenção especial a licenças abertas).

> O ciclo completo de pré-treinamento em escala real (curadoria → tokenização → treino distribuído → checkpoints) é tema do módulo de Treinamento e Alinhamento, mais adiante nesta trilha — aqui o foco é nos ingredientes; você constrói uma versão pequena e completa desse ciclo no Projeto 8.3.

---

## 8.4 Detalhes arquiteturais modernos (2024–2025)

### O que mudou desde o Transformer original

Todo LLM decoder-only moderno (LLaMA, Mistral, Qwen) combina um conjunto de mudanças específicas em relação ao Transformer descrito no paper original de 2017. Cada uma resolve um problema concreto:

- **RMSNorm em vez de LayerNorm**: normaliza a escala de um vetor usando só a raiz quadrada média dos seus valores (`sqrt(mean(x²))`), sem subtrair a média como LayerNorm faz. É uma operação mais simples (menos passos, mais rápida) e, na prática, funciona tão bem ou melhor que LayerNorm em modelos grandes. Você implementa isso do zero no Projeto 8.3 — é uma classe de ~6 linhas.
- **SwiGLU em vez de ReLU/GELU no FFN**: a rede feed-forward de cada bloco Transformer processa cada token individualmente, expandindo e depois comprimindo a dimensão. SwiGLU troca a ativação simples (ReLU/GELU aplicada uma vez) por um mecanismo de "portão" — duas projeções lineares, uma passada por uma ativação (SiLU) e multiplicada elemento a elemento pela outra — que na prática melhora a qualidade do modelo pro mesmo número de parâmetros. Também implementado do zero no Projeto 8.3.
- **RoPE (Rotary Position Embedding) em vez de positional encoding somado**: sem informação de posição, um Transformer não distingue "o cão mordeu o homem" de "o homem mordeu o cão" — são o mesmo conjunto de tokens, só a ordem muda o sentido. RoPE injeta a posição *rotacionando* os vetores de Query e Key dentro do próprio cálculo de attention (em vez de somar um vetor de posição ao embedding, como o Transformer original fazia) — isso faz com que a relação entre dois tokens dependa da *distância relativa* entre eles, não da posição absoluta de cada um, o que ajuda o modelo a generalizar melhor para sequências mais longas que as vistas no treino.
- **Grouped-Query Attention (GQA)**: em multi-head attention comum, cada cabeça de Query tem sua própria cabeça de Key e Value. GQA faz várias cabeças de Query **compartilharem** a mesma cabeça de Key/Value — por exemplo, 8 cabeças de Query dividida em 2 grupos de 4, cada grupo usando uma única cabeça K/V. Isso reduz o tamanho do cache que precisa ser guardado durante geração de texto (o "KV-cache"), tornando a inferência mais barata em memória, com perda de qualidade mínima.
- **Sliding Window Attention** (usado pelo Mistral): em vez de cada token prestar attention em *todos* os tokens anteriores (custo que cresce quadraticamente com o tamanho da sequência), cada token só presta attention numa janela fixa de tokens recentes (ex.: os últimos 4096). Isso limita o custo por token a um valor constante, não crescente — o preço é que, numa única camada, um token não "vê" diretamente algo muito distante (mas informação ainda pode se propagar indiretamente, camada após camada).
- **Mixture of Experts (MoE)**: em vez de uma única rede feed-forward densa processando cada token, MoE tem várias redes "especialistas" (experts) e um roteador aprendido que decide, por token, quais 1-2 especialistas ativar. O modelo tem muito mais parâmetros *totais* (todos os especialistas somados), mas cada token só passa por uma fração pequena deles — é assim que o Mixtral 8×7B tem ~47B parâmetros totais mas custo de inferência por token comparável a um modelo denso de ~13B.
- **Multi-token prediction**: em vez de treinar o modelo pra prever só o próximo token, adiciona-se cabeças de saída extras que também tentam prever o token seguinte ao próximo (t+2, t+3) como sinal de treino auxiliar. Isso melhora a eficiência do treino (mais sinal de aprendizado por passo) e, em alguns casos, permite gerar mais de um token por passo de inferência.
- **Long context (128k → 1M+ tokens)**: técnicas como YaRN e LongRoPE reescalam matematicamente as frequências usadas pelo RoPE para que posições muito além do que o modelo viu no treino ainda produzam rotações sensatas — sem isso, um modelo treinado com contexto de 8k tokens se comporta de forma imprevisível ao processar a posição 100.000.

> **Intuição — por que RMSNorm e Sliding Window Attention resolvem problemas de escala**: a maioria dessas mudanças existe porque, em escala (bilhões de parâmetros, milhões de tokens de contexto), qualquer operação repetida em cada camada ou em cada token vira um custo real, não desprezível. RMSNorm remove uma subtração (o cálculo da média) que LayerNorm faz e que, repetida bilhões de vezes, tem um custo mensurável. Sliding Window Attention limita um crescimento de custo que, sem essa limitação, seria quadrático no tamanho da sequência. Em ambos os casos, o mecanismo central continua sendo o mesmo attention que você implementa no Projeto 8.3 — essas mudanças o tornam mais barato de rodar em escala, não o substituem por outra coisa.
>
> **Checkpoint**: sem olhar o texto, explique por que MoE permite mais parâmetros sem aumentar proporcionalmente o custo de inferência por token. Depois, explique a diferença entre o que GQA economiza (memória de cache) e o que Sliding Window Attention economiza (custo de cálculo por token).

---

## 8.5 Modelos de raciocínio (reasoning)

A onda de 2024–2025 — modelos otimizados para "pensar" antes de responder (o1, DeepSeek-R1).

### Conceito
Modelos geram uma "chain of thought" (cadeia de raciocínio) longa *antes* da resposta final. O treinamento usa aprendizado por reforço com recompensa em corretude — o modelo é recompensado quando chega à resposta certa, não quando um humano anota "o passo certo" de raciocínio. Isso aumenta drasticamente a performance em matemática, código e lógica.

> **Intuição**: até aqui, escalar um LLM significava gastar mais compute em *treino* (mais parâmetros, mais dados, seção 8.2). Modelos de raciocínio introduzem um segundo eixo: gastar mais compute em *inferência*, deixando o modelo "pensar em voz alta" (gerar uma cadeia de raciocínio longa) antes de se comprometer com a resposta final — mais tokens gerados internamente, mais chance de corrigir um erro de raciocínio no caminho.
>
> **Aplicação real**: é por isso que modelos de raciocínio custam mais por resposta (mais tokens gerados, mesmo que invisíveis ao usuário final) mas performam muito melhor em tarefas que exigem múltiplos passos lógicos corretos em sequência — errar um passo no meio geralmente invalida a resposta final nessas tarefas, então "pensar mais antes de responder" compensa o custo extra.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre escalar compute de *treino* e escalar compute de *inferência* — qual delas os modelos de raciocínio exploram?

---

## 8.6 Embeddings e modelos especializados

### Embedding models
Modelos treinados especificamente pra produzir vetores densos que capturam significado de texto (não pra gerar texto): **Sentence-BERT**, **E5** (Microsoft), **BGE** (BAAI), **Nomic Embed**, **Jina Embeddings**. **MTEB** (Massive Text Embedding Benchmark) é o benchmark canônico pra comparar todos eles.

### Reranker models
Modelos que reordenam por relevância um conjunto pequeno de resultados já recuperados — mais precisos, mais caros por comparação. Exemplos: cross-encoders (variantes do Sentence-BERT), `bge-reranker`.

### Code models
**CodeLLaMA**, **DeepSeek-Coder**, **Qwen2.5-Coder**, **StarCoder 2** — LLMs especializados/afinados especificamente em código.

### Math/Science models
**DeepSeek-Math**, **Qwen-Math**, **MathStral** — especializados em raciocínio matemático.

Você vai usar embedding models na prática no Projeto 8.5.

---

## 8.7 Onde encontrar modelos abertos

- **Hugging Face Hub** — repositório central. https://huggingface.co/models
- **Open LLM Leaderboard** (HF). https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **LMSys Chatbot Arena** (avaliação humana via comparação A/B). https://arena.lmsys.org/
- **Artificial Analysis** (benchmarks comparativos com custos). https://artificialanalysis.ai/

### Famílias open de 2024–2025 que valem conhecer
| Família | Tamanhos | Origem | Característica |
|---|---|---|---|
| **LLaMA 3.x** | 1B–405B | Meta | Padrão da indústria open |
| **Mistral / Mixtral** | 7B, MoE 8×7B, 8×22B | Mistral AI | Eficiência |
| **Qwen 2.5** | 0.5B–72B | Alibaba | Multilíngue, code, math |
| **Gemma 2** | 2B, 9B, 27B | Google | Distilação de Gemini |
| **Phi-3 / Phi-4** | 3.8B, 7B, 14B | Microsoft | "Small but capable" |
| **DeepSeek V3 / R1** | até ~671B (MoE) | DeepSeek | Raciocínio e custo |
| **Command R+** | 104B | Cohere | RAG-otimizado |
| **OLMo 2** | 7B, 13B | Allen AI | Totalmente aberto (dados + pesos) |

---

## Projetos práticos

### Projeto 8.1 — Comparativo de modelos abertos

**Pré-requisitos**:
- [Ollama](https://ollama.com/) instalado (Mac/Linux/Windows — baixe e instale como qualquer aplicativo).
- Python 3.10+ com `pip install requests`.
- Espaço em disco: cada modelo de ~7-9B em quantização padrão do Ollama ocupa ~5 GB.

**1. Baixe os modelos** (rode no terminal, um de cada vez — cada `pull` demora alguns minutos):

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
```

**2. Escreva o script de comparação**: o código abaixo percorre os 4 modelos e, para cada um, envia os mesmos prompts à API local do Ollama, mede quanto tempo cada resposta levou, e grava tudo (modelo, prompt, resposta, latência) num arquivo JSON para você comparar depois.

```python
import requests
import time
import json

MODELS = ["llama3:8b", "mistral:7b", "gemma2:9b", "qwen2.5:7b"]
PROMPTS = [
    "Explique o que é overfitting em uma frase.",
    "Escreva uma função Python que inverte uma string sem usar [::-1].",
    "Traduza para inglês: 'a matemática é a base de tudo em machine learning'.",
    # adicione o resto até 20 prompts próprios, cobrindo geração, raciocínio, código e PT-BR
]

def query_ollama(model, prompt):
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    elapsed = time.time() - start
    return response.json()["response"], elapsed

results = []
for model in MODELS:
    for prompt in PROMPTS:
        text, elapsed = query_ollama(model, prompt)
        results.append({
            "model": model,
            "prompt": prompt,
            "response": text,
            "latency_s": round(elapsed, 2),
        })
        print(f"{model:15s} {elapsed:5.2f}s  {prompt[:40]}")

with open("comparativo.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

**3. Rode** (`python comparativo.py`, com o Ollama aberto em segundo plano) **e analise**: abra `comparativo.json` e compare as respostas lado a lado pra cada prompt — não só a latência. Documente: qual modelo foi mais rápido, qual teve a melhor qualidade subjetiva, e como cada um se saiu especificamente nos prompts em PT-BR (costuma ser onde a diferença entre modelos aparece mais).

> Use exatamente os mesmos 20 prompts para os 4 modelos, na mesma ordem. Se os prompts variarem entre modelos, a diferença observada pode vir do prompt, não do modelo — controlar a entrada é o que torna a comparação válida.

---

### Projeto 8.2 — Ler e fichar 3 papers de LLMs

**Pré-requisitos**: nenhuma instalação — só acesso à internet.

**Como ler um paper técnico sem travar**: na primeira passada, não tente entender cada equação — leia só o abstract, a introdução, e os títulos das seções/figuras, pra mapear a estrutura geral. Na segunda passada, foque nas seções sobre arquitetura e dados de treino (geralmente é onde está a informação mais reaproveitável). Deixe a seção de resultados/benchmarks pro fim.

**1. Escolha 3 papers** entre os citados neste módulo (sugestão: *The Llama 3 Herd of Models*, *Mixtral of Experts*, *DeepSeek-V3 Technical Report* — todos gratuitos no arXiv).

**2. Para cada paper, produza uma ficha de 1-2 páginas em Markdown, respondendo**:
- **Arquitetura**: decoder-only? MoE? Quantas camadas, qual dimensão, quantas cabeças de attention? GQA? Que tipo de positional encoding?
- **Dados**: quantos tokens de treino, de onde vieram, como foram filtrados/deduplicados?
- **Treino**: hardware usado, tempo de treino, hiperparâmetros principais (learning rate, batch size).
- **Avaliação**: em quais benchmarks o modelo foi testado, e como ele se compara aos concorrentes da época?
- **O que me surpreendeu**: pelo menos uma decisão de design que você não esperava, ou não entendia até ler o paper.

---

### Projeto 8.3 — Implemente uma mini-LLaMA

Este é o projeto mais importante do módulo — você vai construir um Transformer decoder-only com a arquitetura moderna (RMSNorm, SwiGLU, RoPE, GQA), treiná-lo do zero, e gerar texto com ele.

**Pré-requisitos**:
- Python 3.10+ com `pip install torch numpy`.
- Não precisa de GPU — o modelo é pequeno o bastante pra treinar em CPU em minutos, só mais devagar. Se tiver GPU (mesmo modesta), o código já funciona com ela (`.to("cuda")` ou `.to("mps")` no Mac).

**O que é um tensor, rapidamente**: um tensor é só um array multidimensional (como um array do NumPy) com suporte a diferenciação automática — o PyTorch calcula os gradientes de qualquer operação feita com tensores automaticamente, o que é o que permite treinar a rede depois.

#### 1. RMSNorm no lugar de LayerNorm

Normalizar um vetor, aqui, significa reescalar seus valores para que fiquem numa faixa estável antes de seguirem para a próxima camada — sem isso, à medida que um sinal atravessa dezenas de camadas, sua escala tende a crescer ou encolher de forma descontrolada, e o treino fica instável. LayerNorm faz isso subtraindo a média do vetor e dividindo pelo desvio padrão. RMSNorm simplifica esse cálculo: em vez de centralizar e depois escalar, ele só escala, dividindo cada valor pela raiz quadrada da média dos quadrados do vetor (`sqrt(mean(x²))`). Um passo a menos, mesma função, e — para o tamanho de modelo em que LLaMA/Mistral operam — qualidade final equivalente ou melhor. O código abaixo implementa exatamente essa operação:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
```

`.rsqrt()` calcula `1/sqrt(...)` diretamente; `self.eps` (um número bem pequeno, `1e-6`) evita divisão por zero caso o vetor seja todo zero; `self.weight` é um parâmetro aprendido que permite à rede reescalar o resultado se isso ajudar o treino. Teste isolado: `RMSNorm(8)(torch.randn(1, 5, 8)).shape` deve dar `torch.Size([1, 5, 8])` — a forma não muda, só a escala dos valores.

> Quer ver a comparação completa entre Batch Norm, Layer Norm e RMSNorm, com a intuição de por que normalizar estabiliza o treino? A versão Clássico cobre isso na seção "Normalização" do [Módulo 05 — Otimização e regularização para DL](/trilha-llm/v2/05_deep_learning#53-otimização-e-regularização-para-dl).

#### 2. SwiGLU no lugar de ReLU/GELU

Depois da attention, cada bloco Transformer processa cada token individualmente numa rede feed-forward (FFN): expande a dimensão do vetor, aplica uma não-linearidade, e comprime de volta. ReLU/GELU aplicam essa não-linearidade uma vez, numa única projeção. SwiGLU usa duas projeções lineares em paralelo (`w1` e `w3` abaixo) — uma passa por uma ativação chamada SiLU, a outra não — e multiplica os dois resultados elemento a elemento antes de projetar de volta (`w2`). Esse produto funciona como um portão: os valores de uma projeção controlam quanto da outra passa adiante. Para o mesmo número de parâmetros, isso costuma produzir um modelo de qualidade um pouco melhor do que ReLU/GELU simples — é por isso que virou padrão em LLaMA, Mistral e Qwen.

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`F.silu` é a função de ativação SiLU (`x * sigmoid(x)`); o restante é álgebra linear pura (as três `nn.Linear`). O resultado desta classe é o FFN completo de um bloco LLaMA-style — você o usa como parte do bloco Transformer montado no passo 5.

> A anatomia completa de um bloco Transformer (onde o FFN entra, por que vem depois da attention, o que muda entre pre-norm e post-norm) está no [Módulo 07 — Anatomia de um bloco Transformer](/trilha-llm/v2/07_transformers#75-anatomia-de-um-bloco-transformer) da versão Clássico.

#### 3. RoPE (aplicado dentro da attention, não somado ao embedding)

Sem alguma informação de posição, um Transformer trata a entrada como um conjunto de tokens, não uma sequência — "o cão mordeu o homem" e "o homem mordeu o cão" teriam a mesma representação, já que attention por si só não distingue ordem. O Transformer original resolvia isso somando um vetor de posição fixo ao embedding de cada token, antes da primeira camada. RoPE (Rotary Position Embedding) faz diferente: em vez de somar algo ao embedding, ele *rotaciona* os vetores de Query e Key dentro do próprio cálculo de attention, por um ângulo proporcional à posição do token. O efeito prático é que a relação calculada entre dois tokens passa a depender da distância relativa entre eles (quantas posições um está do outro), não da posição absoluta de cada um — o que generaliza melhor para sequências mais longas do que as vistas em treino. As duas funções abaixo implementam essa rotação:

```python
def build_rope_cache(seq_len, dim, base=10000, device="cpu"):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # (seq_len, dim/2)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    # x: (batch, n_heads, seq_len, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rot_even = x1 * cos - x2 * sin
    x_rot_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out
```

`build_rope_cache` calcula, para cada posição da sequência, um conjunto de ângulos de rotação (`cos`/`sin`) — um par de valores por posição e por par de dimensões do vetor. `apply_rope` usa esses ângulos para rotacionar cada par de coordenadas consecutivas do vetor (`x1`, `x2`), o que é a rotação em si. Você aplica `apply_rope` às Queries e às Keys logo antes do cálculo de attention — é isso que torna a distância entre tokens parte do próprio produto Q·K, sem precisar somar nada ao embedding.

> A dedução completa de por que essa rotação preserva a distância relativa entre tokens, e a comparação com outras formas de codificar posição (sinusoidal, aprendida, ALiBi), está no [Módulo 07 — Posicionamento](/trilha-llm/v2/07_transformers#73-posicionamento) da versão Clássico.

#### 4. GQA (Grouped-Query Attention), com RoPE embutido

Durante a geração de texto, o modelo mantém em memória as projeções de Key e Value já calculadas para todos os tokens gerados até agora (o "KV-cache"), para não recalculá-las a cada novo token. Em multi-head attention comum, cada cabeça de Query tem sua própria cabeça de Key e Value — quanto mais cabeças, maior esse cache. GQA reduz o cache fazendo várias cabeças de Query **compartilharem** a mesma cabeça de Key/Value: no código abaixo, `n_heads` cabeças de Query são divididas em `n_kv_heads` grupos, e cada grupo reutiliza um único par K/V (é o que `repeat_interleave` faz — replica cada cabeça K/V o número de vezes necessário para casar com as cabeças de Query do seu grupo). Menos K/V distintos guardados, cache menor, inferência mais barata em memória, com perda de qualidade pequena.

```python
class GQACausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
```

`is_causal=True` aplica automaticamente a máscara causal — cada token só "vê" tokens anteriores a ele, nunca o futuro, o que é o que torna a geração autoregressiva coerente (sem isso, o modelo "trapacearia" olhando a resposta durante o treino).

> A comparação entre GQA, Multi-Query Attention e Sliding Window Attention — os três ataques diferentes ao custo de attention em produção — está no [Módulo 07 — Eficiência de Attention](/trilha-llm/v2/07_transformers#77-eficiência-de-attention) da versão Clássico.

#### 5. Monte o bloco e o modelo completo

As quatro peças anteriores (RMSNorm, SwiGLU, RoPE, GQA) ainda não formam um modelo — faltam juntá-las na ordem certa (um bloco Transformer) e empilhar vários blocos. O código abaixo faz isso em duas classes: `LlamaStyleBlock` monta um único bloco (normaliza, aplica attention, soma de volta a entrada original — a "conexão residual" —, normaliza de novo, aplica o FFN, soma de novo), e `MiniLlama` empilha `n_layers` desses blocos entre uma camada de embedding (que transforma tokens em vetores) e uma camada de saída (que transforma vetores de volta em pontuações sobre o vocabulário).

```python
class LlamaStyleBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ff_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQACausalSelfAttention(dim, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, dim * ff_mult)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)  # residual: soma a entrada de volta
        x = x + self.ff(self.norm2(x))
        return x


class MiniLlama(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, n_kv_heads, n_layers, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [LlamaStyleBlock(dim, n_heads, n_kv_heads) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head_dim = dim // n_heads

    def forward(self, idx):
        B, T = idx.shape
        cos, sin = build_rope_cache(T, self.head_dim, device=idx.device)
        x = self.tok_emb(idx)  # (B, T) -> (B, T, dim): cada token vira um vetor aprendido
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        return self.head(x)  # (B, T, dim) -> (B, T, vocab_size): logits sobre o vocabulário
```

`nn.Embedding` é uma tabela de consulta: cada índice de token (um número inteiro) vira um vetor aprendido de tamanho `dim` — é a primeira coisa que acontece com um token antes de qualquer processamento pelo modelo. `self.head` faz o caminho inverso no final: transforma o vetor processado de volta numa pontuação (logit) pra cada palavra possível do vocabulário, e a maior pontuação é a previsão do modelo pro próximo token.

Teste antes de treinar:

```python
model = MiniLlama(vocab_size=1000, dim=128, n_heads=8, n_kv_heads=2, n_layers=4, max_len=256)
idx = torch.randint(0, 1000, (1, 32))
logits = model(idx)
print(logits.shape)  # torch.Size([1, 32, 1000])
```

Se a forma bater, as 4 peças estão encaixadas corretamente.

#### 6. Prepare os dados: Tiny Shakespeare

Um modelo aprende a partir de texto, mas uma rede neural só processa números — antes de treinar, você precisa de um corpus de texto e de uma forma de converter esse texto em uma sequência de inteiros (tokenização) e de volta. O código abaixo baixa um corpus pequeno (as peças de Shakespeare, ~1MB, um padrão para experimentos desse tipo) e constrói a forma mais simples possível de tokenizador: um mapeamento de cada caractere único do texto para um número inteiro.

```python
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt").read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join(itos[t] for t in tokens)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

Tokenizar "caractere a caractere" é a forma mais simples de tokenização: cada caractere único do texto vira um número inteiro (o vocabulário aqui é só o alfabeto e a pontuação usados no texto, tipicamente menos de 100 símbolos). É mais grosseiro que BPE/SentencePiece — os esquemas de tokenização usados em LLMs de produção, que agrupam sequências de caracteres frequentes num único token — mas suficiente para este experimento, e mais simples de implementar do zero.

#### 7. O loop de treino, explicado linha a linha

Com o modelo montado (passo 5) e os dados tokenizados (passo 6), falta o processo que ajusta os pesos do modelo repetidamente até que ele preveja bem o próximo caractere: o loop de treino. Cada iteração faz a mesma sequência de 5 operações — pegar um lote de dados, calcular a previsão do modelo, medir o erro dessa previsão, calcular quanto cada peso contribuiu para o erro, e ajustar os pesos — e repete isso milhares de vezes.

```python
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniLlama(vocab_size=vocab_size, dim=128, n_heads=8, n_kv_heads=2, n_layers=4, max_len=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

block_size = 128
batch_size = 32

for step in range(2000):
    x, y = get_batch(train_data, block_size, batch_size, device)

    logits = model(x)                                        # forward: previsão do modelo
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))  # o quão errada a previsão está

    optimizer.zero_grad()   # zera gradientes acumulados do passo anterior
    loss.backward()         # calcula o gradiente da loss em relação a cada peso
    optimizer.step()        # atualiza os pesos na direção que reduz a loss

    if step % 200 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
```

O que está acontecendo: `get_batch` pega pedaços aleatórios do texto — `x` é o pedaço, `y` é o mesmo pedaço deslocado 1 caractere pra frente (o que o modelo deveria prever a cada posição). `cross_entropy` mede o quão distante a distribuição de probabilidade prevista pelo modelo está do caractere certo — quanto menor, melhor. `loss.backward()` é o PyTorch calculando automaticamente o quanto cada peso do modelo contribuiu pro erro; `optimizer.step()` ajusta cada peso um pouquinho na direção que diminuiria esse erro. Repita isso milhares de vezes e o modelo aprende o padrão estatístico do texto.

Loss começando em torno de `ln(vocab_size)` (~4.2 pra ~65 caracteres) e caindo pra perto de 1.0-1.5 depois de 2000 passos é um sinal de que o treino está funcionando.

#### 8. Gere texto

Com o modelo treinado, gerar texto é um processo iterativo: dado um texto inicial, o modelo prevê uma distribuição de probabilidade sobre o próximo caractere, você sorteia um caractere dessa distribuição, anexa ao texto, e repete — usando agora o texto já estendido como nova entrada. `@torch.no_grad()` desliga o cálculo de gradientes durante a geração, já que aqui você só está usando o modelo, não treinando-o (isso deixa a geração mais rápida e usa menos memória).

```python
@torch.no_grad()
def generate(model, start_text, max_new_tokens, device):
    idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])       # olha só os últimos block_size tokens
        probs = F.softmax(logits[:, -1, :], dim=-1)  # distribuição de probabilidade do próximo token
        next_token = torch.multinomial(probs, num_samples=1)  # amostra um token dessa distribuição
        idx = torch.cat([idx, next_token], dim=1)
    return decode(idx[0].tolist())

print(generate(model, start_text="ROMEO:", max_new_tokens=200, device=device))
```

`torch.multinomial` amostra aleatoriamente um token proporcional à probabilidade que o modelo atribuiu a cada um (em vez de sempre pegar o mais provável) — é por isso que rodar `generate` duas vezes dá textos diferentes.

> Quer aprofundar em cada peça (por que RoPE extrapola melhor, como Flash Attention otimiza o `scaled_dot_product_attention` por baixo dos panos, o que muda entre pre-norm e post-norm)? A versão Clássico desta trilha cobre isso em detalhe: [Módulo 07 — Anatomia de um bloco Transformer](/trilha-llm/v2/07_transformers#75-anatomia-de-um-bloco-transformer).

**Checklist deste projeto**:
- [ ] RMSNorm, SwiGLU, RoPE e GQA implementados e testados isoladamente.
- [ ] `MiniLlama` com forward pass correto (forma `[batch, seq, vocab_size]`).
- [ ] Treino rodando, loss caindo de ~4.2 pra ~1.0-1.5 em Tiny Shakespeare.
- [ ] `generate()` produzindo texto que se parece (nem que vagamente) com português/inglês estruturado.

---

### Projeto 8.4 — Análise de scaling laws

Reusa o `MiniLlama` e o loop de treino do Projeto 8.3 — a novidade aqui é rodar o mesmo treino em 3 tamanhos diferentes e comparar.

**Pré-requisitos**: os mesmos do Projeto 8.3, mais `pip install matplotlib`.

**1. Transforme o treino do 8.3 numa função reutilizável**:

```python
def train_and_get_final_loss(dim, n_layers, n_heads, n_kv_heads, steps=1500):
    model = MiniLlama(vocab_size, dim, n_heads, n_kv_heads, n_layers, max_len=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(steps):
        x, y = get_batch(train_data, block_size=128, batch_size=32, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return n_params, loss.item()
```

`sum(p.numel() for p in model.parameters())` conta quantos parâmetros treináveis o modelo tem — `numel()` retorna o número de elementos de cada tensor de peso, e somamos todos.

**2. Rode para 3 tamanhos** (ajuste `dim`/`n_layers` até obter aproximadamente 1M, 5M e 25M parâmetros — imprima `n_params` e ajuste por tentativa):

```python
configs = [
    {"dim": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2},   # ~1M params
    {"dim": 64, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2},   # ~5M params
    {"dim": 128, "n_layers": 6, "n_heads": 8, "n_kv_heads": 2},  # ~25M params
]

results = []
for cfg in configs:
    n_params, final_loss = train_and_get_final_loss(**cfg)
    results.append((n_params, final_loss))
    print(f"{n_params:,} params -> loss final {final_loss:.4f}")
```

**3. Plote loss × parâmetros** — o que "plotar um gráfico" significa na prática: usar a biblioteca `matplotlib` pra desenhar um gráfico de linha/dispersão e salvar como imagem:

```python
import matplotlib.pyplot as plt

n_params_list = [r[0] for r in results]
losses = [r[1] for r in results]

plt.plot(n_params_list, losses, marker="o")
plt.xscale("log")          # escala log no eixo X: parâmetros variam em ordens de grandeza
plt.xlabel("Número de parâmetros")
plt.ylabel("Loss final")
plt.title("Loss × Parâmetros (scaling law empírica)")
plt.savefig("scaling_law.png")
plt.show()
```

`plt.plot(x, y)` desenha uma linha conectando os pontos `(x[i], y[i])`; `marker="o"` marca cada ponto real com um círculo; `plt.xscale("log")` usa escala logarítmica no eixo X porque os tamanhos de modelo variam em ordens de grandeza (1M, 5M, 25M) — numa escala linear, os pontos ficariam amontoados; `plt.savefig` salva a imagem em disco.

**4. Verifique o trade-off Chinchilla**: para cada configuração, calcule `tokens_processados = steps × batch_size × block_size` e divida pelo número de parâmetros — compare essa razão com os ~20 tokens/parâmetro da seção 8.2. Seus modelos provavelmente estão bem *acima* dessa razão (poucos parâmetros, muitos passos de treino no mesmo corpus pequeno) — isso é esperado num experimento de bolso, mas a tendência qualitativa (mais parâmetros → loss final menor, com retornos decrescentes) deve aparecer no seu gráfico.

---

### Projeto 8.5 — Embeddings e MTEB

**Pré-requisitos**: `pip install sentence-transformers`.

**O que é similaridade de cosseno, rapidamente**: dois vetores são "parecidos" quando apontam pra direção similar no espaço — o cosseno do ângulo entre eles mede exatamente isso, dando 1.0 quando os vetores são idênticos em direção, 0 quando são perpendiculares (sem relação), e valores negativos quando são opostos. Embeddings de texto são vetores onde textos com significado parecido têm alta similaridade de cosseno.

O código abaixo carrega 3 modelos de embedding, converte uma lista de frases-documento e uma frase-consulta em vetores, e usa a similaridade de cosseno entre a consulta e cada documento para descobrir qual documento cada modelo considera mais relevante:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

models = {
    "e5": SentenceTransformer("intfloat/multilingual-e5-large"),
    "bge": SentenceTransformer("BAAI/bge-m3"),
    "nomic": SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
}

docs = [
    "O gato dormiu no sofá a tarde toda.",
    "Um felino tirou uma soneca no sofá.",
    "A bolsa de valores caiu 2% hoje.",
]
query = "Onde o gato estava descansando?"

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for name, model in models.items():
    doc_embeddings = model.encode(docs)
    query_embedding = model.encode(query)
    sims = [cosine_sim(query_embedding, d) for d in doc_embeddings]
    best = int(np.argmax(sims))
    print(f"\n{name}: melhor match = '{docs[best]}' (sim={sims[best]:.3f})")
    for doc, sim in zip(docs, sims):
        print(f"  {sim:.3f}  {doc}")
```

Rode e confirme: a primeira e a segunda frase (sobre o gato) devem ter similaridade bem mais alta com a query do que a terceira (sobre bolsa de valores), em todos os 3 modelos — é isso que faz um embedding "bom": captura semântica, não só palavras em comum (repare que "gato"/"felino" e "sofá dormiu"/"soneca" não compartilham palavras exatas).

**Reproduza 1 tarefa do MTEB**: acesse o [leaderboard do MTEB](https://huggingface.co/spaces/mteb/leaderboard), escolha uma tarefa de classificação ou retrieval pequena, baixe o dataset de exemplo indicado na página da tarefa, e repita o mesmo padrão acima (embeddar, comparar por similaridade) nesse dataset — compare seus 3 modelos na mesma tarefa.

---

## Erros comuns

- **"Modelo X é melhor"** sem qualificar tarefa, idioma, contexto.
- **Ignorar licenças** — nem todo modelo "aberto" tem uso comercial permitido (LLaMA tem cláusulas; Gemma tem termos próprios).
- **Avaliar apenas em inglês** — muitos modelos colapsam em PT-BR.
- **Confiar em leaderboards de benchmark** sem entender saturação e contaminação.
- **Achar que "MoE = mais caro"** — em compute por token ativo, MoE pode ser mais barato.
- **Esquecer `optimizer.zero_grad()`** no loop de treino do Projeto 8.3 — gradientes acumulam entre passos se você não zerar, e o treino diverge de forma confusa.

---

## Checklist de saída

- [ ] Sei explicar a diferença entre BERT, GPT e T5 sem hesitação (se não, revise a seção 8.1 — em particular a intuição do revisor/contador de histórias/tradutor).
- [ ] Conheço pelo menos 5 famílias open de 2024–2025 e suas diferenças (se não, revise a tabela da seção 8.7).
- [ ] Entendo o que Chinchilla corrigiu em relação ao trabalho de Kaplan, e reproduzi o trade-off parâmetros/dados empiricamente no Projeto 8.4 (se não, revise a seção 8.2 e o passo 4 do Projeto 8.4).
- [ ] Implementei mini-LLaMA com RMSNorm + RoPE + SwiGLU + GQA, do zero, e treinei até gerar texto coerente (se não, revise o Projeto 8.3 — cada peça tem um teste isolado antes do treino completo, comece por eles).
- [ ] Sei consultar e interpretar leaderboards (MTEB, Open LLM, Chatbot Arena) (se não, revise a seção 8.7 e o Projeto 8.5).
- [ ] Sei explicar o que cada termo de "8.4 Detalhes arquiteturais modernos" faz e por que existe, não só o nome (se não, revise a seção 8.4 — RMSNorm, SwiGLU, RoPE e GQA têm, além disso, a explicação repetida no ponto de código correspondente no Projeto 8.3).
