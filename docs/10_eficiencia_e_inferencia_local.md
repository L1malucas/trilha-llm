---
id: 10_eficiencia_e_inferencia_local
title: "Módulo 10 — Eficiência e Inferência Local"
sidebar_position: 3
---

# Módulo 10 — Eficiência e Inferência Local

> **Objetivo**: rodar e otimizar LLMs em hardware modesto (laptop, GPU consumer, CPU, edge). Quantização, distilação, KV-cache, servidores de inferência, edge inference.
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[09](09_treinamento_e_alinhamento.mdx). Este é o primeiro módulo com um projeto em TypeScript/Node — se você nunca escreveu nada em Node, o Projeto 10.2 ensina o mínimo necessário antes de pedir para você escrever código.
>
> **Tempo de referência**: 3–5 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Calcular quanta VRAM um modelo precisa em qualquer quantização, e explicar por que Q4 não destrói a qualidade.
- Explicar por que KV-cache é indispensável em geração autoregressiva — o que aconteceria sem ele.
- Explicar como speculative decoding acelera geração sem mudar a distribuição de saída do modelo grande.
- Explicar por que um modelo destilado pode aprender mais do "soft label" do teacher do que de rótulos binários.
- Escolher os parâmetros de sampling (temperature, top-k, top-p) certos para uma tarefa dada, com justificativa.

---

## Por que isso importa

Um projeto que depende inteiramente de uma API paga de LLM tem um custo por requisição, uma dependência de rede, e nenhuma garantia de que os dados enviados fiquem privados. Este módulo cobre a alternativa: rodar modelos localmente, decidir conscientemente o trade-off entre custo, latência e qualidade, e operar offline quando a rede ou a privacidade exigem isso (dado médico, jurídico, ou interno de empresa). Nem todo hardware disponível é uma GPU de datacenter — os projetos deste módulo são desenhados para rodar em GPUs de consumo, CPU, e até no navegador.

---

## 10.1 Quantização

### Conceito
Reduzir precisão dos pesos (e/ou ativações): FP32 → FP16 → BF16 → INT8 → INT4. Menos memória, mais velocidade, perda controlada de qualidade.

### Tipos
- **Post-Training Quantization (PTQ)** — sem retreino, mais rápido.
- **Quantization-Aware Training (QAT)** — com fine-tuning, melhor qualidade.

### Formatos importantes
- **GGUF** — formato do `llama.cpp` (o motor de inferência que roda por baixo do Ollama, que você já usa desde o Projeto 8.1). Suporta níveis de quantização como Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 — o número indica os bits por peso, e a letra o esquema específico de agrupamento usado.
- **GPTQ** — quantização que usa informação de segunda ordem (derivadas) para decidir como comprimir cada peso com o menor erro possível. Boa para GPU.
- **AWQ (Activation-aware Weight Quantization)** — usa estatísticas de ativação (quais pesos realmente importam na prática, observando ativações reais) para proteger os pesos mais influentes durante a quantização.
- **bitsandbytes** — a biblioteca que você já usou no Projeto 9.1 para carregar um modelo em 4-bit (`BitsAndBytesConfig`) é justamente uma implementação de quantização PTQ para PyTorch.
- **FP8** — quantização usada já durante o treino (não só inferência), em GPUs recentes (H100, MI300).

> **Intuição**: por que reduzir de 32 bits para 4 bits por peso não destrói o modelo? Pesos de uma rede treinada não usam a precisão total do float32 de forma uniformemente importante — a maior parte da informação relevante está na *magnitude relativa* dos pesos (este peso é bem maior que aquele), não nos últimos bits de precisão decimal. Quantização mapeia a faixa de valores de cada peso para um número menor de "níveis" discretos (16 níveis em INT4) escolhidos para minimizar o erro — e a rede, sendo robusta a ruído por natureza (ela já foi treinada com dropout, batch noise etc. — mod. [05](05_deep_learning.md#53-otimização-e-regularização-para-dl)), tolera bem essa perda controlada. AWQ e GPTQ melhoram sobre quantização ingênua ao proteger especificamente os pesos que mais impactam a saída, em vez de quantizar tudo com o mesmo cuidado.

### Regra de bolso
Para um modelo de **N** bilhões de parâmetros, VRAM aproximada para inferência:
- FP16 / BF16: 2N GB
- INT8: N GB
- Q4 (GGUF Q4_K_M): ~0.6N GB
- Q3: ~0.45N GB
(Adicione ~1–2 GB para KV-cache e overhead.)

Exemplo: LLaMA-3 8B em Q4_K_M cabe folgado em GPU de 8 GB.

> **Exemplo resolvido**: para um modelo de 13B — em FP16, `13 × 2 = 26 GB`. Em Q4_K_M, usando a regra de bolso acima: `13 × 0.6 ≈ 7.8 GB` — cabe numa GPU de consumo de 8-12GB. Essa é a mesma conta de `sum(p.numel() for p in model.parameters())` do Projeto 8.4 (contar parâmetros), só convertida para bytes de memória em vez de contagem — é essa conta, feita antes de qualquer outra decisão, que determina se um modelo roda na sua máquina local ou precisa de cloud.

### Trade-offs
- **Q4_K_M** é o "sweet spot" comum: ~99% da qualidade do FP16 em muitos casos.
- **Q3** já degrada perceptivelmente em modelos pequenos.
- **Q8** é praticamente lossless mas dobra o footprint vs Q4.

> **Checkpoint**: sem olhar o texto, explique por que um modelo tolera bem a perda de precisão de FP16 para Q4, mas degrada visivelmente em Q3 ou Q2. Depois, calcule a VRAM aproximada (Q4_K_M) de um modelo de 30B parâmetros.

---

## 10.2 KV-Cache e otimizações de inferência

### KV-Cache
Em geração autoregressiva, K e V de tokens passados não mudam. Cachear evita recomputar.

### Otimizações modernas
- **PagedAttention** (vLLM) — gerência de KV-cache inspirada em memória virtual de sistemas operacionais: em vez de reservar um bloco contínuo de memória para o KV-cache de cada requisição (desperdiçando espaço quando a geração termina antes do limite reservado), aloca em blocos menores, sob demanda, e reaproveita blocos liberados por requisições que já terminaram.
- **Continuous Batching** — agrupa requisições em diferentes estágios de geração no mesmo lote, em vez de esperar um lote inteiro terminar antes de começar o próximo. Padrão em vLLM e TGI.
- **Speculative Decoding** — modelo pequeno propõe tokens, modelo grande valida (detalhado abaixo, e implementado no Projeto 10.6).
- **Flash Decoding** — a mesma ideia de Flash Attention (mod. [07](07_transformers.mdx#77-eficiência-de-attention), quando você chegar lá), adaptada para a fase de geração token a token.
- **Prompt Caching** — cachear o KV-cache de prefixos comuns entre chamadas diferentes (por exemplo, o mesmo system prompt reaparecendo em toda requisição de uma aplicação).

> **Intuição — KV-Cache**: gerar cada novo token exige calcular attention contra *todos* os tokens anteriores — o mesmo mecanismo que você implementou em `GQACausalSelfAttention` no Projeto 8.3. Sem cache, gerar o token N exigiria refazer o cálculo de K e V para os N-1 tokens anteriores *do zero*, toda vez — desperdício enorme, já que esses valores não mudam token a token (só dependem dos tokens já processados, que são fixos). O KV-cache guarda os K e V já calculados e só computa os do token novo a cada passo — é a diferença entre custo linear e custo quadrático repetido por token gerado. (O `generate()` que você escreveu no Projeto 8.3 recalcula tudo a cada passo, sem cache — funciona para um experimento pequeno, mas é exatamente o tipo de ineficiência que essas otimizações resolvem em produção.)
>
> **Intuição — Speculative Decoding**: um modelo pequeno e rápido "chuta" vários tokens seguintes de uma vez; o modelo grande então verifica *em paralelo* (não sequencialmente) se aceitaria cada um desses tokens — aceitar é barato (um forward pass grande, verificando vários tokens de uma vez), rejeitar custa só voltar ao ponto de divergência. Quando o modelo pequeno acerta a maioria dos tokens (comum em texto previsível), o resultado final tem a *mesma distribuição* que rodar o modelo grande sozinho — a técnica não aproxima nem degrada qualidade, só explora que verificar é mais barato que gerar sequencialmente token a token. Você implementa essa mecânica manualmente no Projeto 10.6.
>
> **Checkpoint**: sem olhar o texto, explique por que gerar sem KV-cache seria muito mais lento — o que teria que ser recalculado a cada novo token? Depois, explique por que speculative decoding não muda a qualidade da saída, só a velocidade.

---

## 10.3 Servidores de inferência

### Locais (uma máquina)
Você já usa o Ollama desde o Projeto 8.1 — ele é a opção de UX mais simples, rodando sobre o `llama.cpp` (o backend em C++ que efetivamente executa o modelo, e que você usa diretamente no Projeto 10.1). LM Studio, GPT4All e Jan são alternativas com interface gráfica desktop sobre o mesmo tipo de backend.

### Servidor (alta performance, multi-usuário)
- **vLLM** — o padrão de fato para servir LLMs em GPU com muitos usuários simultâneos, com PagedAttention e continuous batching nativos. Usado no Projeto 10.3.
- **TGI (Text Generation Inference)** — o servidor de inferência da Hugging Face, com objetivo semelhante ao vLLM.
- **TensorRT-LLM** (NVIDIA) — performance máxima, mas específico para hardware NVIDIA e mais complexo de configurar.
- **SGLang** — alternativa mais recente, com um runtime próprio otimizado para padrões de uso como agentes (várias chamadas encadeadas reaproveitando contexto).
- **MLC-LLM** — compila modelos para rodar em múltiplos backends (CUDA, Metal, Vulkan, WebGPU) a partir de uma única definição.

### API compatível com OpenAI
A maioria dos servidores acima — incluindo o próprio Ollama — expõe uma API compatível com o formato usado pela OpenAI (`/v1/chat/completions`). Isso significa que o mesmo código cliente (a mesma chamada HTTP, com o mesmo formato de corpo JSON) funciona apontando para Ollama local, para vLLM, ou para a API paga da OpenAI — só a URL base muda. Você aproveita isso diretamente no Projeto 10.2.

---

## 10.4 Inferência em CPU

Sim, é possível. Útil quando:
- Não há GPU.
- Custo de cloud > tempo de inferência.
- Requisições raras.

O `llama.cpp` (com otimizações OpenBLAS, Accelerate ou OpenMP dependendo da plataforma), o ONNX Runtime com provider de CPU, e o Intel OpenVINO são as opções mais usadas. Na prática: um modelo de 7B em Q4 num CPU moderno gera algo entre 5 e 20 tokens por segundo — lento comparado a uma GPU, mas suficiente para um chatbot interno de baixo volume ou processamento em lote sem pressa.

---

## 10.5 Inferência em edge

### Mobile / Browser
- **transformers.js** — roda modelos via ONNX Runtime diretamente no navegador ou em Node, sem servidor. Usado no Projeto 10.5.
- **MLC-LLM Web (web-llm)** — modelos rodando via WebGPU, direto no navegador.
- **Apple MLX** — framework da Apple otimizado para Apple Silicon.
- **MediaPipe LLM** (Google) — inferência local em apps mobile.

### Apple Silicon
Macs com chip M1/M2/M3 têm memória unificada (CPU e GPU compartilham a mesma memória física), o que os torna surpreendentemente competentes para rodar LLMs localmente — o `llama.cpp` tem um backend Metal otimizado especificamente para isso, e modelos de até 70B em Q4 rodam em um Mac Studio com 64GB+ de memória.

### Considerações TS
Para inferência client-side (rodando dentro do navegador do usuário, sem chamada a servidor) numa aplicação web: `transformers.js` é a opção mais madura, ONNX é o formato de modelo mais comum nesse contexto, e o tamanho do modelo importa de um jeito que não importa em backend — um modelo de mais de 1GB pesa visivelmente o carregamento inicial da página.

---

## 10.6 Distillation (destilação)

### Conceito
Treinar modelo pequeno (student) para imitar saídas de modelo grande (teacher). Resultado: 5–10× menor com 80–95% da qualidade.

### Tipos
- **Logit distillation** (clássica): o student é treinado para imitar a distribuição de probabilidade completa do teacher (não só a resposta mais provável), usando KL-divergence entre as duas distribuições como loss.
- **Sequence distillation**: o student aprende diretamente a partir do texto gerado pelo teacher, como dado de treino — é o método que você usa no Projeto 10.4, essencialmente o mesmo SFT do Projeto 9.1, só que com dados gerados por um modelo em vez de escritos por humanos.
- **Feature distillation**: alinhar representações intermediárias (as ativações de camadas do meio da rede, não só a saída final) entre teacher e student.

Modelos conhecidos por terem sido construídos assim incluem o DistilBERT (40% menor que BERT, ~97% da performance), o TinyLlama, e o Phi-3, cuja filosofia de treino é justamente usar um dataset sintético de alta qualidade gerado por um modelo maior.

> **Intuição — "dark knowledge"**: um rótulo binário ("isto é um gato") carrega 1 bit de informação. A distribuição de probabilidade completa que o teacher produz ("87% gato, 10% raposa, 2% cachorro, ...") carrega muito mais — ela expressa *o quanto* o teacher "acha" que a entrada se parece com cada classe, não só a resposta final. Treinar o student para imitar essa distribuição inteira (logit distillation, via KL-divergence — mod. [01](01_matematica.md#13-probabilidade-e-estatística)) transfere esse conhecimento mais rico, chamado informalmente de "dark knowledge", em vez de só o rótulo final que um dataset rotulado convencional daria. No Projeto 10.4 você usa a versão mais simples (sequence distillation, sem acesso aos logits do teacher, só ao texto gerado) porque, ao usar um teacher via API, você não tem acesso às probabilidades internas dele — só ao texto de saída.
>
> **Checkpoint**: sem olhar o texto, explique por que a distribuição de probabilidade completa do teacher carrega mais informação que só o rótulo mais provável.

### Quando vale a pena
- Você tem teacher (próprio ou via API barata).
- Tarefa específica (classificação, extração, conversão de formato).
- Precisa de inferência rápida em produção.

---

## 10.7 Pruning, Sparsity e Mixture of Experts

### Pruning
Remover pesos pouco importantes. Pode ser:
- **Estruturado** (cabeças/camadas inteiras) — ganho real em hardware.
- **Não-estruturado** (pesos individuais) — ganho real exige hardware sparse-aware.

### Modelos esparsos
- **NVIDIA 2:4 sparsity** — em A100/H100, pesa menos com pouca perda.

### MoE em inferência
Inferência de Mixtral 8×7B usa apenas ~13B "ativos" por token, mesmo tendo 47B totais. Vantagem para inferência; desvantagem em VRAM total necessária.

> A intuição de MoE (por que mais parâmetros totais não significa proporcionalmente mais custo por token) já foi construída no mod. [08](08_llms_arquiteturas.md#84-detalhes-arquiteturais-modernos-20242025) — aqui a nuance extra é que, mesmo com poucos parâmetros *ativos* por token, o modelo inteiro (todos os experts) ainda precisa estar carregado na VRAM, já que o roteador pode escolher qualquer expert a cada token. MoE economiza *compute*, não necessariamente *memória*.

---

## 10.8 Sampling e parâmetros de geração

### Métodos
- **Greedy** — sempre o token mais provável. Determinístico. Repetitivo.
- **Beam Search** — mantém top-k hipóteses. Bom para tradução, ruim para geração aberta.
- **Temperature sampling** — softmax(logits / T). T < 1 = mais focado, T > 1 = mais aleatório.
- **Top-k** — amostra só dos k tokens mais prováveis.
- **Top-p (nucleus)** — amostra do menor conjunto cuja prob acumulada ≥ p.
- **Min-p** — variante moderna, mais robusta.
- **Repetition penalty**, **frequency penalty**, **presence penalty**.

> **Intuição — temperature**: dividir os logits por `T` antes do softmax (mod. [01](01_matematica.md#13-probabilidade-e-estatística)) achata ou afia a distribuição de probabilidade. `T < 1` amplifica a diferença entre o token mais provável e os demais (a distribuição fica mais "pontiaguda", saída mais determinística e repetitiva); `T > 1` achata a distribuição (mais tokens ficam com chance razoável, saída mais criativa/aleatória, mas também mais propensa a erros). `T = 0` equivale a greedy (sempre o mais provável). Você já usou `T=1` implicitamente no `generate()` do Projeto 8.3 (`F.softmax(logits[:, -1, :], dim=-1)` sem dividir por nada equivale a `T=1`). Top-k e top-p resolvem um problema diferente de temperature: mesmo com temperature moderada, a "cauda" da distribuição (milhares de tokens improváveis, mas não zero) pode ocasionalmente ser amostrada, gerando texto bizarro — top-k corta essa cauda mantendo só os k tokens mais prováveis; top-p corta dinamicamente (mantém tokens até a probabilidade acumulada atingir p), o que se adapta melhor a distribuições ora muito concentradas, ora muito espalhadas.
>
> **Aplicação real**: geração de código ou extração estruturada tipicamente usa temperature baixa (0-0.3, precisão importa mais que criatividade); brainstorming ou escrita criativa usa temperature mais alta (0.7-1.0). É por isso que a maioria das APIs de LLM expõe `temperature` como o parâmetro mais visível ao usuário — é o dial mais direto entre "confiável e previsível" e "criativo e variado".
>
> **Checkpoint**: sem olhar o texto, explique o que acontece com a distribuição de probabilidade quando `T > 1` vs `T < 1`. Depois, explique a diferença entre top-k e top-p.

### Constrained generation
Grammars (GBNF, usado pelo `llama.cpp`), a biblioteca `Outlines`, e o modo "JSON Schema" oferecido por algumas APIs forçam a saída do modelo a respeitar uma estrutura definida (por exemplo, um schema JSON), manipulando diretamente quais tokens são permitidos a cada passo de geração — em vez de só pedir no prompt para "responder em JSON" e torcer para o modelo obedecer. Você compara as duas abordagens no Projeto 10.7.

---

## Projetos práticos

### Projeto 10.1 — Quantizar e comparar

Você vai comparar o mesmo modelo em diferentes níveis de quantização, medindo latência, tamanho em disco e qualidade subjetiva — reaproveitando o Ollama e o padrão de chamada via `requests` que você já usa desde o Projeto 8.1. O Ollama já baixa modelos em GGUF (o formato do `llama.cpp`) e disponibiliza várias tags de quantização prontas para o mesmo modelo, sem que você precise compilar nada.

**Pré-requisitos**: Ollama (já instalado no Projeto 8.1), Python com `pip install requests`.

**1. Baixe o mesmo modelo em 4 níveis de quantização diferentes**:

```bash
ollama pull qwen2.5:7b-instruct-q3_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q5_K_M
ollama pull qwen2.5:7b-instruct-q8_0
```

**2. Compare latência e tamanho em disco**: o script abaixo reaproveita a função `query_ollama` do Projeto 8.1 para rodar os mesmos 20 prompts em cada quantização, e usa o comando `ollama list` (via `subprocess`) para ler o tamanho em disco de cada modelo:

```python
import requests
import subprocess
import time
import json

MODELS = [
    "qwen2.5:7b-instruct-q3_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q5_K_M",
    "qwen2.5:7b-instruct-q8_0",
]
PROMPTS = [
    "Explique o que é overfitting em uma frase.",
    "Escreva uma função Python que verifica se uma string é um palíndromo.",
    # complete até 20 prompts, cobrindo geração, raciocínio, código e PT-BR
]

def query_ollama(model, prompt):
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    elapsed = time.time() - start
    return response.json()["response"], elapsed

def model_size_gb(model):
    output = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    for line in output.splitlines():
        if line.startswith(model.split(":")[0]) and model.split(":")[1] in line:
            return line.split()[2], line.split()[3]  # tamanho, unidade (GB/MB)
    return None, None

results = []
for model in MODELS:
    size, unit = model_size_gb(model)
    for prompt in PROMPTS:
        text, elapsed = query_ollama(model, prompt)
        results.append({"model": model, "size": f"{size}{unit}", "prompt": prompt, "response": text, "latency_s": round(elapsed, 2)})
        print(f"{model:30s} {size}{unit:3s} {elapsed:5.2f}s  {prompt[:40]}")

with open("quantizacao.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

**3. Documente o "ponto de quebra"**: abra `quantizacao.json` e, para os mesmos prompts, compare as respostas da Q8_0 (praticamente sem perda) contra Q5, Q4 e Q3 — normalmente a diferença entre Q8/Q5/Q4 é sutil, mas Q3 costuma introduzir erros perceptíveis (repetição, incoerência ocasional) em modelos de 7B. Anote em qual nível de quantização você começou a notar diferença real, não só teórica.

> Use exatamente os mesmos 20 prompts em todas as quantizações, na mesma ordem — a única variável mudando deve ser o nível de quantização, senão a comparação mistura ruído de amostragem com o efeito real da quantização.

---

### Projeto 10.2 — Servidor local com Ollama + cliente em TypeScript

Você vai escrever um cliente em TypeScript/Node que conversa com o Ollama (já rodando localmente desde o Projeto 8.1) com streaming — a resposta aparece token a token, como num chat de verdade, em vez de tudo de uma vez no final.

**Pré-requisitos**: [Node.js](https://nodejs.org/) 20+ instalado, Ollama rodando localmente.

**Se você nunca escreveu nada em Node/TypeScript**: um projeto Node começa com um arquivo `package.json` (descreve dependências e scripts) e roda arquivos `.ts` diretamente com uma ferramenta como `tsx` (sem precisar compilar manualmente para JavaScript primeiro).

**1. Crie o projeto**:

```bash
mkdir ollama-client && cd ollama-client
npm init -y
npm install ai ollama-ai-provider
npm install -D tsx typescript @types/node
```

`ai` é o Vercel AI SDK — uma biblioteca que padroniza a forma de chamar diferentes provedores de LLM (OpenAI, Anthropic, modelos locais) com a mesma interface, incluindo suporte a streaming pronto. `ollama-ai-provider` é o adaptador que conecta o AI SDK à API local do Ollama.

**2. Escreva o cliente com streaming** (`chat.ts`):

```typescript
import { ollama } from "ollama-ai-provider";
import { streamText } from "ai";

async function main() {
  const { textStream } = streamText({
    model: ollama("qwen2.5:7b"),
    prompt: "Explique o que é KV-cache para alguém que já sabe o que é attention.",
  });

  for await (const chunk of textStream) {
    process.stdout.write(chunk);
  }
}

main();
```

`streamText` inicia a geração e retorna `textStream`, um async iterator — o `for await` consome cada pedaço de texto (`chunk`) assim que o modelo o gera, em vez de esperar a resposta inteira. `process.stdout.write` (em vez de `console.log`) escreve sem quebra de linha automática, dando o efeito visual de texto "digitando" no terminal.

**3. Rode**: `npx tsx chat.ts`. Compare a experiência com o `requests.post(..., stream=False)` que você usou em Python nos projetos anteriores — ali, você esperava a resposta inteira antes de ver qualquer coisa; aqui, o texto aparece progressivamente, que é como praticamente toda interface de chat de LLM em produção se comporta.

---

### Projeto 10.3 — vLLM em GPU + benchmark de throughput

Você vai subir um servidor vLLM e medir como o throughput (respostas processadas por segundo) muda com o número de requisições concorrentes — isso é o que PagedAttention e continuous batching (seção 10.2) otimizam na prática.

**Pré-requisitos**: uma GPU (uma instância cloud com GPU, ou Colab com GPU habilitada), `pip install vllm`.

**1. Suba o servidor vLLM** (linha de comando, expõe API compatível OpenAI na porta 8000):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

**2. Escreva o benchmark de concorrência** — o código abaixo dispara N requisições *ao mesmo tempo* (não uma de cada vez) e mede o tempo total, o que revela o ganho do batching contínuo do vLLM (que processa múltiplas requisições no mesmo lote de GPU):

```python
import asyncio
import aiohttp
import time

async def send_request(session, prompt):
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100},
    ) as response:
        return await response.json()

async def run_benchmark(n_concurrent, prompt="Explique o que é machine learning em 3 frases."):
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [send_request(session, prompt) for _ in range(n_concurrent)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start
    print(f"{n_concurrent:3d} requisições concorrentes: {elapsed:.2f}s total, {n_concurrent/elapsed:.2f} req/s")

for n in [1, 4, 16, 64]:
    asyncio.run(run_benchmark(n))
```

`asyncio.gather(*tasks)` dispara todas as `n_concurrent` requisições de uma vez e espera todas terminarem — diferente de um loop `for` sequencial (que esperaria uma resposta antes de enviar a próxima), aqui todas competem pela mesma GPU ao mesmo tempo, e é isso que expõe o ganho do batching contínuo: `req/s` deve crescer com a concorrência, não cair proporcionalmente, porque o vLLM processa várias requisições no mesmo passo de GPU em vez de uma por vez.

**3. Compare com o Ollama**: rode o mesmo benchmark (adaptando a URL e o payload para o formato do endpoint `/api/generate` do Ollama) contra o Ollama servindo o mesmo modelo, e compare o `req/s` em concorrência 16 e 64 — Ollama, otimizado para uso individual, tende a degradar mais com concorrência alta do que o vLLM, que foi desenhado especificamente para servir muitos usuários ao mesmo tempo.

---

### Projeto 10.4 — Distilar para tarefa específica

Você vai usar um modelo grande, via Ollama, como "professor" para gerar dados de treino rotulados para uma tarefa específica (extração de dados estruturados), e treinar um modelo pequeno para fazer a mesma tarefa sozinho, mais rápido — reaproveitando diretamente o SFT+LoRA do Projeto 9.1.

**Pré-requisitos**: os mesmos do Projeto 9.1 (transformers, peft, trl), mais Ollama.

**1. Gere o dataset de treino usando o teacher** — o padrão é o mesmo `requests.post` ao Ollama de sempre, mas agora o prompt pede uma saída estruturada:

```python
import requests
import json
import random

def ask_ollama(model, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
    )
    return response.json()["response"]

produtos_exemplo = [
    "Notebook Dell Inspiron 15, 16GB RAM, SSD 512GB, por R$ 3.499,00, cor prata",
    "Tênis Nike Air Max 90, tamanho 42, branco e preto, R$ 599,90",
    # gere ou colete ~200 descrições de produto variadas
]

dataset = []
for descricao in produtos_exemplo:
    prompt = (
        f"Extraia um JSON com os campos nome, preco (número), e atributos (lista de strings) "
        f"a partir desta descrição de produto: {descricao}"
    )
    saida_json = ask_ollama("qwen2.5:7b", prompt)
    dataset.append({"prompt": descricao, "completion": saida_json})

with open("distillation_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

`"format": "json"` instrui o Ollama a restringir a saída a JSON válido — uma forma simples de constrained generation (seção 10.8, aprofundada no Projeto 10.7), útil aqui justamente porque você precisa que os dados de treino do student sejam JSON parseável, sem exceção.

**2. Treine o student com o mesmo `SFTTrainer` do Projeto 9.1**, agora usando `distillation_dataset.jsonl` no lugar do Alpaca — o código de treino é idêntico ao passo 3-4 do Projeto 9.1 (formatar como conversa com `apply_chat_template`, configurar `SFTConfig`/`SFTTrainer`, treinar), só o dataset de origem muda.

**3. Compare qualidade e latência**: rode o mesmo conjunto de descrições de produto (um subconjunto separado, não usado no treino) no teacher (via Ollama) e no student (seu modelo treinado), meça o tempo de resposta de cada um, e verifique se o JSON gerado pelo student é válido (`json.loads` sem exceção) e razoavelmente próximo do gerado pelo teacher. O ganho esperado: o student, sendo muito menor, responde bem mais rápido, à custa de alguma qualidade — quanto essa troca vale depende da tarefa.

---

### Projeto 10.5 — LLM no browser

Você vai rodar um modelo pequeno inteiramente no navegador, sem nenhum servidor por trás, usando `transformers.js`.

**Pré-requisitos**: Node.js (mesmo do Projeto 10.2), navegador moderno.

**1. Crie um projeto web mínimo**:

```bash
mkdir llm-browser && cd llm-browser
npm init -y
npm install @xenova/transformers
```

**2. Escreva um classificador de sentimento client-side** (`index.html` com um `<script type="module">`, ou um arquivo `main.js` importado por um bundler simples como Vite):

```javascript
import { pipeline } from "@xenova/transformers";

async function run() {
  const t0 = performance.now();
  const classifier = await pipeline("sentiment-analysis");
  const t1 = performance.now();
  console.log(`Tempo de carregamento do modelo: ${(t1 - t0).toFixed(0)}ms`);

  const t2 = performance.now();
  const resultado = await classifier("This tutorial is surprisingly clear.");
  const t3 = performance.now();
  console.log(`Tempo até o resultado: ${(t3 - t2).toFixed(0)}ms`);
  console.log(resultado);
}

run();
```

`pipeline("sentiment-analysis")` baixa (na primeira vez, ficando em cache do navegador depois) e carrega um modelo pequeno já preparado para classificação de sentimento, convertido para ONNX — o formato que roda eficientemente no navegador via WebAssembly ou WebGPU, dependendo do que o navegador suporta. `performance.now()` é a forma padrão do JavaScript de medir tempo com precisão de milissegundos.

**3. Rode num servidor local simples** (navegadores bloqueiam módulos ES carregados via `file://` por segurança): `npx serve .` (ou `python -m http.server`) e abra a página no navegador, com o console de desenvolvedor aberto para ver os tempos impressos.

---

### Projeto 10.6 — Speculative decoding manual

Você vai implementar, à mão, o mecanismo de speculative decoding descrito na seção 10.2: um modelo pequeno (draft) propõe vários tokens, um modelo maior (target) verifica todos de uma vez, e você confirma que a saída final é idêntica à do modelo grande sozinho.

**Pré-requisitos**: os mesmos do Projeto 9.1 (`transformers`, `torch`).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")

@torch.no_grad()
def speculative_generate(prompt, max_new_tokens=50, k=4):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = 0

    while generated < max_new_tokens:
        # 1. o modelo draft propõe k tokens, gerando um de cada vez
        draft_ids = input_ids.clone()
        for _ in range(k):
            logits = draft_model(draft_ids).logits[:, -1, :]
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            draft_ids = torch.cat([draft_ids, next_token], dim=1)
        proposed = draft_ids[:, input_ids.shape[1]:]

        # 2. o modelo target verifica todos os k tokens propostos numa única passada
        target_logits = target_model(draft_ids).logits[:, input_ids.shape[1] - 1:-1, :]
        target_probs = torch.softmax(target_logits, dim=-1)

        # 3. aceita tokens propostos enquanto o target concordar (amostrando da distribuição dele)
        accepted = 0
        for i in range(k):
            token_id = proposed[0, i].item()
            if torch.rand(1).item() < target_probs[0, i, token_id].item():
                accepted += 1
            else:
                break

        if accepted > 0:
            input_ids = torch.cat([input_ids, proposed[:, :accepted]], dim=1)
            generated += accepted

        # 4. se algum token foi rejeitado, o target gera o token correto naquela posição
        if accepted < k:
            correction_logits = target_logits[:, accepted, :]
            correction = torch.multinomial(torch.softmax(correction_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, correction], dim=1)
            generated += 1

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

print(speculative_generate("A capital da França é"))
```

O passo 1 é o modelo pequeno gerando normalmente (o mesmo loop token-a-token do `generate()` do Projeto 8.3). O passo 2 é o que torna a técnica rápida: em vez de o modelo grande gerar token a token, ele processa a sequência inteira proposta pelo draft **numa única passada** (`target_model(draft_ids)`), obtendo a distribuição de probabilidade que ele *teria* usado em cada uma daquelas posições — isso é possível porque, com a sequência já completa, calcular os logits em todas as posições simultaneamente custa aproximadamente o mesmo que calcular numa posição só (a maior parte do custo é o mesmo forward pass). O passo 3 decide, posição por posição, se aceita o token do draft (usando a probabilidade que o *target* atribuiria a ele — não simplesmente "o target concorda ou não", mas uma checagem probabilisticamente correta que garante que a distribuição final seja idêntica à do target sozinho). O passo 4 corrige a partir do primeiro ponto de divergência.

**Confirme a corretude antes de medir velocidade**: rode `speculative_generate` e, separadamente, gere com `target_model.generate(...)` sozinho, usando a mesma seed (`torch.manual_seed(0)` antes de cada chamada) — a saída deve ser idêntica ou estatisticamente equivalente. Só depois disso meça e compare o tempo de execução das duas abordagens; um ganho de velocidade sem essa confirmação de corretude não prova que a implementação está certa.

---

### Projeto 10.7 — Constrained generation

Você vai comparar duas formas de obter JSON estruturado de um LLM: pedir educadamente no prompt, versus forçar a estrutura com `outlines`.

**Pré-requisitos**: `pip install outlines`.

**1. Defina o schema esperado e gere com `outlines`** — a biblioteca manipula diretamente quais tokens são permitidos a cada passo de geração, então o resultado é *garantidamente* JSON válido no formato pedido, não apenas provável:

```python
import outlines
from pydantic import BaseModel

class Produto(BaseModel):
    nome: str
    preco: float
    em_estoque: bool

model = outlines.models.transformers("Qwen/Qwen2.5-0.5B-Instruct")
gerador = outlines.generate.json(model, Produto)

resultado = gerador("Notebook Dell Inspiron, R$ 3499.00, disponível para compra imediata")
print(resultado)  # já é uma instância de Produto, não uma string para fazer parse
```

`Produto` é um schema Pydantic — a mesma forma de descrever estrutura de dados que você usaria para validação de API. `outlines.generate.json` usa esse schema para restringir, token a token, quais continuações são sintaticamente possíveis (por exemplo, depois de `{"nome": "`, só tokens que continuam uma string ou a fecham são permitidos) — é uma versão automatizada e mais geral do `"format": "json"` que você já usou no Ollama no Projeto 10.4.

**2. Compare com a abordagem de prompt** — rode o mesmo teste 20 vezes pedindo "responda em JSON" num prompt comum (sem `outlines`), e conte quantas dessas 20 respostas falham em `json.loads` ou não seguem o schema exato:

```python
import json

def gerar_com_prompt_comum(descricao):
    prompt = f"Extraia nome, preco (número) e em_estoque (true/false) em JSON: {descricao}"
    # reaproveite aqui o generate_response do Projeto 9.1 ou uma chamada ao Ollama
    return resposta_do_modelo(prompt)

falhas = 0
for _ in range(20):
    saida = gerar_com_prompt_comum("Notebook Dell Inspiron, R$ 3499.00, disponível para compra imediata")
    try:
        json.loads(saida)
    except json.JSONDecodeError:
        falhas += 1

print(f"{falhas}/20 respostas não eram JSON válido")
```

O resultado esperado: `outlines` acerta 20/20 por construção (é estruturalmente impossível gerar algo fora do schema), enquanto a abordagem por prompt tem uma taxa de falha maior que zero — variando com o modelo, mas raramente é 0/20 de forma consistente. Essa diferença é exatamente o motivo de constrained generation existir: confiabilidade estrutural, não só uma instrução mais bem escrita.

---

## Erros comuns

- **Quantizar excessivamente** modelos pequenos — degradação grande. Q4 em 70B ≠ Q4 em 1B.
- **Esquecer KV-cache** ao implementar inferência manual — fica 100× mais lento (é por isso que o `generate()` do Projeto 8.3, sem cache, é aceitável só em escala de experimento).
- **Comparar latências** sem warmup — primeiros tokens incluem inicialização.
- **Não controlar `seed` e `temperature`** ao avaliar qualidade — comparações são ruído.
- **Achar que CPU = lento sempre** — modelos pequenos quantizados em CPU moderna são úteis.
- **Misturar contextos** em servidor sem prompt cache adequado.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Servidor local | RAG (mod. [12](12_rag.mdx)), Agentes (mod. [13](13_agentes_tools_protocolos.md)), Produção (mod. [15](15_engenharia_producao.mdx)) |
| Quantização | Deploy mobile (mod. [18](18_multimodal.mdx)), edge |
| Sampling | Prompt engineering (mod. [11](11_prompt_engineering.md)), avaliação (mod. [14](14_avaliacao_e_seguranca.md)) |
| Constrained generation | Tools/structured output (mod. [13](13_agentes_tools_protocolos.md)) |

---

## Checklist de saída

- [ ] Tenho um modelo open rodando localmente, sem API externa (se não, revise o Projeto 8.1 e a seção 10.3).
- [ ] Sei quantizar um modelo e comparei níveis diferentes na prática (se não, revise o Projeto 10.1).
- [ ] Implementei um cliente que consome um servidor local com streaming (se não, revise o Projeto 10.2).
- [ ] Sei usar vLLM (ou entendo por que ele escala melhor que Ollama sob concorrência) (se não, revise o Projeto 10.3 e a seção 10.2).
- [ ] Domino parâmetros de sampling e sei quando usar cada um (se não, revise a seção 10.8).
- [ ] Fiz inferência client-side no navegador e medi o tempo de carregamento (se não, revise o Projeto 10.5).
- [ ] Implementei speculative decoding e confirmei que a saída é idêntica à do modelo grande sozinho (se não, revise o Projeto 10.6 e a seção 10.2).
- [ ] Entendo trade-offs de quantização, distillation e MoE conceitualmente (se não, revise as seções 10.1, 10.6 e 10.7).
