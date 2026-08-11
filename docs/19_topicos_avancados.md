---
id: 19_topicos_avancados
title: "Módulo 19 — Tópicos Avançados"
sidebar_position: 12
---

# Módulo 19 — Tópicos Avançados

> **Objetivo**: cobrir frentes de pesquisa ativas — Mixture of Experts, State Space Models (Mamba), modelos de difusão, world models, neuro-symbolic, interpretability profundo, e fronteiras emergentes.
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[18](18_multimodal.mdx). Conforto com leitura de papers — este módulo é o primeiro que pede isso explicitamente, porque o objetivo aqui não é só implementar, é chegar a ler pesquisa de fronteira sem depender de resumo de terceiros.
>
> **Tempo de referência**: 6–10 semanas (não-linear; escolha sub-tópicos).

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar por que SSMs (Mamba) escalam linearmente onde attention escala quadraticamente, e o que se perde nessa troca.
- Explicar o processo de difusão (forward de ruído, reverse de denoising) com intuição, não só o nome.
- Discutir com fluência pelo menos 3 destes tópicos o suficiente para ler o paper original sem depender de blog explicativo.
- Manter ceticismo calibrado: distinguir o que já é usado em produção do que ainda é pesquisa especulativa.

Este módulo é não-linear — escolha sub-tópicos por interesse, não precisa ler tudo em sequência. Os projetos práticos seguem a mesma lógica: a recomendação é escolher 2-3, não os 8.

---

## Por que isso importa

Aqui você sai de "engenheiro que aplica" para "engenheiro que acompanha pesquisa". Nenhum desses tópicos é obrigatório para construir produtos hoje, mas várias dessas frentes (MoE, reasoning RL, model merging) já saíram do papel e viraram prática comum em menos de dois anos — a fronteira de pesquisa de hoje tende a virar o "óbvio" de amanhã, e ler papers com fluência é o que permite perceber essa transição cedo.

---

## 19.1 Mixture of Experts (MoE) em profundidade

### Conceito
Em vez de um FFN denso, ter N FFNs (experts). Um **router** escolhe top-k experts por token. Compute por token cresce sublinearmente em parâmetros totais — o mesmo mecanismo que você já usou em prosa desde o mod. 08, agora implementado do zero no Projeto 19.1.

### Desafios
- **Load balancing** entre experts (auxiliary loss, detalhado abaixo).
- **Comunicação** em treinamento distribuído (all-to-all, mod. 09).
- **Memória total** alta para inferência — todos os experts precisam estar carregados, mesmo que só alguns sejam usados por token (mod. 10).
- **Fine-tuning** mais delicado.

### Inferência eficiente de MoE
Modelos como Mixtral 8×7B usam ~13B parâmetros ativos por token, mas precisam dos 47B totais em VRAM — otimizações como expert offloading (mover experts pouco usados para CPU/disco) e expert caching atacam esse descompasso entre compute ativo e memória total.

> A intuição central de MoE (roteador escolhendo especialistas, mais parâmetros totais sem custo proporcional por token) já foi construída no mod. [08](08_llms_arquiteturas.md#84-detalhes-arquiteturais-modernos-20242025). Aqui, o desafio novo é **load balancing**: sem incentivo explícito, o router tende a colapsar, favorecendo sempre os mesmos poucos experts (que ficam bem treinados) e ignorando os demais (que ficam subtreinados, criando um ciclo vicioso) — a auxiliary loss penaliza justamente essa distribuição desigual, empurrando o router a usar todos os experts de forma mais balanceada ao longo do treino.

---

## 19.2 State Space Models (SSMs) e Mamba

### Motivação
Transformers têm complexidade O(n²) em attention (mod. 07). RNNs são O(n) mas têm gradientes problemáticos em sequências longas (mod. 05). SSMs prometem o melhor dos dois mundos.

> **Intuição**: SSMs herdam a estrutura matemática de sistemas de controle clássicos — mantêm um "estado" comprimido de tamanho fixo que resume tudo relevante da sequência vista até agora, atualizado a cada passo por uma transformação (aprendida). Isso é estruturalmente parecido com o hidden state de uma RNN — e herda o benefício de custo linear (processar um passo a mais custa um incremento fixo, não uma comparação com todos os passos anteriores como em attention). A inovação do Mamba é tornar essa atualização de estado **seletiva**: em vez de uma transformação fixa igual pra todo input (como SSMs anteriores, S4), os parâmetros da atualização dependem do próprio input atual — o modelo pode "decidir" dinamicamente o que vale a pena reter no estado e o que descartar, mitigando o gargalo clássico de RNN (informação distante sendo progressivamente diluída) sem pagar o custo quadrático de attention completa.
>
> **Aplicação real**: o trade-off real é que SSMs processam sequencialmente durante *geração* (como RNN), mas o cálculo de treino pode ser paralelizado de forma eficiente (diferente de RNN clássica) — é essa combinação que torna Mamba competitivo com Transformers em benchmarks de linguagem, especialmente em contextos muito longos, onde o custo quadrático de attention se torna proibitivo (mod. [07](07_transformers.mdx#77-eficiência-de-attention)). Modelos híbridos (Jamba) apostam que combinar camadas de attention (melhores em recuperar informação específica e distante) com camadas Mamba (mais baratas, boas em resumir contexto) pode ser melhor que qualquer um dos dois puro. Você mede essa diferença de escala diretamente no Projeto 19.2.
>
> **Checkpoint**: sem olhar o texto, explique por que Mamba escala linearmente onde attention escala quadraticamente. Depois, explique o que "seletividade" adiciona sobre SSMs anteriores como S4.

### Por que importa
- **Linear scaling** em comprimento.
- Memória constante na geração.
- Competitivos com Transformers em escalas pequenas-médias.
- Experimentos híbridos (Jamba, Zamba, Hymba) combinando Mamba com attention.

### Outras arquiteturas alternativas
RWKV (híbrido RNN-Transformer), RetNet (Microsoft) e Hyena são outras tentativas de escapar do custo quadrático de attention por caminhos distintos; Liquid Foundation Models (LFMs) exploram uma formulação inspirada em redes neurais líquidas (dinâmica contínua, não discreta).

---

## 19.3 Modelos de difusão

### Para imagens
A revolução pós-GAN. Aprendem a "des-ruidar" passo a passo — a técnica por trás de Stable Diffusion, DALL-E 3, Midjourney e a maioria dos geradores de imagem modernos.

> **Intuição**: o treino de um modelo de difusão tem duas metades. **Forward** (fixo, sem aprendizado): pegue uma imagem real e adicione ruído gaussiano gradualmente, passo a passo, até virar ruído puro — um processo simples e conhecido matematicamente. **Reverse** (o que se aprende): treine uma rede para prever, dado uma imagem ruidosa num passo qualquer, "o que foi adicionado" — ou seja, a rede aprende a **reverter** um passo de ruído por vez. Uma vez treinada, gerar uma imagem nova é começar de ruído puro (aleatório) e aplicar repetidamente o denoiser aprendido, passo a passo, até emergir uma imagem coerente — é literalmente esculpir uma imagem a partir de estática, removendo ruído iterativamente na direção que o modelo aprendeu ser "mais provável de gerar imagens reais". Você implementa exatamente esse processo, forward e reverse, do zero no Projeto 19.3.
>
> Classifier-free guidance é o mecanismo que permite controlar a geração por texto: durante o treino, o modelo aprende tanto a versão condicionada (com prompt) quanto a não-condicionada (sem prompt) da mesma tarefa; na geração, extrapolar na direção "condicionado menos não-condicionado" amplifica a influência do prompt, produzindo imagens mais fiéis à descrição (à custa de menos diversidade).
>
> **Checkpoint**: sem olhar o texto, explique o processo forward e reverse de um modelo de difusão com suas próprias palavras — o que exatamente a rede neural aprende a fazer?

### Modelos atuais (open)
Stable Diffusion 3/3.5, FLUX.1 (Black Forest Labs — estado da arte open), HunyuanDiT (Tencent), Sana (NVIDIA, otimizado para eficiência) e Lumina-Next são as opções abertas mais citadas atualmente; você usa um deles com LoRA no Projeto 19.4.

### Difusão para texto?
Pesquisa ativa, ainda longe de competir com geração autoregressiva em qualidade — a natureza discreta do texto (tokens, não valores contínuos como pixels) torna a formulação de "ruído gaussiano" original menos natural, e as variantes propostas (discrete diffusion) ainda são uma área em desenvolvimento.

### Difusão para vídeo
O mod. [18](18_multimodal.mdx) cobre — a mesma ideia central, com condicionamento temporal adicional entre frames.

### Aplicações além de imagem
Música (Stable Audio), áudio (AudioLDM), moléculas (drug discovery) e robótica (diffusion policy, onde a "imagem" gerada é uma sequência de ações de um robô) mostram que o princípio de difusão generaliza bem além de pixels.

---

## 19.4 World models e simuladores neurais

### Conceito
Aprender um modelo do mundo — `P(s' | s, a)`, a mesma notação de transição do MDP no mod. [17](17_aprendizado_reforco.md#171-formalismo-markov-decision-process-mdp) — com uma rede neural, e usar esse modelo para planejamento, exploração, ou avaliação contrafactual ("o que aconteceria se eu fizesse X?").

GameNGen é o exemplo mais vívido dessa ideia levada ao extremo: um modelo de difusão que aprendeu a simular o jogo DOOM jogável quadro a quadro, sem nenhum motor de jogo real por trás — cada frame seguinte é *gerado*, não computado por lógica de jogo explícita. Genie (DeepMind) generaliza isso para gerar ambientes jogáveis inteiros a partir de imagens.

> Mesmo conceito de model-based RL do mod. [17](17_aprendizado_reforco.md#175-model-based-rl-e-planning), levado ao extremo: em vez de um modelo simples do ambiente, treinar um simulador neural completo.

---

## 19.5 Long context e infinite context

### Técnicas de extensão
Position interpolation e NTK-aware scaling foram as primeiras tentativas de estender o contexto de um modelo além do que ele viu no treino; YaRN e LongRoPE (já mencionados no mod. 08) refinam essa ideia. StreamingLLM mantém uma janela deslizante de contexto recente, mas preserva alguns "sink tokens" iniciais fixos, que empiricamente acumulam atenção desproporcional e ajudam a estabilizar a geração mesmo com contexto efetivamente ilimitado.

### Compressão de contexto
AutoCompressor e GIST tokens tentam comprimir um contexto longo num resumo denso de poucos tokens, sem perder informação crítica — uma alternativa a RAG (mod. 12) para o mesmo problema de "não cabe tudo no contexto".

### Memória externa
MemGPT (mod. 13), Infini-attention (Google) e Titans (Google, mais recente) exploram formas de dar a um LLM acesso a memória que persiste além da janela de contexto imediata, sem precisar reprocessar tudo a cada chamada.

### Trade-off "Lost in the Middle"
> O trade-off "lost in the middle" e a comparação long context vs RAG já foram discutidos com intuição no mod. [12](12_rag.mdx#127-geração-com-contexto). As técnicas aqui (YaRN, LongRoPE) são extensões diretas do RoPE do Projeto 8.3 — ajustam matematicamente como a rotação de posição se comporta além do comprimento visto no treino, permitindo extrapolar sem retreinar do zero. Você mede esse efeito empiricamente no Projeto 19.8.

---

## 19.6 Reasoning models a fundo

O mod. [09](09_treinamento_e_alinhamento.mdx) introduziu reasoning RL; aqui aprofundamos.

### Linhas de pesquisa
- **CoT supervisionado** (treina diretamente em traces de raciocínio já escritos).
- **Process Reward Models (PRM)** — recompensa em passos intermediários, não só na resposta final (detalhado abaixo). Implementado no Projeto 19.5.
- **Outcome-only RL** (R1-style, o que você já implementou nos Projetos 9.5 e 17.5).
- **Search augmented** — busca em árvore (parecida com MCTS, Projeto 17.6) durante a própria geração, explorando múltiplos caminhos de raciocínio.
- **Test-time compute scaling** — gastar mais compute na inferência (gerar mais, ou pensar mais) em vez de treinar um modelo maior, como formalizado num paper de 2024 que mostrou que, para o mesmo orçamento total de compute, às vezes vale mais investir em inferência do que em treino.

### Implicações
- Modelos pequenos + reasoning RL podem bater modelos grandes em tarefas verificáveis.
- Reasoning emergente é parcialmente entendido — o "porquê" exato de cadeias de raciocínio longas emergirem de treino outcome-only continua sendo uma área ativa de investigação, não um mecanismo totalmente mapeado.

> Process Reward Models vs Outcome-only RL é uma escolha de *onde* colocar o sinal de recompensa: PRM recompensa cada passo intermediário do raciocínio (exige anotação mais granular, mas dá sinal mais denso); outcome-only (GRPO/R1-style, mod. [09](09_treinamento_e_alinhamento.mdx#96-reasoning-rl-estilo-r1)) recompensa só a resposta final, deixando o modelo "descobrir" que padrões de raciocínio levam a acertos — mais simples de implementar, mas exige mais exploração para o sinal esparso se propagar até os passos intermediários.

---

## 19.7 Neuro-symbolic e híbridos

Combinar redes neurais com raciocínio simbólico/lógico — abordagens como DeepProbLog e Logical Neural Networks tentam unir a capacidade de generalização de redes neurais com as garantias formais de sistemas baseados em regras lógicas explícitas. O Neuro-Symbolic Concept Learner (MIT) aplica isso a aprendizado visual estruturado. Tool-augmented LLMs (mod. 13) são, em certo sentido, neuro-symbolic na prática: o LLM (neural) decide *quando* invocar uma calculadora ou um solver lógico (simbólico), combinando os dois paradigmas sem fundi-los numa única arquitetura. Sistemas como AlphaProof e AlphaGeometry (DeepMind) usam LLMs combinados com provadores de teorema formais para resolver problemas de matemática de competição.

---

## 19.8 Interpretability profundo

O mod. [14](14_avaliacao_e_seguranca.md) introduziu mechanistic interpretability. Aqui, frentes ativas:

- **Sparse Autoencoders (SAEs)** — a técnica que você já usou de forma introdutória no Projeto 14.6, aqui aprofundada; a Anthropic publicou resultados extraindo milhões de features interpretáveis de modelos de produção (Claude) usando essa técnica em escala.
- **Activation steering** — manipular comportamento via vetores no espaço de ativação, detalhado abaixo.
- **Circuit-level interpretability** — descobrir "circuitos" (como os induction heads do Projeto 14.6) para tarefas específicas.
- **Concept-based explanations**.

### Por que isso importa
Interpretability é ferramenta para safety (detectar capacidades latentes antes de se manifestarem em comportamento observável), debugging (entender por que um modelo específico errou, não só que errou), e edição de modelo (corrigir um comportamento específico sem o custo de um retreino completo — técnicas como ROME e MEMIT editam associações factuais específicas dentro dos pesos, diretamente).

> Activation steering estende diretamente a ideia de "features como direções no espaço de ativação" (mod. [14](14_avaliacao_e_seguranca.md#1411-mechanistic-interpretability)) de diagnóstico passivo para intervenção ativa: se uma direção corresponde a um conceito, somar (ou subtrair) essa direção às ativações durante a geração pode amplificar (ou suprimir) esse conceito no comportamento do modelo, sem retreinar nada — uma forma de "editar" comportamento diretamente na representação interna. Você explora isso na prática no Projeto 19.6.

---

## 19.9 Federated learning e privacy-preserving ML

Federated learning treina um modelo sem centralizar os dados — cada dispositivo/organização treina localmente e só compartilha atualizações de peso (não os dados brutos), agregadas depois num modelo global. Differential privacy adiciona ruído calibrado matematicamente para garantir que nenhum exemplo individual do dataset possa ser identificado a partir do modelo treinado. Homomorphic encryption e secure multi-party computation permitem, em teoria, computar sobre dados criptografados sem nunca descriptografá-los — computacionalmente caras, mas ativamente pesquisadas para casos de altíssima sensibilidade (saúde, finanças). Confidential computing (Intel SGX, AMD SEV) oferece uma alternativa baseada em hardware: um enclave isolado onde nem o operador da máquina consegue inspecionar o que está rodando dentro.

---

## 19.10 Continual learning

Aprender sequencialmente sem esquecer (catastrophic forgetting, já discutido com intuição no mod. [09](09_treinamento_e_alinhamento.mdx#97-continued-pretraining-cpt), no contexto de mistura de dados como mitigação).

### Métodos
Elastic Weight Consolidation (EWC) ataca o mesmo problema de forma diferente: identifica quais pesos foram mais importantes para tarefas anteriores (via uma aproximação da curvatura da loss em torno deles) e penaliza mudanças grandes nesses pesos específicos durante o treino da tarefa nova — uma forma de regularização direcionada, em vez de misturar dados. Métodos baseados em replay (misturar exemplos antigos de volta no treino) e progressive networks (adicionar capacidade nova sem sobrescrever a antiga) são abordagens complementares; adapters (o mesmo princípio de LoRA, mod. 09) também se encaixam aqui — cada tarefa nova ganha seu próprio adaptador pequeno, sem tocar nos pesos base compartilhados.

### Conexão com LLMs
Continued pretraining (mod. 09) é uma instância prática de continual learning. Model merging (TIES, DARE, MoE-merge — combinar dois ou mais modelos fine-tunados diretamente nos pesos, sem retreino) é uma abordagem alternativa e surpreendentemente eficaz, explorada com a ferramenta `mergekit` no Projeto 19.7.

---

## 19.11 Geometric Deep Learning, GNNs

Para dados estruturados como grafos: redes sociais, moléculas, recommender systems. GNNs (Graph Neural Networks) — nas variantes GCN, GraphSAGE, GAT (com attention, um parente distante do attention de Transformers, aqui aplicado entre nós vizinhos de um grafo em vez de posições de uma sequência), MPNN e Graph Transformer — processam dados onde a estrutura de conexão em si carrega informação essencial, algo que uma rede densa ou convolucional tradicional não captura naturalmente. AlphaFold 2 e 3 (previsão de estrutura de proteínas) são, no fundo, aplicações sofisticadas dessa família de ideias combinadas com attention.

---

## 19.12 AI for Science

AlphaFold 2/3 (estrutura de proteínas) é talvez o exemplo mais citado de impacto de ML fora da tecnologia tradicional; GNoME acelera a descoberta de novos materiais; NeuralGCM aplica redes neurais a modelagem climática; Equiformer e SchNet aplicam geometric deep learning (seção 19.11) a química quântica, respeitando simetrias físicas (como invariância a rotação) diretamente na arquitetura.

---

## 19.13 Frontier safety e alignment

Tópicos de pesquisa ativa e ainda sem resposta definitiva incluem: sandbagging detection (detectar se um modelo está deliberadamente performando abaixo de sua capacidade real durante avaliação), deceptive alignment (a possibilidade teórica de um modelo parecer alinhado durante treino/avaliação mas não sê-lo de fato), power-seeking (tendências emergentes de buscar mais recursos/controle como sub-objetivo instrumental), scalable oversight (como avaliar um sistema mais capaz que o avaliador humano — debate entre modelos e recursive reward modeling são duas propostas), e weak-to-strong generalization (se um modelo fraco consegue supervisionar efetivamente o treino de um modelo mais forte). Institutos como o AISI (Reino Unido e EUA) conduzem avaliações formais de risco nessa linha.

---

## Projetos práticos (escolha 2–3)

### Projeto 19.1 — MoE pequeno do zero

Você vai modificar o `MiniLlama` do Projeto 8.3, substituindo o FFN denso (`SwiGLU`) por uma camada MoE com roteamento top-2 e auxiliary loss de load balancing.

**Pré-requisitos**: o código do Projeto 8.3.

```python
class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        logits_router = self.router(x_flat)  # (B*T, n_experts)
        pesos_top_k, indices_top_k = logits_router.topk(self.top_k, dim=-1)
        pesos_top_k = torch.softmax(pesos_top_k, dim=-1)

        saida = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = indices_top_k[:, i]
            peso = pesos_top_k[:, i].unsqueeze(-1)
            for e in range(self.n_experts):
                mascara = expert_idx == e
                if mascara.any():
                    saida[mascara] += peso[mascara] * self.experts[e](x_flat[mascara])

        # auxiliary loss: incentiva o router a distribuir tokens uniformemente entre experts
        prob_media_por_expert = torch.softmax(logits_router, dim=-1).mean(dim=0)
        aux_loss = self.n_experts * (prob_media_por_expert ** 2).sum()

        return saida.view(B, T, D), aux_loss
```

`self.router` é uma camada linear simples que produz, para cada token, um score por expert; `topk` seleciona os `top_k=2` experts de maior score, e `softmax` normaliza esses dois scores para que somem 1 (o peso de cada expert na combinação final). O loop `for e in range(self.n_experts)` roteia cada token só para os experts que o selecionaram (via `mascara`) — ineficiente nesta implementação didática (um MoE de produção usa kernels especializados para isso), mas deixa claro exatamente qual token vai para qual expert. `aux_loss` é a auxiliary loss da seção 19.1: se `prob_media_por_expert` fosse perfeitamente uniforme (`1/n_experts` para todos), a soma dos quadrados seria mínima; quando o router colapsa para poucos experts favoritos, alguns valores de `prob_media_por_expert` ficam altos e a penalidade cresce. Some `aux_loss` (com um peso pequeno, tipicamente 0.01) à loss principal de `cross_entropy` do Projeto 8.3 durante o treino.

Treine essa versão MoE do `MiniLlama` (trocando `SwiGLU` por `MoEFeedForward` dentro de `LlamaStyleBlock`) no mesmo Tiny Shakespeare do Projeto 8.3, e verifique, ao final, se os tokens se distribuem razoavelmente entre os experts (não todos concentrados em 1-2) — imprima `prob_media_por_expert` periodicamente durante o treino para acompanhar isso.

---

### Projeto 19.2 — Mamba pequeno from scratch

Você vai treinar um modelo de linguagem pequeno baseado em Mamba no mesmo Tiny Shakespeare do Projeto 8.3, e comparar o crescimento do tempo de inferência com o do `MiniLlama`.

**Pré-requisitos**: `pip install mamba-ssm`.

```python
from mamba_ssm import Mamba
import torch.nn as nn

class MiniMamba(nn.Module):
    def __init__(self, vocab_size, dim, n_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(RMSNorm(dim), Mamba(d_model=dim))  # RMSNorm do Projeto 8.3, reaproveitado
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for bloco in self.blocks:
            x = x + bloco(x)  # a mesma conexão residual do LlamaStyleBlock
        return self.head(self.norm_f(x))
```

`Mamba(d_model=dim)` já encapsula o mecanismo de state space seletivo (a parte matematicamente mais densa do paper original) — reaproveitada aqui como um bloco pronto, do mesmo jeito que você reaproveitaria `nn.MultiheadAttention` em vez de reescrever attention do zero, se não fosse o objetivo didático do Projeto 8.3. Note que a estrutura externa (embedding → blocos com residual → norma final → head) é idêntica ao `MiniLlama` — só o mecanismo de mistura de informação entre posições muda (Mamba em vez de `GQACausalSelfAttention`).

Treine com o mesmo loop do Projeto 8.3 (`get_batch`, `AdamW`, `cross_entropy`) sobre Tiny Shakespeare. **Meça o tempo de geração** conforme a sequência cresce (gere 100, depois 500, depois 1000 tokens, cronometrando cada geração com `time.time()`) para o `MiniMamba` e para o `MiniLlama` do Projeto 8.3 (sem KV-cache, como implementado ali) — o tempo do Mamba deve crescer aproximadamente linearmente com o comprimento, enquanto o do Transformer sem cache cresce mais rápido que linear, evidenciando concretamente a diferença de escala discutida na seção 19.2.

---

### Projeto 19.3 — DDPM no MNIST

Você vai implementar um modelo de difusão completo (forward, reverse, sampling) do zero, e visualizar o processo de geração.

**Pré-requisitos**: `pip install torchvision`.

**1. Processo forward** (adicionar ruído progressivamente — fixo, sem parâmetros aprendidos):

```python
import torch
import torch.nn as nn

T = 1000  # número de passos de ruído
betas = torch.linspace(1e-4, 0.02, T)  # quanto ruído é adicionado em cada passo
alphas = 1 - betas
alphas_cumulativos = torch.cumprod(alphas, dim=0)

def adicionar_ruido(x0, t):
    ruido = torch.randn_like(x0)
    alpha_cum = alphas_cumulativos[t].view(-1, 1, 1, 1)
    x_ruidoso = alpha_cum.sqrt() * x0 + (1 - alpha_cum).sqrt() * ruido
    return x_ruidoso, ruido
```

`adicionar_ruido` implementa a fórmula fechada do processo forward: em vez de aplicar ruído 1000 vezes em sequência, uma propriedade matemática do processo (a soma de gaussianas ainda é gaussiana) permite pular direto para o estado no passo `t` numa única operação. **Antes de treinar qualquer coisa**, visualize `adicionar_ruido(x0, t)` para `t` crescente (0, 250, 500, 750, 999) numa imagem de exemplo do MNIST — confirme visualmente que a imagem vira ruído puro gradualmente, antes de depurar a parte aprendida.

**2. Rede de denoising** (a parte que aprende — dado uma imagem ruidosa e o passo `t`, prever o ruído que foi adicionado):

```python
class DenoiserSimples(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tempo = nn.Embedding(T, 32)
        self.rede = nn.Sequential(
            nn.Conv2d(1 + 32 // 32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, x_ruidoso, t):
        emb_t = self.embed_tempo(t).view(-1, 32, 1, 1).expand(-1, 32, 28, 28).mean(dim=1, keepdim=True)
        entrada = torch.cat([x_ruidoso, emb_t], dim=1)
        return self.rede(entrada)
```

Um denoiser real usa uma arquitetura UNet (com downsampling/upsampling e skip connections); esta versão simplificada usa só convoluções diretas, suficiente para MNIST (28×28, um único canal) e mais fácil de entender linha a linha. `embed_tempo` dá à rede informação sobre "em qual passo do processo de ruído estou" — sem isso, a rede não teria como saber se deve remover muito ruído (passos iniciais do reverse) ou pouco (passos finais).

**3. Loop de treino** (a loss é simplesmente o erro entre o ruído previsto e o ruído real adicionado):

```python
denoiser = DenoiserSimples()
optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)

for epoca in range(10):
    for x0, _ in dataloader_mnist:  # DataLoader padrão do torchvision sobre MNIST
        t = torch.randint(0, T, (x0.shape[0],))
        x_ruidoso, ruido_real = adicionar_ruido(x0, t)
        ruido_previsto = denoiser(x_ruidoso, t)
        loss = nn.functional.mse_loss(ruido_previsto, ruido_real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**4. Sampling** (gerar uma imagem nova, partindo de ruído puro e revertendo passo a passo):

```python
@torch.no_grad()
def gerar_imagem(denoiser, n_passos_visualizar=5):
    x = torch.randn(1, 1, 28, 28)
    trajetoria = []
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t])
        ruido_previsto = denoiser(x, t_tensor)
        alpha, alpha_cum, beta = alphas[t], alphas_cumulativos[t], betas[t]
        x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cum).sqrt()) * ruido_previsto)
        if t > 0:
            x += beta.sqrt() * torch.randn_like(x)
        if t % (T // n_passos_visualizar) == 0:
            trajetoria.append(x.clone())
    return x, trajetoria
```

Esse loop é a reversão passo a passo descrita na Intuição da seção 19.3: em cada passo `t`, o denoiser prevê o ruído presente, e a fórmula remove uma fração desse ruído (proporcional a `beta`/`alpha` daquele passo especificamente), com um pouco de ruído novo adicionado de volta (exceto no último passo) para manter a variabilidade estatística do processo. `trajetoria` guarda alguns pontos intermediários — visualize-os em sequência para ver a imagem emergir gradualmente do ruído.

---

### Projeto 19.4 — Fine-tune de Stable Diffusion com LoRA

Você vai fazer fine-tuning de um modelo de difusão para gerar imagens num estilo ou sujeito específico, usando LoRA (o mesmo princípio do mod. 09, aplicado a um modelo de difusão em vez de um LLM).

**Pré-requisitos**: `pip install diffusers peft`, 20-50 imagens próprias (do mesmo estilo/sujeito).

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

lora_config = LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # as projeções de attention dentro do UNet do denoiser
)
pipe.unet.add_adapter(lora_config)

# o loop de treino segue o mesmo padrão do Projeto 19.3 (prever ruído, mse_loss, backward, step),
# mas usando pipe.unet como denoiser e otimizando só os parâmetros LoRA adicionados
optimizer = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=1e-4)
```

`target_modules` aqui mira as projeções de Q/K/V da attention *dentro do UNet* do modelo de difusão — não do LLM, mas o princípio de LoRA (matrizes de baixo rank sobre camadas específicas) é idêntico ao do Projeto 9.1. O restante do loop de treino (amostrar um timestep `t`, adicionar ruído, prever, calcular `mse_loss`, `backward`, `step`) é literalmente o mesmo do Projeto 19.3, com `pipe.unet` no lugar de `DenoiserSimples` e `pipe.vae` codificando suas imagens de treino para o espaço latente antes de adicionar ruído (Stable Diffusion faz difusão no espaço latente comprimido, não em pixels diretamente — mais eficiente, mesmo princípio).

**Compare duas abordagens**: LoRA-style (como acima, adaptando só pesos de attention) vs DreamBooth-style (fine-tuning mais completo do UNet, associando uma palavra-chave rara ao seu sujeito específico) — DreamBooth tende a capturar o sujeito com mais fidelidade, mas exige mais memória e é mais propenso a "esquecer" a diversidade do modelo original (o mesmo catastrophic forgetting do mod. 09), enquanto LoRA é mais leve e mais fácil de combinar com outros LoRAs depois.

---

### Projeto 19.5 — Reasoning RL com Process Reward Model

Você vai estender o GRPO outcome-only dos Projetos 9.5/17.5 treinando um Process Reward Model que avalia cada passo intermediário do raciocínio, não só a resposta final.

**Pré-requisitos**: os mesmos do Projeto 9.5, mais um dataset pequeno de traces de raciocínio anotados passo a passo (você mesmo anota: para ~100 problemas, marque cada passo intermediário da solução como correto/incorreto).

**1. Treine o PRM** como um classificador que recebe (problema, passo-a-passo-até-aqui) e prevê se o último passo está correto:

```python
from transformers import AutoModelForSequenceClassification

prm = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B", num_labels=2)

def formatar_exemplo_prm(problema, passos_ate_aqui, rotulo):
    texto = f"Problema: {problema}\nPassos: {' -> '.join(passos_ate_aqui)}"
    return {"text": texto, "label": rotulo}  # rotulo: 1 = passo correto, 0 = incorreto

# treine com SFTTrainer/Trainer padrão sobre esse dataset formatado — mesma mecânica do Projeto 17.4
```

**2. Use o PRM como recompensa, passo a passo, em vez de só no final**:

```python
def recompensa_prm(problema, trace_completo):
    passos = trace_completo.split("\n")
    recompensa_total = 0
    for i in range(len(passos)):
        score = prm(**formatar_exemplo_prm(problema, passos[:i+1], None))  # ignora o rótulo aqui, só quer o score
        recompensa_total += torch.softmax(score.logits, dim=-1)[0, 1].item()  # probabilidade de "correto"
    return recompensa_total / len(passos)  # recompensa média por passo, não só a resposta final
```

Diferente da `reward_correct_answer` do Projeto 9.5 (que só olha a resposta final), `recompensa_prm` dá crédito parcial por passos intermediários corretos, mesmo que a resposta final erre — um sinal mais denso, ao custo de exigir anotação mais granular no passo 1. Use essa função no lugar de `reward_correct_answer` no mesmo `GRPOTrainer` do Projeto 9.5, e compare a velocidade de convergência (quantos passos de treino até a taxa de acerto subir) entre outcome-only e PRM-based, no mesmo conjunto de problemas.

---

### Projeto 19.6 — SAE em modelo pequeno

Você vai treinar um Sparse Autoencoder sobre as ativações do GPT-2 (capturadas com TransformerLens, como no Projeto 14.6) e identificar features interpretáveis.

**Pré-requisitos**: o Projeto 14.6 completo.

```python
class SAE(nn.Module):
    def __init__(self, dim_ativacao, dim_esparsa):
        super().__init__()
        self.encoder = nn.Linear(dim_ativacao, dim_esparsa)
        self.decoder = nn.Linear(dim_esparsa, dim_ativacao)

    def forward(self, ativacoes):
        codigo_esparso = torch.relu(self.encoder(ativacoes))  # ReLU incentiva esparsidade (muitos zeros)
        reconstrucao = self.decoder(codigo_esparso)
        return reconstrucao, codigo_esparso

sae = SAE(dim_ativacao=768, dim_esparsa=768 * 8)  # espaço esparso bem maior que o original
optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)

for lote_ativacoes in dataloader_ativacoes:  # capturadas via cache["resid_post", camada] do Projeto 14.6, em lotes
    reconstrucao, codigo = sae(lote_ativacoes)
    loss_reconstrucao = nn.functional.mse_loss(reconstrucao, lote_ativacoes)
    loss_esparsidade = codigo.abs().mean()  # penaliza códigos densos, incentiva a maioria dos valores ficar em 0
    loss = loss_reconstrucao + 1e-3 * loss_esparsidade
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`dim_esparsa` maior que `dim_ativacao` é proposital — a hipótese de superposition (mod. 14) é que o modelo original comprime mais conceitos do que tem neurônios; o SAE expande para um espaço maior onde há "espaço" suficiente para cada conceito ter sua própria dimensão, sem sobreposição, desde que a maioria das dimensões fique em zero para qualquer ativação dada (`loss_esparsidade` incentiva isso).

**Identifique features interpretáveis**: para cada uma das `dim_esparsa` dimensões do `codigo_esparso`, encontre os textos (do seu corpus de teste) que mais fortemente ativam aquela dimensão específica — se uma dimensão se ativa consistentemente em textos sobre um tema específico (números, um idioma, um tipo de pontuação), você encontrou uma feature interpretável. Documente 5-10 delas.

---

### Projeto 19.7 — Model merging

Você vai combinar 2-3 fine-tunes do mesmo modelo base sem retreinar nada, usando `mergekit`, e avaliar se o merge supera cada modelo isolado.

**Pré-requisitos**: `pip install mergekit`, 2-3 modelos fine-tunados a partir do mesmo base (podem ser os adaptadores dos Projetos 9.1/9.2, mesclados de volta ao modelo base, ou outros fine-tunes públicos do mesmo modelo base no HF Hub).

```yaml
# merge_config.yaml
models:
  - model: qwen2.5-0.5b-sft-dominio-a
    parameters:
      weight: 1.0
  - model: qwen2.5-0.5b-sft-dominio-b
    parameters:
      weight: 1.0
merge_method: ties
base_model: Qwen/Qwen2.5-0.5B
parameters:
  density: 0.5
dtype: bfloat16
```

```bash
mergekit-yaml merge_config.yaml ./modelo-combinado
```

`merge_method: ties` implementa o TIES-merging: em vez de simplesmente fazer uma média dos pesos dos dois modelos (o que pode cancelar atualizações importantes que apontam em direções opostas), identifica os pesos com maior magnitude de mudança em cada modelo, resolve conflitos de sinal entre eles (mantendo a direção "majoritária"), e combina só essa fração mais significativa (`density: 0.5` mantém os 50% de maior magnitude, zera o resto). O resultado é um único modelo, do mesmo tamanho que qualquer um dos originais, sem nenhum passo de treino.

**Avalie com o LM-Eval-Harness** (Projeto 9.4/14.1) o modelo combinado contra cada um dos modelos originais, nas tasks relevantes a cada domínio — o resultado, surpreendentemente frequente na literatura, é que o merge iguala ou supera cada modelo isolado nas tarefas de ambos, sem o custo de manter e rotear entre múltiplos modelos separados.

---

### Projeto 19.8 — Long context "needle in haystack"

Você vai avaliar empiricamente onde um modelo perde informação dentro de um contexto longo, reproduzindo o experimento "lost in the middle".

**Pré-requisitos**: um modelo com contexto longo disponível (via Ollama ou API), `pip install matplotlib`.

```python
import random

def montar_contexto_com_agulha(tamanho_contexto_tokens, posicao_relativa, agulha="A senha secreta é XYZABC123."):
    texto_recheio = " ".join(["Este é um texto de preenchimento sem relação com a pergunta."] * (tamanho_contexto_tokens // 10))
    palavras = texto_recheio.split()
    posicao_insercao = int(len(palavras) * posicao_relativa)
    palavras.insert(posicao_insercao, agulha)
    return " ".join(palavras)

resultados = {}
for tamanho in [1000, 4000, 16000]:
    for posicao in [0.0, 0.25, 0.5, 0.75, 1.0]:
        contexto = montar_contexto_com_agulha(tamanho, posicao)
        prompt = f"{contexto}\n\nQual é a senha secreta mencionada no texto acima?"
        resposta = query_ollama(prompt)
        acertou = "XYZABC123" in resposta
        resultados[(tamanho, posicao)] = acertou

import matplotlib.pyplot as plt
import numpy as np

tamanhos = [1000, 4000, 16000]
posicoes = [0.0, 0.25, 0.5, 0.75, 1.0]
matriz = np.array([[resultados[(t, p)] for p in posicoes] for t in tamanhos])

plt.imshow(matriz, cmap="RdYlGn", aspect="auto")
plt.xticks(range(len(posicoes)), posicoes)
plt.yticks(range(len(tamanhos)), tamanhos)
plt.xlabel("Posição relativa da agulha no contexto")
plt.ylabel("Tamanho do contexto (tokens)")
plt.title("Needle in a Haystack — verde = achou, vermelho = não achou")
plt.savefig("needle_in_haystack.png")
```

`montar_contexto_com_agulha` esconde uma frase específica ("a agulha") dentro de um texto de preenchimento longo ("o palheiro"), numa posição relativa controlada (`0.0` = início, `1.0` = fim, `0.5` = meio); o heatmap resultante visualiza diretamente onde o modelo consegue (verde) ou não consegue (vermelho) recuperar a informação, cruzando tamanho de contexto com posição. Se a intuição de "lost in the middle" (mod. 12) se confirmar no seu modelo, espere ver uma faixa vermelha concentrada em `posicao=0.5` nos contextos maiores, com desempenho melhor nas bordas (`0.0` e `1.0`).

---

## Erros comuns

- **Tratar tudo aqui como "vai dominar logo"** — muitas dessas frentes são especulativas; mantenha ceticismo.
- **Confundir performance em paper com resultado em produção** — papers reportam o melhor caso.
- **Pular os fundamentos práticos** dos módulos anteriores — sem o loop de treino do Projeto 8.3 e o SwiGLU/attention que ele implementa, os Projetos 19.1 e 19.2 (que reaproveitam esse código diretamente) não fazem sentido.
- **Treinar difusão sem GPU adequada** — é caro e lento; comece com modelos pequenos (o MNIST do Projeto 19.3 roda até em CPU, ainda que devagar).

---

## Conexão com outros módulos

Este módulo recombina e estende todos os anteriores. Não há "próximo módulo natural"; vá para o 20 quando estiver pronto para projetos integradores.

---

## Checklist de saída

- [ ] Implementei (ou reproduzi) MoE pequeno from scratch, com auxiliary loss de load balancing (se não, revise o Projeto 19.1).
- [ ] Implementei DDPM completo e gerei imagens com ele, visualizando forward e reverse (se não, revise o Projeto 19.3).
- [ ] Tive contato prático com Mamba/SSMs e medi a diferença de escala de latência contra um Transformer (se não, revise o Projeto 19.2).
- [ ] Treinei um SAE introdutório e identifiquei features interpretáveis (se não, revise o Projeto 19.6 e o Projeto 14.6).
- [ ] Sei discutir com fluência pelo menos 3 destes tópicos: MoE, SSMs, difusão, reasoning RL, long context (se não, releia a seção correspondente e tente explicar em voz alta, sem consultar o texto).
- [ ] Consigo ler papers da fronteira sem depender de resumo/blog de terceiros.
