---
id: 20_projetos_integradores
title: "Módulo 20 — Projetos Integradores"
sidebar_position: 13
---

# Módulo 20 — Projetos Integradores

> **Objetivo**: provar a si mesmo que o tronco desta trilha foi consolidado via **5 projetos finais grandes** que integram múltiplos módulos. Cada um vale por um portfólio.
>
> **Pré-requisitos**: Módulos [08](08_llms_arquiteturas.md)–[19](19_topicos_avancados.md). Os módulos de fundamento (matemática, programação, ML clássico/moderno, Deep Learning, NLP clássico, Transformers) só vêm depois deste, na ordem desta trilha — mas você já construiu na prática o essencial de cada um deles ao longo dos módulos anteriores (um Transformer do zero no Projeto 8.3, um pipeline de treino completo no mod. 09, entre outros). Onde um projeto abaixo pede algo genuinamente novo (como treinar um tokenizer do zero, no Projeto 20.2), o texto ensina isso no ponto em que aparece.
>
> **Tempo de referência**: 2–4 meses no total (escolha 2–3 projetos para fazer com profundidade, em vez de 5 superficialmente).

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Projetar, construir e operar um sistema de IA completo, não só um componente isolado.
- Tomar decisões de arquitetura defensáveis (documentadas, com trade-offs explícitos), não copiadas de tutorial.
- Avaliar um sistema com rigor (métricas, eval set, red-team) em vez de "parece bom".
- Escrever um post-mortem honesto — o que funcionou, o que não, o que você faria diferente.

Diferente dos módulos anteriores, aqui não há novo conceito a aprender — é onde tudo que você já sabe precisa funcionar junto, sob as restrições reais de um sistema completo.

---

## Por que isso importa

Aprendizado real consolida-se em **trabalho integrado**. Saber CNN + saber Transformer + saber RAG não significa nada até você projetar, construir, debugar e operar sistemas que combinem isso. Esses projetos são o "exame final" auto-aplicado.

**Para cada projeto**:
- Documente decisões em `DESIGN.md`.
- Use observability desde o início.
- Versione tudo (código, prompts, dados, modelo).
- Avalie objetivamente, não por "vibe".
- Escreva post-mortem honesto ao final.

---

## Projeto 20.1 — Assistente RAG profissional (PT-BR)

**Módulos integrados**: 02, 06, 08, 10, 11, 12, 14, 15

### Escopo
Sistema RAG sério em português brasileiro sobre corpus do seu interesse: legislação pública, papers científicos, documentação interna, livros didáticos, ou diários pessoais.

### Requisitos técnicos
- **Corpus**: ≥1000 documentos (PDFs, MDs, HTMLs).
- **Pré-processamento**: extração estrutural (Marker/Docling/Unstructured), chunking semântico ou contextual.
- **Embeddings**: comparar 3 modelos (multilingual-e5, BGE-M3, Jina v3) com benchmark próprio em PT-BR.
- **Vector DB**: self-hosted (Qdrant ou pgvector).
- **Retrieval**: hybrid (BM25 + dense) + reranker (bge-reranker-v2-m3).
- **Geração**: modelo open local (Qwen 2.5 7B / LLaMA 3 8B / Gemma 2) via vLLM ou Ollama.
- **Streaming** + citações com IDs de docs.
- **Avaliação**: 100 perguntas com gabarito; medir Recall@5, Faithfulness (RAGAS), Answer Relevance.
- **Observability**: Langfuse self-hosted.
- **Frontend**: web app em Next.js + Vercel AI SDK (TS) ou Streamlit (Python).
- **Deploy**: containerizado, com reprodutibilidade.

### Entregáveis
- Repositório com README detalhado.
- DESIGN.md com decisões e trade-offs.
- Eval set + relatório.
- Post-mortem: o que funcionou, o que não.

### O que isso prova
Você consegue conduzir um produto LLM ponta-a-ponta, com avaliação rigorosa, em condições próximas de produção, em PT-BR.

> **Como abordar**: construa na ordem em que os requisitos aparecem — retrieval simples primeiro (embedding único, sem hybrid nem reranker), confirme que ele já responde razoavelmente às 100 perguntas de eval, e só então adicione hybrid retrieval, reranker e observability, medindo o ganho incremental de cada peça. Adicionar tudo de uma vez torna impossível saber qual componente está ajudando ou atrapalhando quando algo sai errado.

---

## Projeto 20.2 — Mini-LLM treinada do zero

**Módulos integrados**: 01, 02, 05, 06, 07, 08, 09, 10

### Escopo
Treinar uma LLM **pequena, mas funcional**, do zero. Não é "criar o próximo GPT-4" — é provar domínio do ciclo completo.

### Requisitos técnicos
- **Tamanho**: 50M–500M parâmetros (alvo razoável: ~125M, escala de GPT-2 small) — a mesma arquitetura `MiniLlama` do Projeto 8.3, só que maior (`dim`, `n_layers`, `n_heads` maiores) e treinada num corpus de verdade, não Tiny Shakespeare.
- **Arquitetura LLaMA-style**: RMSNorm + RoPE + SwiGLU + GQA — já implementados por você no Projeto 8.3; reaproveite o código diretamente.
- **Tokenizer próprio** (BPE ou SentencePiece) treinado em corpus — a única peça genuinamente nova deste projeto, detalhada abaixo.
- **Corpus de pretrain**: subconjunto do FineWeb-Edu, ou corpus PT-BR (Wikipedia + OSCAR-PT + livros públicos), ~5–20B tokens.

**Treinando um tokenizer BPE do zero**: até aqui, você usou tokenização caractere-a-caractere (Projeto 8.3) ou tokenizadores já prontos, carregados via `AutoTokenizer` (Projeto 9.1 em diante). BPE (Byte-Pair Encoding) é o meio-termo usado por LLMs de produção: começa com caracteres individuais e, repetidamente, funde o par de tokens adjacentes mais frequente no corpus num novo token único — depois de milhares de fusões, os tokens resultantes tendem a ser sílabas, prefixos/sufixos comuns, e até palavras inteiras frequentes, sem que ninguém tenha definido essas unidades manualmente.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("meu_tokenizer.json")

saida = tokenizer.encode("RMSNorm é mais simples que LayerNorm.")
print(saida.tokens)  # ex.: ['RMS', 'Norm', 'Ġé', 'Ġmais', 'Ġsimples', ...]
```

`pre_tokenizers.ByteLevel()` trata o texto como uma sequência de bytes antes de aplicar BPE — isso garante que qualquer caractere Unicode (incluindo acentos do português, emojis, símbolos raros) tenha uma representação válida, sem precisar de um token `[UNK]` para tudo que não foi visto no treino do tokenizer. `vocab_size=32000` é o número final de tokens no vocabulário — bem maior que os ~65 símbolos do tokenizer caractere-a-caractere do Projeto 8.3, e é exatamente essa diferença de granularidade que torna sequências de texto real mais curtas em tokens BPE (cada token carrega mais informação) do que seriam em tokens de caractere único. Depois de treinado, use esse tokenizer no lugar de `encode`/`decode` do Projeto 8.3 — a arquitetura do modelo não muda nada, só o `vocab_size` do `MiniLlama` passa a refletir os 32000 tokens do BPE em vez dos ~65 caracteres.
- **Treinamento**: GPU única ou múltipla; usar gradient accumulation, mixed precision (BF16), AdamW + warmup + cosine.
- **Logging**: W&B.
- **Avaliação durante treino**: perplexity em dev set, plus benchmarks pequenos.
- **SFT**: instruir o modelo com Alpaca-PT ou dataset próprio (≥10k pares).
- **DPO**: ≥1k pares de preferência.
- **Quantização**: exportar para GGUF Q4_K_M.
- **Inferência**: rodar via llama.cpp e via vLLM.

### Entregáveis
- Repositório com pipeline reproduzível.
- Modelo publicado no Hugging Face Hub.
- Relatório técnico (~5 páginas) com gráficos de loss e benchmarks.

### O que isso prova
Você entende **toda a stack** de criação de uma LLM. Você não é "usuário"; é construtor.

### Realismo
- Corpus pequeno (1B tokens) já produz modelo "que conversa de jeito feio".
- Corpus médio (~10B tokens) já produz modelo razoável em domínio limitado.
- Não espere competir com Mistral 7B; espere **entender por que Mistral 7B é o que é**.

> **Como abordar**: valide o pipeline inteiro (tokenizer → pretrain → SFT → DPO → quantização → inferência) num corpus minúsculo e num modelo minúsculo primeiro (ex.: 1M parâmetros, poucos milhões de tokens, treino de minutos) — só depois de confirmar que o pipeline completo funciona ponta a ponta, escale para o tamanho real. Descobrir um bug de pipeline depois de dias de treino em escala real é o erro mais caro possível neste projeto.

---

## Projeto 20.3 — Agente autônomo multi-modal

**Módulos integrados**: 11, 12, 13, 14, 15, 18

### Escopo
Construir agente que executa tarefas multi-passo no mundo real, integrando texto, imagem (e opcionalmente fala).

### Sugestões de domínio
- Assistente de pesquisa que lê papers e gera relatório citando fontes.
- Assistente que analisa screenshots de dashboards e gera insights.
- Pipeline editorial: extrai dados de imagens de produtos → categoriza → gera descrições.
- Agente de "suporte de TI" que lê screenshots de erros e sugere soluções.

### Requisitos técnicos
- **Framework**: LangGraph (Python) ou Mastra (TS) — escolha o stack.
- **LLM**: pelo menos uma versão usa modelo open local (não pode ficar refém de API).
- **VLM**: Qwen2.5-VL ou InternVL 2.5 para componentes visuais.
- **Tools**: ≥6 ferramentas (busca, fetch URL, file ops, DB, calculadora, etc.).
- **MCP**: pelo menos 1 server MCP customizado integrado.
- **Memória**: short-term + long-term (vector DB).
- **Observability**: Langfuse, com traces completos.
- **Resiliência**: circuit breaker, retry com backoff, max_steps.
- **Avaliação**: 30 tarefas com gabarito + métricas (success rate, steps, custo).
- **Red-team**: 10 ataques de prompt injection com mitigação.

### Entregáveis
- Repositório.
- Vídeo (≤5 min) demonstrando execuções reais.
- Relatório de avaliação + red-team.

### O que isso prova
Você sabe projetar agente sério, não só seguir tutorial.

> **Como abordar**: implemente e teste cada tool isoladamente antes de dar acesso a elas pro agente (mesmo conselho do Projeto 13.1) — com 6+ ferramentas, um agente que "não está funcionando" pode ter o problema em qualquer uma delas; eliminar essa incerteza antes de integrar economiza dias de debug depois.

---

## Projeto 20.4 — Fine-tuning especialista

**Módulos integrados**: 03, 04, 09, 10, 14

### Escopo
Pegar um problema **específico e bem delimitado** e dominar via fine-tuning vs alternativas.

### Sugestões de domínio
- Classificação fina em domínio próprio (ex: tickets de suporte em 20 categorias).
- Extração estruturada de receitas culinárias para JSON com schema rígido.
- Tradução EN ↔ PT-BR especializada (técnica, jurídica, médica).
- Gerador de código em DSL específica (SQL para schema seu, regex, configs).
- Resumo de documentos legais com formato fixo.

### Requisitos técnicos
- **Comparação justa**:
  1. Baseline: LLM grande via prompt engineering puro (sem fine-tuning).
  2. RAG sobre corpus de exemplos.
  3. Fine-tuning com LoRA/QLoRA.
  4. (opcional) DPO sobre o fine-tune.
  5. (opcional) Distillation para modelo ainda menor.
- **Dataset**: ≥1000 exemplos para SFT (manuais ou semi-sintéticos), ≥200 para eval.
- **Métricas específicas** ao problema (não apenas accuracy).
- **Análise de custo**: $/inferência para cada abordagem.
- **Análise de latência**: p50/p95.
- **Análise de erro**: confusion matrix, exemplos qualitativos.

### Entregáveis
- Repositório.
- Relatório técnico comparando todas as abordagens.
- Modelo final no HF Hub (se permitido pela licença).

### O que isso prova
Você sabe escolher a abordagem certa para o problema, com base em evidência.

> **Como abordar**: construa o eval set (≥200 exemplos com gabarito) **antes** de implementar qualquer uma das 5 abordagens — com o eval pronto de antemão, cada abordagem é medida com a mesma régua, sem viés de "ajustar o gabarito pra parecer que aquela abordagem foi melhor" depois de já ter visto os resultados.

---

## Projeto 20.5 — Sistema completo em produção

**Módulos integrados**: TODOS os anteriores aplicados a um sistema real.

### Escopo
Lançar um sistema de IA real, com usuários reais (mesmo que poucos amigos/colegas), durante ≥30 dias.

### Sugestões
- Bot Telegram/Discord/Slack com utilidade real para um grupo.
- Web app que resolve um problema seu próprio.
- Plugin ou extensão para ferramenta que você usa.
- Servidor MCP que expõe seus dados pessoais para Claude/IDEs.

### Requisitos não-funcionais
- **≥30 dias** em produção.
- **≥10 usuários** distintos.
- **Custo monitorado** com budget alerting.
- **Métricas** colhidas e analisadas (engagement, retenção, qualidade).
- **Pelo menos 1 incidente** documentado e resolvido.
- **Pelo menos 1 iteração** baseada em feedback (release v2).
- **Eval contínuo** de qualidade.
- **Plano de recuperação** se provider cair.
- **Compliance**: termos de uso, política de privacidade básica, redação de PII.

### Entregáveis
- Sistema rodando.
- Repositório público (ou descrição se for privado).
- Relatório retrospectivo: o que aprendeu sobre operar IA em produção.

### O que isso prova
Você não é só estudante; você é operador. Conhece o que dá errado quando a teoria encontra usuário.

> **Como abordar**: lance a versão mais simples possível que já seja útil, o quanto antes — cada dia adicional em produção com usuários reais é mais valioso pra esse projeto do que mais uma semana polindo antes de lançar. O "pelo menos 1 incidente" e "pelo menos 1 iteração baseada em feedback" só acontecem depois que existe uso real; adiar o lançamento adia esses aprendizados.

---

## Como escolher quais projetos fazer

### Se seu objetivo é **engenharia de aplicações**:
20.1 + 20.3 + 20.5.

### Se seu objetivo é **pesquisa/treinamento**:
20.2 + 20.4 + 19.x (mod. [19](19_topicos_avancados.md)) com profundidade.

### Se seu objetivo é **fundador/produto**:
20.1 + 20.5 são suficientes; o resto vem por demanda.

### Se você quer **cobrir tudo**:
20.1 → 20.4 → 20.3 → 20.5 → (eventualmente) 20.2.

20.2 (mini-LLM do zero) é o mais "acadêmico" e o mais demorado em GPU. Faça por último, ou só se realmente quer essa experiência.

---

## Padrões de qualidade para os projetos

Independente do escopo, estes itens **devem** estar presentes:

### Repositório
- README claro: o que faz, como rodar, exemplos.
- Estrutura limpa.
- `.env.example` com variáveis necessárias.
- Docker/Compose para reprodução.
- CI básico (lint + tests).

### Documentação
- DESIGN.md: decisões arquiteturais e por quê.
- EVAL.md: como foi avaliado, com números.
- POSTMORTEM.md: erros, soluções, o que faria diferente.

### Código
- Linting (ruff/black em Python; biome/eslint em TS).
- Tipagem (Python: typing; TS: estrita).
- Tests para componentes críticos (parsing de outputs, retrieval, etc.).

### Observability
- Logs estruturados.
- Tracing (Langfuse/OpenTelemetry).
- Métricas (Prometheus, ou via SaaS).
- Dashboards (Grafana ou no provider).

---

## Checklist do tronco (módulos 08–20)

Quando você concluir 2–3 desses projetos, deve poder afirmar:

- [ ] Sei criar um LLM do zero (mesmo que pequeno) (se não, revise o Projeto 8.3 e o Projeto 20.2).
- [ ] Sei treinar e alinhar (SFT + DPO/RL) um LLM (se não, revise o mod. 09).
- [ ] Sei rodar inferência local com performance (se não, revise o mod. 10).
- [ ] Sei construir RAG profissional com avaliação rigorosa (se não, revise o mod. 12 e o Projeto 20.1).
- [ ] Sei construir agentes complexos com observability (se não, revise o mod. 13 e o Projeto 20.3).
- [ ] Sei fazer multimodal (visão + texto, e/ou áudio) (se não, revise os mod. 16 e 18).
- [ ] Sei fazer eval rigoroso e red-team (se não, revise o mod. 14).
- [ ] Sei levar para produção e operar (se não, revise o mod. 15 e o Projeto 20.5).
- [ ] Leio papers de fronteira sem ajuda externa (se não, revise o mod. 19).
- [ ] Tenho produtos rodando para usuários reais (se não, revise o Projeto 20.5).

Os itens sobre matemática, ML clássico/moderno e TypeScript/Python como papéis distintos ficam para o checklist final, depois dos módulos de fundamento (7→1) que vêm a seguir — você já usou boa parte disso na prática, mas a explicação formal ainda está por vir.

---

## O que vem a seguir nesta trilha

Os módulos 08–20 formam o tronco de LLMs e as especializações — a parte da trilha voltada a construir e operar sistemas. A partir daqui, a navegação desce para os módulos de fundamento, em ordem decrescente de proximidade com o que você acabou de fazer: [07 — Transformers](07_transformers.mdx) (a fundo, formalizando o que você já implementou no Projeto 8.3), depois NLP Clássico, Deep Learning, ML Moderno, ML Clássico, Programação e Ferramentas, e por fim Matemática. Não é revisão no sentido de "repetir" — é a base formal por trás de código que você já escreveu e já viu funcionar, o que costuma tornar a teoria mais fácil de reter do que teria sido antes de qualquer prática.

Se seu objetivo era só aplicação (engenharia de produtos com LLM), você já tem o essencial — os módulos seguintes são para quem quer a base explicada a fundo, não só usada. Ainda assim, vale pelo menos passar pelo mod. [07](07_transformers.mdx): é o mais diretamente conectado a tudo que você construiu até aqui.
