# Módulo 20 — Projetos Integradores

> **Objetivo**: provar a si mesmo que a trilha foi consolidada via **5 projetos finais grandes** que integram múltiplos módulos. Cada um vale por um portfólio.
>
> **Pré-requisitos**: trilha completa (módulos 01–19, ou pelo menos os referenciados em cada projeto).
>
> **Tempo de referência**: 2–4 meses no total (escolha 2–3 projetos para fazer com profundidade, em vez de 5 superficialmente).

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

---

## Projeto 20.2 — Mini-LLM treinada do zero

**Módulos integrados**: 01, 02, 05, 06, 07, 08, 09, 10

### Escopo
Treinar uma LLM **pequena, mas funcional**, do zero. Não é "criar o próximo GPT-4" — é provar domínio do ciclo completo.

### Requisitos técnicos
- **Tamanho**: 50M–500M parâmetros (alvo razoável: ~125M, escala de GPT-2 small).
- **Arquitetura LLaMA-style**: RMSNorm + RoPE + SwiGLU + GQA.
- **Tokenizer próprio** (BPE ou SentencePiece) treinado em corpus.
- **Corpus de pretrain**: subconjunto do FineWeb-Edu, ou corpus PT-BR (Wikipedia + OSCAR-PT + livros públicos), ~5–20B tokens.
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

---

## Como escolher quais projetos fazer

### Se seu objetivo é **engenharia de aplicações**:
20.1 + 20.3 + 20.5.

### Se seu objetivo é **pesquisa/treinamento**:
20.2 + 20.4 + 19.x (mod. 19) com profundidade.

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

## Checklist final da trilha

Quando você concluir 2–3 desses projetos, deve poder afirmar:

- [ ] Sei criar um LLM do zero (mesmo que pequeno).
- [ ] Sei treinar e alinhar (SFT + DPO/RL) um LLM.
- [ ] Sei rodar inferência local com performance.
- [ ] Sei construir RAG profissional com avaliação rigorosa.
- [ ] Sei construir agentes complexos com observability.
- [ ] Sei fazer multimodal (visão + texto, e/ou áudio).
- [ ] Sei fazer eval rigoroso e red-team.
- [ ] Sei levar para produção e operar.
- [ ] Leio papers de fronteira sem ajuda externa.
- [ ] Domino Python e TypeScript para os papéis distintos que cada um tem em IA.
- [ ] Tenho intuição matemática real, não decorada.
- [ ] Sei escolher entre ML clássico, DL, e LLM para cada problema.
- [ ] Tenho produtos rodando para usuários reais.

---

## Depois da trilha

Você não "termina" IA — a área se move semanalmente. Mantenha hábito de:

- 📰 **arXiv (cs.CL, cs.LG)** — sanity check semanal.
- 📰 **Hugging Face Daily Papers**. https://huggingface.co/papers
- 📰 **Anthropic Engineering blog**, **OpenAI**, **Mistral**, **DeepMind**, **Google AI**.
- 📰 **Sebastian Raschka — Ahead of AI** newsletter.
- 📰 **Andrej Karpathy** — toda saída dele.
- 📰 **Lilian Weng — lilianweng.github.io** — long-form deep dives.
- 📰 **Jay Alammar** — visualizações didáticas.
- 📰 **Simon Willison** — engenharia prática com LLMs.

E principalmente: **construa**. Toda nova técnica ganha solidez quando passa pelas suas próprias mãos.

---

## Última nota

Esta trilha tem ~21 arquivos e centenas de referências. **Não é um currículo passivo**. É um cardápio para 12–24 meses de trabalho ativo.

Você não precisa fazer tudo. Você precisa fazer **o suficiente para virar quem você quer ser**. O resto, conforme demanda.

Bom trabalho. 🚀
