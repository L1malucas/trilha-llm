---
id: 15_engenharia_producao
title: "Módulo 15 — Engenharia de Produção (LLMOps)"
sidebar_position: 15
---

# Módulo 15 — Engenharia de Produção (LLMOps)

> **Objetivo**: levar sistemas de LLM da prototipagem para produção real — observability, custo, latência, cache, deploy, monitoring, drift, A/B testing, CI/CD.
>
> **Pré-requisitos**: Módulos 10–14.
>
> **Tempo de referência**: 3–5 semanas.

---

## Por que isso importa

Um demo bonito ≠ produto. Em produção você lida com:
- **Custo real** (cada token é dinheiro).
- **Latência** que afasta usuários.
- **Falhas em cascata**: API caiu, agente trava, usuário fica sem resposta.
- **Drift**: modelo, prompts, dados mudam; qualidade despenca silenciosamente.
- **Compliance**: logs, PII, auditoria.

Isto é o módulo que separa "fiz um chatbot" de "operou um sistema de IA".

---

## 15.1 Observability

### O tripé: traces, logs, metrics
- **Traces**: visão fim-a-fim de uma requisição (cada chamada LLM, cada tool, latência).
- **Logs**: eventos discretos, com contexto.
- **Metrics**: agregações (P50/P95/P99 de latência, taxa de erro, tokens/min).

### Específico de LLM
Adicione: tokens in/out por chamada, custo, prompt versionado, modelo, sampling params, score do juiz/eval, score do guardrail.

### Ferramentas
- 🛠 **Langfuse** — open-source, self-hostable. https://langfuse.com/
- 🛠 **LangSmith** — SaaS LangChain. https://www.langchain.com/langsmith
- 🛠 **Helicone** — proxy + analytics. https://www.helicone.ai/
- 🛠 **Phoenix** (Arize). https://github.com/Arize-ai/phoenix
- 🛠 **Weights & Biases — Weave**. https://wandb.ai/site/weave
- 🛠 **OpenTelemetry GenAI semantic conventions** — padrão emergente. https://opentelemetry.io/docs/specs/semconv/gen-ai/

### TS / JS
- Langfuse, Helicone e LangSmith têm SDKs TS.
- OpenTelemetry tem SDK Node.js completo.

### Princípios
- **Trace por sessão e por requisição** — sempre conseguir reconstruir a história.
- **Tags estruturadas** — modelo, versão de prompt, feature flag, tier de usuário.
- **Privacidade**: logs com PII precisam de redação ou criptografia em rest.

---

## 15.2 Custo e cost engineering

### Modelo mental de custo
```
custo = ∑ (tokens_in × preço_in + tokens_out × preço_out) [+ chamadas a tools / embeddings / vector DB]
```

### Comparativos públicos
- **Artificial Analysis**. https://artificialanalysis.ai/
- **OpenRouter** (preços e benchmarks). https://openrouter.ai/

### Estratégias de redução
- **Cache de prompts** (Anthropic prompt caching, OpenAI prompt caching) — reduz custo de prompts longos repetidos.
- **Cache semântico** — cache em nível de query (com embedding match).
- **Roteamento entre modelos** — modelo barato resolve o fácil; modelo caro só para o difícil. Routers: RouteLLM, Martian.
- **Compactação de prompts** — LLMLingua. 📄 https://arxiv.org/abs/2310.05736
- **Modelos menores fine-tunados** para tarefas específicas.
- **Batching** quando latência permite.
- **Streaming** para latência percebida menor (não custo, mas UX).

### Calculadoras
- Faça **a sua**: custo médio por user, por feature, p95 — e expresse em $/dia, $/mês.

---

## 15.3 Latência e performance

### Fatores
- **TTFT (Time To First Token)**: critico em chat.
- **TPOT (Time Per Output Token)**: define velocidade de geração.
- **Tokens totais**.
- **Network round-trips** entre serviços.
- **Cold starts** em serverless.

### Otimizações
- **Modelos menores** quando bastam.
- **Streaming** para esconder TPOT.
- **Speculative decoding** (mod. 10).
- **Prompt caching**.
- **Edge inference** quando possível.
- **Reduzir prompt** sem perder qualidade.

### Benchmarks
- **Lateralidade real do usuário** (browser → backend → LLM → backend → browser) é maior que latência da API.

---

## 15.4 Cache

### Níveis
- **Exact-match**: hash da query → resposta. Útil em FAQ.
- **Semantic cache**: embedding da query → busca top-1 no cache; se similaridade > threshold, retorna.
- **Prompt prefix cache** (servidor): mesmo system prompt entre usuários compartilha computação.
- **HTTP cache padrão** para recursos (docs, embeddings).

### Ferramentas
- 🛠 **GPTCache**. https://github.com/zilliztech/GPTCache
- 🛠 **Redis** com vector search.
- 🛠 **Provedor cache** (Anthropic, OpenAI).

### Cuidados
- Stale cache em domínios dinâmicos.
- Cache poisoning via prompt injection.

---

## 15.5 Deploy

### Padrões de deploy de aplicação LLM
- **Backend tradicional**: API (FastAPI, Express, Hono) chamando provedor LLM ou modelo self-hosted.
- **Serverless**: Vercel, Cloudflare Workers, AWS Lambda — bom para tráfego intermitente.
- **Edge**: Cloudflare AI, Vercel Edge — latência baixa, mas restrições de bibliotecas (TS-friendly).
- **Self-hosted modelo** + frontend: Kubernetes/Nomad com vLLM/TGI atrás de gateway.

### Containers
- **Docker** com multi-stage builds.
- **GPU containers**: imagens NVIDIA NGC ou builds próprios com CUDA + Python.
- **Air-gapped** quando dado é sensível.

### Frameworks/templates
- **LangServe** (Python) — deploy rápido de chains LangChain. https://www.langchain.com/langserve
- **BentoML**, **Ray Serve**.
- **Modal**, **Replicate**, **Banana** — serverless GPU.
- **HuggingFace Inference Endpoints** — managed.

### CI/CD para LLM apps
- Testes que incluem **eval contra dataset** em PR.
- **Versionamento de prompts** + dataset + modelo + código.
- **Canary deployment** — % do tráfego em nova versão.
- **Rollback rápido** quando métrica cai.

---

## 15.6 A/B testing e experimentação

### Específico para LLMs
- A/B em **prompt versions**, **modelo**, **temperatura**, **tool sets**.
- Métricas:
  - **Engajamento**: tempo de sessão, mensagens por sessão.
  - **Outcome**: tarefa completada, conversão, satisfação.
  - **Qualidade**: thumbs up/down, eval automática.
- **Statistical power**: LLM gera variância maior que features estáticas; n maior necessário.

### Ferramentas
- **GrowthBook**, **PostHog**, **Statsig** — feature flags + experimentos.
- **Promptfoo** integra eval com PRs.

---

## 15.7 Monitoring e drift

### O que monitorar
- **Saúde do sistema**: latência, taxa de erro, custo por dia.
- **Qualidade**: eval automático em sample de produção, thumbs up/down.
- **Drift**:
  - **Data drift**: distribuição de queries muda.
  - **Concept drift**: mesma query, resposta correta diferente (ex: notícias).
  - **Prompt drift**: alterações acumuladas degradam.
  - **Provider drift**: provider atualiza modelo silenciosamente.
- **Safety incidents**: jailbreaks, alucinação grave.

### Ferramentas
- Langfuse / LangSmith / Helicone agregam métricas.
- **Evidently AI** — drift em features tradicionais.
- **Arize Phoenix** — drift e tracing combinados.

### Ciclo
```
prod logs → sample → human/auto eval → identificar regressões → ajustar (prompt/model/RAG) → eval offline → deploy
```

---

## 15.8 Versionamento e governance

### O que versionar
- **Código** (Git, padrão).
- **Prompts** (banco, ou Git, ou ferramenta de prompt management).
- **Datasets** (DVC, Hugging Face datasets, Git LFS, blob storage).
- **Modelos** (HF Hub, MLflow Model Registry).
- **Embeddings indexados** (snapshots do vector DB).
- **Configurações** de pipeline.

### Reprodutibilidade
- Cada resposta em produção rastreável a:
  - Versão de código.
  - Versão de prompt.
  - Versão de modelo.
  - Versão de índice RAG.
  - Random seed (quando aplicável).

### Ferramentas
- 🛠 **MLflow**. https://mlflow.org/
- 🛠 **Weights & Biases**. https://wandb.ai/
- 🛠 **Aim**. https://github.com/aimhubio/aim
- 🛠 **DVC**. https://dvc.org/

---

## 15.9 Compliance e regulação

### Frameworks relevantes
- **GDPR** (UE) — direito à informação, deleção, portabilidade.
- **LGPD** (Brasil) — análoga ao GDPR.
- **EU AI Act** (2024+) — categorias de risco, transparência.
- **HIPAA** (saúde EUA), **PCI-DSS** (pagamentos).
- **NIST AI Risk Management Framework**. https://www.nist.gov/itl/ai-risk-management-framework

### Implicações práticas
- **Logs com PII**: criptografia, retenção limitada, redação.
- **Direito de explicação**: deve poder justificar saídas.
- **Direito de não submissão a decisão automatizada** (LGPD/GDPR).
- **Auditoria**: trilhas reproduzíveis.

---

## 15.10 Failure modes em produção

### Recorrentes
- **API do provider** caiu / latente. Mitigação: fallback para outro provider.
- **Rate limits** estouraram. Mitigação: queue, backoff exponencial, multi-key.
- **Timeout** em geração longa. Mitigação: streaming, max_tokens, deadline.
- **Output em formato errado**. Mitigação: validação + retry com mensagem de correção.
- **Loop em agente**. Mitigação: max_steps, detecção de repetição.
- **Custo descontrolado**. Mitigação: rate limit por user, alertas em $/h.
- **Prompt injection** explorada. Mitigação: guardrails + monitoring.
- **Jailbreak** viral. Mitigação: refusal mais forte + detecção.

### Padrão Circuit Breaker
Como em microsserviços tradicionais: se taxa de erro > X em janela Y, "abre" o circuito, falha rapidamente, retorna fallback. Especialmente útil com LLMs externos.

---

## 15.11 Padrões de arquitetura de aplicação

### Single-call (mais simples)
```
Frontend → Backend → LLM Provider → Backend → Frontend
```

### RAG-augmented
```
Frontend → Backend → [Retriever → Vector DB] → LLM → Backend → Frontend
```

### Agentic
```
Frontend → Agent Orchestrator (LangGraph/Mastra)
                ↓ ↑ múltiplas iterações
            LLM + Tools (DB, APIs, Files, MCP servers)
```

### Multi-tenant
Atenção a:
- Isolamento de contexto (PII de A não vaza para B).
- Rate limit por tenant.
- Custo allocado por tenant.
- Modelos/prompts customizados por tenant.

---

## 15.12 Custos invisíveis frequentes

- **Embeddings**: rebuild de índice a cada mudança de chunking custa caro.
- **Vector DB**: storage + queries em escala.
- **Tracing/observability** em SaaS pode escalar com volume.
- **Egress** (cloud egress fees) ao mover dados.
- **Cold-start em serverless GPU**: primeiros segundos pagam o preço do warmup.

---

## 🧪 Projetos práticos

### Projeto 15.1 — Observability completo
- Pegue um pipeline RAG/agente do mod. 12 ou 13.
- Adicione Langfuse self-hosted (Docker compose).
- Capture traces detalhados + custo por chamada.
- Construa dashboard com p50/p95 de TTFT, custo/dia, taxa de erro.

### Projeto 15.2 — Cache semântico
- Implemente semantic cache em Redis (com módulo de vetores) ou GPTCache.
- Meça hit rate em workload realista (logs de projeto anterior).
- Documente quando cache leva a respostas defasadas/erradas.

### Projeto 15.3 — Roteamento de modelos
- Implemente router: queries simples → modelo pequeno open; complexas → modelo grande.
- Use classificador (zero-shot ou pequeno LLM) para decidir.
- Compare custo total vs sempre usar modelo grande.

### Projeto 15.4 — Deploy multi-ambiente
- Mesmo backend (FastAPI ou Hono) deployado em:
  - Vercel / Cloudflare Workers (edge).
  - Container em VPS / Fly.io.
  - Self-hosted com vLLM atrás.
- Compare TTFT, custo, complexidade.

### Projeto 15.5 — Pipeline CI/CD com eval
- GitHub Actions:
  - PR triggera linting + testes unitários + suite de eval (Promptfoo) contra dataset fixo.
  - Métricas regredem? Bloqueia merge.
  - Merge dispara deploy canary.

### Projeto 15.6 — Detecção de drift
- Capture 1k queries reais.
- Calcule embeddings; detecte se distribuição mudou (KS-test em projeções, ou clustering).
- Alerte quando > X% de queries são "novas" comparadas à base de eval.

### Projeto 15.7 — Resiliência
- Implemente fallback: provider primário → secundário → modelo local.
- Circuit breaker.
- Teste com chaos: derrube providers e meça comportamento.

---

## ⚠️ Erros comuns

- **Não logar prompt + resposta** completos — debug fica impossível.
- **Logar PII** sem redação — exposição legal.
- **Eval só offline** — sem feedback de produção, drift mata silenciosamente.
- **Não medir custo** — surpresa no fim do mês.
- **Acoplamento com 1 provider** — quando ele cai/aumenta preço, sistema quebra.
- **Esquecer rate limit** quando tráfego sobe.
- **Confundir thumbs up/down com eval real** — sinal ruidoso, mas útil agregado.
- **Subestimar drift** silencioso quando provider atualiza modelo.

---

## Conexão com módulos seguintes

Este módulo é o "meta" sobre os anteriores. Os galhos (mod. 16–19) podem reaproveitar tudo aqui.

---

## Checklist de saída

- [ ] Tenho observability completo (traces + custo + qualidade) em projeto real.
- [ ] Pipeline com eval automático bloqueando regressões em CI.
- [ ] Implementei cache semântico ou prompt caching com ganho mensurável.
- [ ] Sei estimar e monitorar $ por usuário, $ por feature.
- [ ] Tenho fallback / circuit breaker para falhas de provider.
- [ ] Sei o básico de compliance (LGPD/GDPR) aplicado a LLM apps.
- [ ] Posso fazer A/B test de prompts/modelos com métricas válidas.
