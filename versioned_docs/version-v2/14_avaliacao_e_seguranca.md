---
id: 14_avaliacao_e_seguranca
title: "Módulo 14 — Avaliação e Segurança de LLMs"
sidebar_position: 14
---

# Módulo 14 — Avaliação e Segurança de LLMs

> **Objetivo**: avaliar LLMs cientificamente — benchmarks, LLM-as-judge, alucinação, viés, calibração — e proteger sistemas contra ataques (prompt injection, jailbreaks, data leakage). Inclui interpretabilidade mecanística como ferramenta de safety.
>
> **Pré-requisitos**: Módulos [09](09_treinamento_e_alinhamento.mdx), [11](11_prompt_engineering.md)–[13](13_agentes_tools_protocolos.md).
>
> **Tempo de referência**: 3–4 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Explicar as principais fontes de viés em LLM-as-judge e pelo menos uma mitigação para cada.
- Distinguir os tipos de alucinação e escolher a mitigação certa por tipo.
- Calcular calibração (ECE) conceitualmente e explicar por que RLHF tende a piorá-la.
- Explicar a diferença entre contaminação de benchmark e generalização real.
- Explicar, num nível intuitivo, o que "circuito" e "feature" significam em interpretabilidade mecanística.

---

## Por que isso importa

"O modelo está bom" não é avaliação. Sem metodologia, decisões viram folclore. **E**: aplicação real exige garantia de safety, viés, robustez. Lançar LLM em produção sem eval e sem red-team é negligência profissional.

---

## 14.1 Benchmarks acadêmicos clássicos

### Conhecimento e raciocínio
- **MMLU (Massive Multitask Language Understanding)** — 57 áreas de conhecimento. `Paper` https://arxiv.org/abs/2009.03300
- **MMLU-Pro** — versão mais difícil. `Paper` https://arxiv.org/abs/2406.01574
- **BIG-Bench / BIG-Bench Hard** — 200+ tarefas diversas. `Paper` https://arxiv.org/abs/2206.04615
- **HellaSwag** — commonsense reasoning. `Paper` https://arxiv.org/abs/1905.07830
- **ARC (AI2 Reasoning Challenge)**. `Paper` https://arxiv.org/abs/1803.05457
- **TruthfulQA** — checa veracidade contra mitos comuns. `Paper` https://arxiv.org/abs/2109.07958
- **GPQA** — perguntas de pós-graduação resistentes a Google. `Paper` https://arxiv.org/abs/2311.12022

### Matemática
- **GSM8K** — problemas de matemática elementar. `Paper` https://arxiv.org/abs/2110.14168
- **MATH** — problemas de competição. `Paper` https://arxiv.org/abs/2103.03874
- **AIME** — provas de olimpíada (em modo eval).

### Código
- **HumanEval** — completar funções Python. `Paper` https://arxiv.org/abs/2107.03374
- **MBPP** — Mostly Basic Python Problems.
- **LiveCodeBench** — problemas atualizados, anti-contaminação. https://livecodebench.github.io/
- **SWE-bench / SWE-bench Verified** — bugs reais em repos. https://www.swebench.com/

### Linguagem geral
- **GLUE / SuperGLUE** — clássicos pré-LLM.
- **HELM (Holistic Evaluation of LMs)** — Stanford CRFM. `Paper` https://arxiv.org/abs/2211.09110
- **LiveBench** — benchmark dinâmico. https://livebench.ai/

### Português / multilíngue
- **MMLU-PT-BR** (vários esforços de tradução).
- **BLUEX** — vestibulares brasileiros. https://github.com/Portuguese-NLP/luana-blue
- **ENEM Challenge**. https://github.com/piresramon/enem-challenge
- **MEGA** — benchmark multilíngue do Microsoft. https://arxiv.org/abs/2303.12528

### Ferramentas
- `Ferramenta` **lm-evaluation-harness** (EleutherAI) — padrão de fato para rodar benchmarks. https://github.com/EleutherAI/lm-evaluation-harness
- `Ferramenta` **HELM** runner. https://github.com/stanford-crfm/helm
- `Ferramenta` **BIG-Bench** runner.

---

## 14.2 LLM-as-judge

### Conceito
Usar LLM forte para avaliar saída de outro LLM. Escala bem para outputs livres (resumos, geração).

### Prós
- Avaliação semântica.
- Escalabilidade.

### Contras
- **Bias do juiz** (favorece próprio estilo).
- **Position bias** (prefere primeira resposta em A/B).
- **Verbosity bias** (prefere respostas longas).
- **Self-preference**.

> **Intuição**: um LLM-juiz não é um oráculo objetivo — ele traz os mesmos vieses estatísticos aprendidos no seu próprio treino (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)) para o julgamento. Verbosity bias, por exemplo, existe porque respostas humanas "melhores" no dataset de preferências usado no RLHF/DPO do juiz tendiam a ser mais completas — o juiz generaliza isso para "mais longo = melhor", mesmo quando conciso seria mais apropriado. Position bias (preferir a primeira opção numa comparação A/B) é um artefato ainda menos "semântico" — mais parecido com um viés posicional de atenção do que um julgamento de conteúdo. Isso não invalida LLM-as-judge (é rápido e escala onde avaliação humana não escala), mas exige as mitigações da lista abaixo tratadas como obrigatórias, não opcionais.

### Mitigações
- Randomizar ordem.
- Múltiplos juízes (ensemble).
- Rubricas estruturadas.
- Humanos no loop para calibrar.

> **Checkpoint**: sem olhar o texto, explique por que um LLM-juiz tende a preferir respostas mais longas, mesmo quando isso não reflete qualidade real.

### Referências
- `Paper` **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al. (2023). https://arxiv.org/abs/2306.05685
- `Paper` **Large Language Models are not Fair Evaluators** — Wang et al. (2023). https://arxiv.org/abs/2305.17926

---

## 14.3 Avaliação humana e Arena-style

- **Chatbot Arena** (LMSys) — comparação A/B humana, ranking ELO. https://arena.lmsys.org/
- **Crowdsourcing** com plataformas como Scale AI, Surge, Toloka.
- **Avaliação interna** com rubrica clara.

### Referências
- `Paper` **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference** — Chiang et al. (2024). https://arxiv.org/abs/2403.04132

---

## 14.4 Contaminação de benchmark

Modelos modernos viram benchmarks por terem visto **o próprio benchmark** durante pretraining.

### Sinais
- Performance >>> em benchmark conhecido vs versão "limpa".
- "Memorização" exata de strings do dataset.

### Combate
- Benchmarks dinâmicos (LiveBench, LiveCodeBench).
- Datasets privados.
- Contamination detection (Min-K%, perplexity-based).

> **Intuição**: um LLM pré-treinado em "todo o texto da internet" tem chance real de ter visto o benchmark de teste, ou uma cópia dele, durante o pré-treinamento (mod. [08](08_llms_arquiteturas.md#83-pré-treinamento-o-que-faz-uma-llm-saber)) — nesse caso, "acertar o benchmark" mede memorização, não capacidade de generalização, que é o que o benchmark deveria medir. É por isso que benchmarks *dinâmicos* (que adicionam problemas novos continuamente, como LiveCodeBench) são mais confiáveis a longo prazo do que benchmarks estáticos publicados há anos — problemas novos não podem ter sido vistos no pré-treino de um modelo já treinado.

### Referências
- `Paper` **Detecting Pretraining Data from LLMs (Min-K%)** — Shi et al. (2023). https://arxiv.org/abs/2310.16789
- `Paper` **Investigating Data Contamination in Modern Benchmarks**. https://arxiv.org/abs/2311.09783

---

## 14.5 Hallucination (alucinação)

### Tipos
- **Intrínseca**: contradiz o input.
- **Extrínseca**: fatos não suportados, plausíveis ou não.
- **Closed-domain** (em RAG): contradiz docs fornecidos.
- **Open-domain**: invenção genérica.

### Causas
- Pretraining noise.
- SFT com rótulos imprecisos.
- Pressão de "responder algo" mesmo sem saber.
- Reasoning model "confiante" em caminho errado.

> **Intuição**: um LLM não tem um mecanismo interno de "sei" vs "não sei" comparável ao de um humano consultando a própria memória — ele gera o próximo token mais provável dado o contexto (mod. [07](07_transformers.mdx)), e "um fato plausível mas inventado" pode ter probabilidade tão alta quanto "um fato real", especialmente em áreas pouco representadas no treino. SFT agrava isso quando o dataset de instrução tem poucos exemplos de "eu não sei" — o modelo aprende implicitamente que sempre deve produzir uma resposta confiante, porque é isso que os exemplos de treino demonstram fazer. Cada tipo de alucinação pede uma mitigação diferente: closed-domain (RAG contradizendo docs) se resolve com grounding mais forte e citação obrigatória; open-domain (invenção sem base nenhuma) se beneficia mais de calibração/refusal treinado (seção 14.7) e chain of verification.

### Mitigação
- **RAG** com grounding obrigatório.
- **Constrained generation** (mod. [11](11_prompt_engineering.md)).
- **Self-consistency**.
- **Chain of Verification**. `Paper` https://arxiv.org/abs/2309.11495
- **Citação obrigatória** com validação.
- **Modelos com refusal calibrado** (treinados a dizer "não sei").

> **Checkpoint**: sem olhar o texto, explique por que "responder algo" pode ter sido implicitamente reforçado durante SFT. Depois, explique a diferença entre alucinação closed-domain e open-domain, e por que a mitigação difere.

### Métricas
- **FactScore** — atomicidade + verificação. https://arxiv.org/abs/2305.14251
- **TruthfulQA**.
- **HaluEval**. https://arxiv.org/abs/2305.11747

### Referências
- `Paper` **Survey of Hallucination in Natural Language Generation** — Ji et al. (2022). https://arxiv.org/abs/2202.03629

---

## 14.6 Bias, fairness, harmful outputs

### Categorias de bias
- **Demográfico** (gênero, raça, religião, classe).
- **Cultural / linguístico** (anglocentrismo).
- **Político**.
- **Profissional / estereótipos ocupacionais**.

### Benchmarks
- **BBQ (Bias Benchmark for QA)**. https://arxiv.org/abs/2110.08193
- **StereoSet**. https://arxiv.org/abs/2004.09456
- **CrowS-Pairs**. https://arxiv.org/abs/2010.00133
- **BOLD** — geração e bias.

### Conceito de fairness
- **Disparate impact**, **demographic parity**, **equalized odds**.
- Trade-off com accuracy.

> Viés em LLMs tem a mesma raiz que viés em ML clássico (mod. [03](03_ml_classico.md)): o modelo aprende os padrões estatísticos do que viu no treino, incluindo estereótipos presentes nos dados — a diferença é de escala (o corpus de treino de um LLM é vastamente maior e mais difícil de auditar linha a linha que um dataset tabular).

### Referências
- `Paper` **On the Dangers of Stochastic Parrots** — Bender et al. (2021). https://dl.acm.org/doi/10.1145/3442188.3445922
- `Livro` **Fairness and Machine Learning: Limitations and Opportunities** — Barocas, Hardt, Narayanan. https://fairmlbook.org/

---

## 14.7 Calibração

### O que é
Modelo bem calibrado: quando diz "80% certeza", está certo 80% das vezes.

### Por que importa
- Decisões automatizadas.
- Compor com outros sistemas.
- "Honestidade" do modelo.

### Métricas
- **Expected Calibration Error (ECE)**.
- **Brier score**.
- Reliability diagrams.

> **Exemplo resolvido — ECE**: agrupe as previsões do modelo por faixa de confiança (ex.: previsões onde o modelo disse "~80% de confiança") e compare com a accuracy real *daquele grupo*. Se o modelo fez 100 previsões com confiança declarada de 80%, e só 60 delas estavam corretas, há um erro de calibração de `|0.80 - 0.60| = 0.20` para esse grupo. ECE é a média ponderada desse erro através de todas as faixas de confiança — um modelo perfeitamente calibrado tem ECE próximo de 0 (a confiança declarada sempre bate com a taxa de acerto real observada).

### LLMs e calibração
LLMs grandes pré-RLHF tendem a ser bem calibrados. **RLHF tipicamente piora calibração** (modelo aprende a parecer mais confiante).

> **Intuição — por que RLHF piora calibração**: durante RLHF/DPO (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)), avaliadores humanos tendem a preferir respostas que *soam* confiantes e diretas sobre respostas que expressam incerteza genuína — mesmo quando a incerteza seria mais honesta. O modelo otimiza pra essa preferência, aprendendo a *soar* mais confiante independente da confiança real subjacente — a distribuição de probabilidade interna pode continuar razoavelmente informativa, mas o texto gerado (que é o que o usuário vê) passa a expressar confiança quase uniformemente alta.
>
> **Checkpoint**: sem olhar o texto, explique com suas palavras o que significa um modelo estar "mal calibrado" com um exemplo numérico simples.

### Referências
- `Paper` **Calibration of Pre-trained Transformers** — Desai & Durrett (2020). https://arxiv.org/abs/2003.07892
- `Paper` **Just Ask for Calibration** — Tian et al. (2023). https://arxiv.org/abs/2305.14975

---

## 14.8 Adversarial robustness e Jailbreaks

### Tipos de ataque
- **Prompt injection** direto e indireto (revisar mod. [11](11_prompt_engineering.md)).
- **Jailbreak**: induzir comportamento proibido (DAN, role-play, encoding, multi-turn).
- **Universal adversarial suffixes**. `Paper` https://arxiv.org/abs/2307.15043
- **Many-shot jailbreaking** (Anthropic). https://www.anthropic.com/news/many-shot-jailbreaking
- **Crescendo** — ataque multi-turno escalado. https://arxiv.org/abs/2404.01833

### Defesas (todas parciais)
- **Input filtering**.
- **Output filtering** (Llama Guard, NeMo Guardrails).
- **Constitutional AI / RLAIF**.
- **Adversarial training**.

> Many-shot jailbreaking é um bom exemplo de como um recurso benéfico (janelas de contexto maiores, que ajudam few-shot learning legítimo — mod. [11](11_prompt_engineering.md#112-técnicas-fundamentais)) também expande a superfície de ataque: preencher o contexto com muitos exemplos de "comportamento proibido sendo seguido" explora o mesmo mecanismo de in-context learning que torna few-shot útil, só que na direção oposta ao alinhamento pretendido.

### Ferramentas open
- `Ferramenta` **Llama Guard 3**. https://huggingface.co/meta-llama/Llama-Guard-3-8B
- `Ferramenta` **NVIDIA NeMo Guardrails**. https://github.com/NVIDIA/NeMo-Guardrails
- `Ferramenta` **Granite Guardian** (IBM).
- `Ferramenta` **promptfoo** com red-team modes.

### Referências
- `Paper` **Universal and Transferable Adversarial Attacks on Aligned LLMs** — Zou et al. (2023). https://arxiv.org/abs/2307.15043
- `Paper` **AdvBench**. https://github.com/llm-attacks/llm-attacks
- `Livro` **OWASP Top 10 for LLM Applications**. https://owasp.org/www-project-top-10-for-large-language-model-applications/

---

## 14.9 Red-teaming

### Definição
Tentativa estruturada e adversarial de fazer o modelo falhar (gerar conteúdo nocivo, vazar secrets, sair de papel, etc.).

### Metodologia
- **Cenários** diversos (segurança, viés, privacidade, decepção, capacidades perigosas).
- **Iteração**: descoberto → mitigado → re-testado.
- **Automated red-teaming** com LLMs.
- **Bug bounty** estilo segurança tradicional.

### Referências
- `Paper` **Red Teaming Language Models with Language Models** — Perez et al. (2022). https://arxiv.org/abs/2202.03286
- `Livro` **Anthropic — Responsible Scaling Policy**. https://www.anthropic.com/responsible-scaling-policy

---

## 14.10 Privacidade e data leakage

### Riscos
- **Memorização** de dados de treinamento (incluindo PII, código proprietário).
- **Extraction attacks**: prompts que extraem dados memorizados.
- **Inference em produção**: leak via logs de provedores.

### Defesas
- **Differential privacy** no treinamento (com custo de qualidade).
- **PII detection** e redaction.
- **Self-hosted** quando dado é sensível.
- **Auditing**: verificar memorização.

> Memorização de treino é o oposto do que geralmente se quer de um modelo (generalização, não decoreba — mod. [03](03_ml_classico.md#31-fundamentos-conceituais)), mas em modelos muito grandes com dados repetidos no corpus, alguma memorização literal é praticamente inevitável — extraction attacks exploram exatamente essa memorização residual, com prompts desenhados para "completar" um trecho memorizado exatamente como apareceu no treino.

### Referências
- `Paper` **Extracting Training Data from Large Language Models** — Carlini et al. (2020). https://arxiv.org/abs/2012.07805
- `Paper` **Scalable Extraction of Training Data from (Production) Language Models** — Nasr et al. (2023). https://arxiv.org/abs/2311.17035

---

## 14.11 Mechanistic Interpretability

### Conceito
Engenharia reversa do que **acontece dentro** dos pesos: circuitos, neurônios, attention heads.

### Por que isso importa para safety
- Detectar capacidades latentes (engano, manipulação) antes de surgirem.
- Verificar alinhamento sem só "ver outputs".

### Conceitos-chave
- **Features** (direções no espaço de ativação).
- **Circuits** (subgrafos de computação).
- **Superposition** — features sobrepostas em neurônios.
- **Sparse Autoencoders (SAEs)** — extrair features interpretáveis.
- **Activation patching**.
- **Induction heads** (cópia em context).

> **Intuição**: avaliação comportamental (seções 14.1–14.9) julga o modelo pelas *saídas* — é como avaliar um funcionário só pelo que ele entrega, sem nunca ver como ele pensa. Mechanistic interpretability tenta abrir a "caixa-preta" e entender a computação interna — quais direções no espaço de ativação (features) correspondem a conceitos específicos, e como grupos de neurônios/attention heads (circuits) implementam comportamentos específicos. Superposition é um dos achados mais importantes e contra-intuitivos da área: redes neurais representam *mais* conceitos do que têm neurônios, sobrepondo múltiplas features no mesmo neurônio de formas que só fazem sentido quando decompostas matematicamente (via Sparse Autoencoders) — um único neurônio raramente corresponde a "um conceito", ao contrário do que a intuição ingênua sugeriria.
>
> **Aplicação real**: induction heads (Olsson et al.) são um circuito identificado que implementa, mecanicamente, o padrão "se vi X seguido de Y antes, e vejo X de novo, preveja Y" — uma peça concreta e verificável do mecanismo que permite in-context learning (mod. [11](11_prompt_engineering.md#117-in-context-learning-icl-por-que-few-shot-funciona)), não apenas uma hipótese comportamental sobre o fenômeno.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre avaliar um modelo pelo comportamento (benchmarks) e avaliar pela mecânica interna (mech interp). Depois, explique com suas palavras o que "superposition" significa.

### Referências (estado da arte 2024–2025)
- `Livro` **Anthropic Transformer Circuits Thread**. https://transformer-circuits.pub/
- `Paper` **A Mathematical Framework for Transformer Circuits** — Elhage et al. (2021).
- `Paper` **In-context Learning and Induction Heads** — Olsson et al. (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- `Paper` **Toy Models of Superposition** — Elhage et al. (2022). https://transformer-circuits.pub/2022/toy_model/
- `Paper` **Towards Monosemanticity / Scaling Monosemanticity** — Anthropic. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- `Paper` **Sparse Autoencoders Find Highly Interpretable Features**. https://arxiv.org/abs/2309.08600

### Ferramentas
- `Ferramenta` **TransformerLens** (Neel Nanda). https://github.com/TransformerLensOrg/TransformerLens
- `Ferramenta` **SAELens**, **nnsight**.
- `Curso` **ARENA — Alignment Research Engineer Accelerator**. https://www.arena.education/

---

## 14.12 Frameworks de evaluation no dia a dia

- `Ferramenta` **Promptfoo** — eval declarativo, integra com CI. https://www.promptfoo.dev/
- `Ferramenta` **OpenAI Evals**. https://github.com/openai/evals
- `Ferramenta` **DeepEval**. https://github.com/confident-ai/deepeval
- `Ferramenta` **TruLens**. https://github.com/truera/trulens
- `Ferramenta` **Inspect AI** (UK AISI) — eval framework moderno. https://inspect.ai-safety-institute.org.uk/
- `Ferramenta` **lm-evaluation-harness** — para benchmarks acadêmicos.

---

## Projetos práticos

### Projeto 14.1 — Suite de eval reprodutível
- Configure `lm-evaluation-harness` em GPU local.
- Rode 3 modelos open (LLaMA 3 8B, Mistral 7B, Qwen 2.5 7B) em: MMLU, GSM8K, HumanEval.
- Compare com leaderboard público — discrepâncias geralmente revelam bugs de avaliação.

### Projeto 14.2 — LLM-as-judge calibrado
- Crie 50 pares (resp_A, resp_B) para uma tarefa de geração.
- Tenha humano (você + amigo) anotando preferência.
- Use 3 LLMs como juízes; meça concordância com humano.
- Documente position bias e self-preference.

> **Variante guiada**: rode cada par duas vezes com a ordem A/B invertida — se o juiz muda de opinião só por causa da ordem, isso quantifica diretamente o position bias, em vez de você ter que inferi-lo indiretamente.

### Projeto 14.3 — Benchmark próprio em PT-BR
- Tarefa de seu domínio (atendimento, suporte técnico, etc.).
- 100 inputs anotados.
- Rode 4 modelos (open + APIs).
- Use Promptfoo ou Inspect AI.

### Projeto 14.4 — Red-team mini
- Pegue um chatbot que você construiu (do mod. [12](12_rag.mdx) ou 13).
- 10 categorias de ataque: jailbreak, prompt injection, PII extraction, role-play malicioso, etc.
- 5 ataques por categoria.
- Adicione Llama Guard 3 e meça redução de incidentes.

### Projeto 14.5 — Hallucination benchmark
- 30 perguntas de fato (com gabarito) onde o modelo "tendência a alucinar".
- Compare: baseline, com RAG, com chain-of-verification.
- Use FactScore-style avaliação atomicizada.

> **Variante guiada**: separe suas 30 perguntas por tipo de alucinação esperada (closed-domain vs open-domain, seção 14.5) antes de rodar — isso permite verificar se a mitigação certa (RAG para closed-domain, verification para open-domain) realmente ajuda mais no tipo correspondente.

### Projeto 14.6 (avançado) — Mech interp introdutório
- Use TransformerLens com GPT-2 small.
- Reproduza experimento de "induction heads" do paper.
- Explore SAEs com SAELens.

---

## Erros comuns

- **Avaliar em 1 seed** — variância de LLM é alta; rode múltiplas seeds.
- **Confundir benchmark score com utilidade real** para *seu* uso.
- **Confiar em LLM-as-judge** sem validar contra humano.
- **Lançar produto sem red-team** — você descobrirá os ataques pela imprensa.
- **Tratar safety como gating** em vez de processo contínuo.
- **Não logar tudo** — sem trace, post-mortem é impossível.

---

## Conexão com módulos seguintes

| Conceito daqui | Aparece em |
|---|---|
| Eval frameworks | Engenharia de produção (mod. [15](15_engenharia_producao.mdx)) |
| Guardrails | Produção (mod. [15](15_engenharia_producao.mdx)) |
| Privacy considerations | Inferência local (revisar mod. [10](10_eficiencia_e_inferencia_local.md)) |
| Bias e fairness | Multimodal (mod. [18](18_multimodal.mdx)) |

---

## Checklist de saída

- [ ] Rodei lm-evaluation-harness em modelo local.
- [ ] Construí benchmark próprio com gabarito + métricas no meu domínio.
- [ ] Implementei LLM-as-judge com mitigação de bias.
- [ ] Fiz red-team estruturado em sistema próprio.
- [ ] Adicionei guardrails (Llama Guard 3 ou similar) e medi efeito.
- [ ] Conheço os principais riscos de privacidade e como mitigá-los.
- [ ] Tive primeiro contato com mech interp (TransformerLens).
