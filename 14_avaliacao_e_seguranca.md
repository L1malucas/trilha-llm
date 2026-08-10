# Módulo 14 — Avaliação e Segurança de LLMs

> **Objetivo**: avaliar LLMs cientificamente — benchmarks, LLM-as-judge, alucinação, viés, calibração — e proteger sistemas contra ataques (prompt injection, jailbreaks, data leakage). Inclui interpretabilidade mecanística como ferramenta de safety.
>
> **Pré-requisitos**: Módulos 09, 11–13.
>
> **Tempo de referência**: 3–4 semanas.

---

## Por que isso importa

"O modelo está bom" não é avaliação. Sem metodologia, decisões viram folclore. **E**: aplicação real exige garantia de safety, viés, robustez. Lançar LLM em produção sem eval e sem red-team é negligência profissional.

---

## 14.1 Benchmarks acadêmicos clássicos

### Conhecimento e raciocínio
- **MMLU (Massive Multitask Language Understanding)** — 57 áreas de conhecimento. 📄 https://arxiv.org/abs/2009.03300
- **MMLU-Pro** — versão mais difícil. 📄 https://arxiv.org/abs/2406.01574
- **BIG-Bench / BIG-Bench Hard** — 200+ tarefas diversas. 📄 https://arxiv.org/abs/2206.04615
- **HellaSwag** — commonsense reasoning. 📄 https://arxiv.org/abs/1905.07830
- **ARC (AI2 Reasoning Challenge)**. 📄 https://arxiv.org/abs/1803.05457
- **TruthfulQA** — checa veracidade contra mitos comuns. 📄 https://arxiv.org/abs/2109.07958
- **GPQA** — perguntas de pós-graduação resistentes a Google. 📄 https://arxiv.org/abs/2311.12022

### Matemática
- **GSM8K** — problemas de matemática elementar. 📄 https://arxiv.org/abs/2110.14168
- **MATH** — problemas de competição. 📄 https://arxiv.org/abs/2103.03874
- **AIME** — provas de olimpíada (em modo eval).

### Código
- **HumanEval** — completar funções Python. 📄 https://arxiv.org/abs/2107.03374
- **MBPP** — Mostly Basic Python Problems.
- **LiveCodeBench** — problemas atualizados, anti-contaminação. https://livecodebench.github.io/
- **SWE-bench / SWE-bench Verified** — bugs reais em repos. https://www.swebench.com/

### Linguagem geral
- **GLUE / SuperGLUE** — clássicos pré-LLM.
- **HELM (Holistic Evaluation of LMs)** — Stanford CRFM. 📄 https://arxiv.org/abs/2211.09110
- **LiveBench** — benchmark dinâmico. https://livebench.ai/

### Português / multilíngue
- **MMLU-PT-BR** (vários esforços de tradução).
- **BLUEX** — vestibulares brasileiros. https://github.com/Portuguese-NLP/luana-blue
- **ENEM Challenge**. https://github.com/piresramon/enem-challenge
- **MEGA** — benchmark multilíngue do Microsoft. https://arxiv.org/abs/2303.12528

### Ferramentas
- 🛠 **lm-evaluation-harness** (EleutherAI) — padrão de fato para rodar benchmarks. https://github.com/EleutherAI/lm-evaluation-harness
- 🛠 **HELM** runner. https://github.com/stanford-crfm/helm
- 🛠 **BIG-Bench** runner.

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

### Mitigações
- Randomizar ordem.
- Múltiplos juízes (ensemble).
- Rubricas estruturadas.
- Humanos no loop para calibrar.

### Referências
- 📄 **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al. (2023). https://arxiv.org/abs/2306.05685
- 📄 **Large Language Models are not Fair Evaluators** — Wang et al. (2023). https://arxiv.org/abs/2305.17926

---

## 14.3 Avaliação humana e Arena-style

- **Chatbot Arena** (LMSys) — comparação A/B humana, ranking ELO. https://arena.lmsys.org/
- **Crowdsourcing** com plataformas como Scale AI, Surge, Toloka.
- **Avaliação interna** com rubrica clara.

### Referências
- 📄 **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference** — Chiang et al. (2024). https://arxiv.org/abs/2403.04132

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

### Referências
- 📄 **Detecting Pretraining Data from LLMs (Min-K%)** — Shi et al. (2023). https://arxiv.org/abs/2310.16789
- 📄 **Investigating Data Contamination in Modern Benchmarks**. https://arxiv.org/abs/2311.09783

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

### Mitigação
- **RAG** com grounding obrigatório.
- **Constrained generation** (mod. 11).
- **Self-consistency**.
- **Chain of Verification**. 📄 https://arxiv.org/abs/2309.11495
- **Citação obrigatória** com validação.
- **Modelos com refusal calibrado** (treinados a dizer "não sei").

### Métricas
- **FactScore** — atomicidade + verificação. https://arxiv.org/abs/2305.14251
- **TruthfulQA**.
- **HaluEval**. https://arxiv.org/abs/2305.11747

### Referências
- 📄 **Survey of Hallucination in Natural Language Generation** — Ji et al. (2022). https://arxiv.org/abs/2202.03629

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

### Referências
- 📄 **On the Dangers of Stochastic Parrots** — Bender et al. (2021). https://dl.acm.org/doi/10.1145/3442188.3445922
- 📚 **Fairness and Machine Learning: Limitations and Opportunities** — Barocas, Hardt, Narayanan. https://fairmlbook.org/

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

### LLMs e calibração
LLMs grandes pré-RLHF tendem a ser bem calibrados. **RLHF tipicamente piora calibração** (modelo aprende a parecer mais confiante).

### Referências
- 📄 **Calibration of Pre-trained Transformers** — Desai & Durrett (2020). https://arxiv.org/abs/2003.07892
- 📄 **Just Ask for Calibration** — Tian et al. (2023). https://arxiv.org/abs/2305.14975

---

## 14.8 Adversarial robustness e Jailbreaks

### Tipos de ataque
- **Prompt injection** direto e indireto (revisar mod. 11).
- **Jailbreak**: induzir comportamento proibido (DAN, role-play, encoding, multi-turn).
- **Universal adversarial suffixes**. 📄 https://arxiv.org/abs/2307.15043
- **Many-shot jailbreaking** (Anthropic). https://www.anthropic.com/news/many-shot-jailbreaking
- **Crescendo** — ataque multi-turno escalado. https://arxiv.org/abs/2404.01833

### Defesas (todas parciais)
- **Input filtering**.
- **Output filtering** (Llama Guard, NeMo Guardrails).
- **Constitutional AI / RLAIF**.
- **Adversarial training**.

### Ferramentas open
- 🛠 **Llama Guard 3**. https://huggingface.co/meta-llama/Llama-Guard-3-8B
- 🛠 **NVIDIA NeMo Guardrails**. https://github.com/NVIDIA/NeMo-Guardrails
- 🛠 **Granite Guardian** (IBM).
- 🛠 **promptfoo** com red-team modes.

### Referências
- 📄 **Universal and Transferable Adversarial Attacks on Aligned LLMs** — Zou et al. (2023). https://arxiv.org/abs/2307.15043
- 📄 **AdvBench**. https://github.com/llm-attacks/llm-attacks
- 📚 **OWASP Top 10 for LLM Applications**. https://owasp.org/www-project-top-10-for-large-language-model-applications/

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
- 📄 **Red Teaming Language Models with Language Models** — Perez et al. (2022). https://arxiv.org/abs/2202.03286
- 📚 **Anthropic — Responsible Scaling Policy**. https://www.anthropic.com/responsible-scaling-policy

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

### Referências
- 📄 **Extracting Training Data from Large Language Models** — Carlini et al. (2020). https://arxiv.org/abs/2012.07805
- 📄 **Scalable Extraction of Training Data from (Production) Language Models** — Nasr et al. (2023). https://arxiv.org/abs/2311.17035

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

### Referências (estado da arte 2024–2025)
- 📚 **Anthropic Transformer Circuits Thread**. https://transformer-circuits.pub/
- 📄 **A Mathematical Framework for Transformer Circuits** — Elhage et al. (2021).
- 📄 **In-context Learning and Induction Heads** — Olsson et al. (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- 📄 **Toy Models of Superposition** — Elhage et al. (2022). https://transformer-circuits.pub/2022/toy_model/
- 📄 **Towards Monosemanticity / Scaling Monosemanticity** — Anthropic. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- 📄 **Sparse Autoencoders Find Highly Interpretable Features**. https://arxiv.org/abs/2309.08600

### Ferramentas
- 🛠 **TransformerLens** (Neel Nanda). https://github.com/TransformerLensOrg/TransformerLens
- 🛠 **SAELens**, **nnsight**.
- 🎓 **ARENA — Alignment Research Engineer Accelerator**. https://www.arena.education/

---

## 14.12 Frameworks de evaluation no dia a dia

- 🛠 **Promptfoo** — eval declarativo, integra com CI. https://www.promptfoo.dev/
- 🛠 **OpenAI Evals**. https://github.com/openai/evals
- 🛠 **DeepEval**. https://github.com/confident-ai/deepeval
- 🛠 **TruLens**. https://github.com/truera/trulens
- 🛠 **Inspect AI** (UK AISI) — eval framework moderno. https://inspect.ai-safety-institute.org.uk/
- 🛠 **lm-evaluation-harness** — para benchmarks acadêmicos.

---

## 🧪 Projetos práticos

### Projeto 14.1 — Suite de eval reprodutível
- Configure `lm-evaluation-harness` em GPU local.
- Rode 3 modelos open (LLaMA 3 8B, Mistral 7B, Qwen 2.5 7B) em: MMLU, GSM8K, HumanEval.
- Compare com leaderboard público — discrepâncias geralmente revelam bugs de avaliação.

### Projeto 14.2 — LLM-as-judge calibrado
- Crie 50 pares (resp_A, resp_B) para uma tarefa de geração.
- Tenha humano (você + amigo) anotando preferência.
- Use 3 LLMs como juízes; meça concordância com humano.
- Documente position bias e self-preference.

### Projeto 14.3 — Benchmark próprio em PT-BR
- Tarefa de seu domínio (atendimento, suporte técnico, etc.).
- 100 inputs anotados.
- Rode 4 modelos (open + APIs).
- Use Promptfoo ou Inspect AI.

### Projeto 14.4 — Red-team mini
- Pegue um chatbot que você construiu (do mod. 12 ou 13).
- 10 categorias de ataque: jailbreak, prompt injection, PII extraction, role-play malicioso, etc.
- 5 ataques por categoria.
- Adicione Llama Guard 3 e meça redução de incidentes.

### Projeto 14.5 — Hallucination benchmark
- 30 perguntas de fato (com gabarito) onde o modelo "tendência a alucinar".
- Compare: baseline, com RAG, com chain-of-verification.
- Use FactScore-style avaliação atomicizada.

### Projeto 14.6 (avançado) — Mech interp introdutório
- Use TransformerLens com GPT-2 small.
- Reproduza experimento de "induction heads" do paper.
- Explore SAEs com SAELens.

---

## ⚠️ Erros comuns

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
| Eval frameworks | Engenharia de produção (mod. 15) |
| Guardrails | Produção (mod. 15) |
| Privacy considerations | Inferência local (revisar mod. 10) |
| Bias e fairness | Multimodal (mod. 18) |

---

## Checklist de saída

- [ ] Rodei lm-evaluation-harness em modelo local.
- [ ] Construí benchmark próprio com gabarito + métricas no meu domínio.
- [ ] Implementei LLM-as-judge com mitigação de bias.
- [ ] Fiz red-team estruturado em sistema próprio.
- [ ] Adicionei guardrails (Llama Guard 3 ou similar) e medi efeito.
- [ ] Conheço os principais riscos de privacidade e como mitigá-los.
- [ ] Tive primeiro contato com mech interp (TransformerLens).
