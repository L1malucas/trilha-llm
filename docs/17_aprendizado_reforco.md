---
id: 17_aprendizado_reforco
title: "Módulo 17 — Aprendizado por Reforço"
sidebar_position: 17
---

# Módulo 17 — Aprendizado por Reforço

> **Objetivo**: dominar RL clássico (MDPs, Q-learning, policy gradient) e moderno (DQN, PPO, AlphaZero, RL para LLMs). Conexão com RLHF/GRPO do módulo [09](09_treinamento_e_alinhamento.mdx).
>
> **Pré-requisitos**: Módulos [01](01_matematica.md) (matemática, especialmente prob/stat e otimização), 05 (DL).
>
> **Tempo de referência**: 5–8 semanas.

## Objetivos de aprendizagem

Ao final deste módulo você deve ser capaz de:

- Formalizar um problema como MDP (estado, ação, recompensa, transição) e explicar a equação de Bellman com intuição, não só a fórmula.
- Explicar a diferença entre métodos value-based (Q-learning, DQN) e policy-based (REINFORCE, PPO).
- Explicar por que PPO usa "clipping" e o que isso evita.
- Explicar por que DPO (mod. 09) pode ser entendido como uma forma de offline RL.
- Escolher entre RL online, offline e RLHF/GRPO para um cenário de treinamento de LLM.

---

## Por que isso importa

RL é o que treina:
- Modelos de raciocínio (DeepSeek-R1, o1).
- Agentes que aprendem em ambiente (jogos, robótica).
- Sistemas com **sequência de decisões** com recompensa atrasada.

Mesmo se você nunca treinar um agente RL puro, **RLHF/GRPO usa as mesmas ferramentas matemáticas**. Sem isso, não entende o que acontece em alinhamento.

---

## 17.1 Formalismo: Markov Decision Process (MDP)

### Componentes
- **Estado** \(S\).
- **Ação** \(A\).
- **Transição** \(P(s' | s, a)\).
- **Recompensa** \(R(s, a, s')\).
- **Desconto** \(\gamma \in [0, 1]\).
- **Política** \(\pi(a | s)\).

### Quantidades-chave
- **Retorno** \(G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\).
- **Value function** \(V^\pi(s) = \mathbb{E}_\pi[G_t | S_t=s]\).
- **Action-value (Q-function)** \(Q^\pi(s, a)\).
- **Equação de Bellman**.

> **Intuição**: um MDP formaliza "tomar decisões em sequência, cada uma afetando o futuro". Estado é "onde você está agora"; ação é "o que você pode fazer"; a propriedade de Markov diz que o futuro depende só do estado atual, não de como chegou ali (o histórico completo não importa, só a situação presente); recompensa é o sinal de feedback, frequentemente atrasado (uma ação boa agora pode só compensar muitos passos depois). O desconto `γ` resolve um problema prático: sem ele, recompensas infinitas no futuro tornariam o retorno total indefinido; `γ < 1` faz recompensas futuras valerem progressivamente menos que recompensas imediatas — parecido com preferir dinheiro hoje a dinheiro daqui a 10 anos.
>
> A **equação de Bellman** é a peça central: ela diz que o valor de um estado é a recompensa imediata mais o valor (descontado) de onde você vai parar depois — `V(s) = E[R + γV(s')]`. É uma definição *recursiva*, e essa recursão é exatamente o que torna RL computacionalmente tratável: em vez de precisar simular o futuro infinito para saber o valor de um estado, você pode calcular valores por bootstrapping (usando a estimativa atual do valor do próximo estado para atualizar a estimativa do estado atual), que é a ideia central por trás de Temporal Difference learning (seção 17.2).
>
> **Checkpoint**: sem olhar o texto, explique a propriedade de Markov com suas palavras. Depois, explique por que a equação de Bellman ser recursiva é útil computacionalmente.

### Referências (canônicas)
- `Livro` **Reinforcement Learning: An Introduction** — Sutton & Barto. Gratuito. http://incompleteideas.net/book/the-book-2nd.html
- `Curso` **DeepMind x UCL — Reinforcement Learning Course** (David Silver). `Gratuito` Vídeos gratuitos. https://www.davidsilver.uk/teaching/
- `Curso` **UC Berkeley CS285 — Deep RL** (Sergey Levine). https://rail.eecs.berkeley.edu/deeprlcourse/

---

## 17.2 RL clássico tabular

### Métodos
- **Dynamic Programming**: policy iteration, value iteration (assume modelo conhecido).
- **Monte Carlo methods**: aprendem por amostragem de episódios completos.
- **Temporal Difference (TD)**: bootstrap, atualização passo a passo.
- **SARSA** (on-policy TD).
- **Q-learning** (off-policy TD).
- **Eligibility traces** (TD(λ)).

> **Intuição — Monte Carlo vs TD**: Monte Carlo espera o episódio *terminar* pra então atualizar os valores com o retorno real observado — preciso, mas lento (só aprende depois de completar uma partida inteira, por exemplo) e alta variância. TD atualiza a cada passo, usando a estimativa atual do próximo estado como aproximação do retorno futuro (bootstrapping, via Bellman) — mais rápido para aprender, funciona até em tarefas contínuas sem fim natural, mas introduz viés (a estimativa usada pode estar errada no início do treino). Q-learning é *off-policy*: aprende sobre a melhor ação possível (`max` sobre ações), mesmo enquanto explora com uma política diferente (ex.: aleatória parte do tempo) — permite aprender sobre a política ótima enquanto ainda explora o ambiente.
>
> **Checkpoint**: sem olhar o texto, explique a diferença entre Monte Carlo e TD learning — qual espera o episódio terminar?

### Da matemática ao código
Implemente em NumPy:
- Value iteration em Gridworld.
- Q-learning em Frozen Lake.
- Visualize convergência das funções de valor.

### Referências
- `Livro` Sutton & Barto, caps. 3–7.

---

## 17.3 Deep RL: combinando RL com redes neurais

### DQN (Deep Q-Network)
Generalização de Q-learning com rede neural aproximando Q.
- `Paper` **Playing Atari with Deep Reinforcement Learning** — Mnih et al. (2013). https://arxiv.org/abs/1312.5602
- `Paper` **Human-level control through deep RL (Nature)** — Mnih et al. (2015). https://www.nature.com/articles/nature14236

### Truques que tornaram DQN viável
- **Experience replay** (descorrelacionar amostras).
- **Target network** (estabilidade).
- **Reward clipping**.
- **Frame stacking**.

> **Intuição**: treinar uma rede neural (mod. [05](05_deep_learning.md#53-otimização-e-regularização-para-dl)) assume implicitamente que os exemplos de treino são razoavelmente independentes — mas em RL, experiências consecutivas (estado, ação, recompensa, próximo estado) são altamente correlacionadas (o estado no passo `t+1` depende diretamente do estado em `t`). Treinar diretamente nessa sequência correlacionada desestabiliza o treino. Experience replay quebra essa correlação: guarda experiências num buffer e amostra *aleatoriamente* dele para cada passo de treino, como embaralhar um dataset antes de cada época. Target network resolve um problema relacionado — usar a mesma rede que está sendo atualizada como alvo do próprio treino (bootstrapping) cria um "alvo em movimento" instável; congelar uma cópia (target network) por um período e só atualizá-la periodicamente estabiliza o treino.

### Variações
- **Double DQN**. `Paper` https://arxiv.org/abs/1509.06461
- **Dueling DQN**. `Paper` https://arxiv.org/abs/1511.06581
- **Prioritized Experience Replay**. `Paper` https://arxiv.org/abs/1511.05952
- **Rainbow** — combina tudo. `Paper` https://arxiv.org/abs/1710.02298

---

## 17.4 Policy Gradient methods

### Conceito
Em vez de aprender Q e derivar política, aprenda diretamente a política \(\pi_\theta\).

### REINFORCE
Algoritmo básico de policy gradient. Estima gradiente via amostragem.

### Actor-Critic
Combinar policy network (actor) + value network (critic) para reduzir variância.

> **Intuição — value-based vs policy-based**: Q-learning/DQN aprendem "o quanto vale cada ação em cada estado" e a política emerge implicitamente (escolha a ação de maior valor). Policy gradient aprende a política *diretamente* — uma rede que recebe estado e produz uma distribuição de probabilidade sobre ações, ajustada via gradiente pra aumentar a probabilidade de ações que levaram a bom retorno. Isso é mais natural para ações contínuas (robótica — não dá pra enumerar "o valor de cada ação possível" quando ações são um vetor contínuo) e permite políticas estocásticas genuínas (útil quando a estratégia ótima envolve aleatoriedade, como blefar num jogo). REINFORCE tem alta variância porque estima o gradiente a partir de retornos de episódios completos amostrados — Actor-Critic reduz essa variância usando uma rede "crítica" que estima o valor esperado, permitindo comparar "esse retorno foi melhor ou pior que o esperado" (a vantagem) em vez do retorno bruto, um sinal de treino mais estável.

### A2C / A3C
Asynchronous Advantage Actor-Critic.
- `Paper` **Asynchronous Methods for Deep RL** — Mnih et al. (2016). https://arxiv.org/abs/1602.01783

### TRPO
Trust Region Policy Optimization — restrição em KL para estabilidade.
- `Paper` **Trust Region Policy Optimization** — Schulman et al. (2015). https://arxiv.org/abs/1502.05477

### PPO (Proximal Policy Optimization)
**O algoritmo padrão moderno**. Simplificação de TRPO usando clipping. Usado em RLHF, jogos, robótica.
- `Paper` **Proximal Policy Optimization Algorithms** — Schulman et al. (2017). https://arxiv.org/abs/1707.06347

> **Intuição — por que "clipping"**: um passo de gradiente grande demais numa política pode mudá-la drasticamente de uma vez — e diferente de supervised learning, onde um passo ruim só piora a loss temporariamente, em RL um passo ruim pode levar o agente a uma região do espaço de políticas da qual é difícil se recuperar (a política nova explora de forma tão diferente que os dados coletados com ela deixam de ser úteis para continuar melhorando). TRPO resolve isso restringindo formalmente o quanto a política pode mudar por passo (via KL-divergence, mod. [01](01_matematica.md#13-probabilidade-e-estatística)) — mas essa restrição é computacionalmente cara. PPO simplifica: em vez de uma restrição matemática rígida, "recorta" (clip) o quanto o gradiente pode empurrar a política pra mudar por passo — muito mais barato de computar, com estabilidade prática similar. É essa simplicidade e estabilidade que tornou PPO o algoritmo padrão tanto em RL clássico quanto em RLHF (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)).
>
> **Checkpoint**: sem olhar o texto, explique por que um passo de gradiente grande demais é mais perigoso em RL do que em supervised learning comum.

### GRPO (Group Relative Policy Optimization)
Variante usada no DeepSeek-R1/Math, sem value model separado.
- `Paper` **DeepSeekMath**. https://arxiv.org/abs/2402.03300

### DDPG, TD3, SAC
Para ações contínuas (robótica).
- `Paper` **DDPG** — Lillicrap et al. (2015). https://arxiv.org/abs/1509.02971
- `Paper` **Soft Actor-Critic (SAC)**. https://arxiv.org/abs/1801.01290

---

## 17.5 Model-based RL e planning

### Conceito
Aprender modelo do ambiente (\(P, R\)), planejar dentro dele.

### Métodos
- **Dyna-Q**.
- **MuZero** — planeja em modelo aprendido. `Paper` https://arxiv.org/abs/1911.08265
- **Dreamer V3** — model-based moderno. `Paper` https://arxiv.org/abs/2301.04104

### Por que importa
Eficiência de amostra (sample efficiency) é frequentemente 10–100× melhor.

> Model-free (DQN, PPO) aprende só de experiência real, que é cara de coletar (cada passo exige interagir com o ambiente de verdade). Model-based aprende um "simulador" aproximado do ambiente e pode planejar/treinar dentro desse simulador, gerando experiência sintética barata — o ganho de sample efficiency vem daí, ao custo de depender da qualidade do modelo aprendido.

---

## 17.6 Marcos históricos: AlphaGo, AlphaZero, MuZero

- `Paper` **Mastering the game of Go with deep neural networks (AlphaGo)** — Silver et al. (2016). https://www.nature.com/articles/nature16961
- `Paper` **Mastering the game of Go without human knowledge (AlphaGo Zero)**. https://www.nature.com/articles/nature24270
- `Paper` **A general reinforcement learning algorithm (AlphaZero)** — Silver et al. (2017). https://arxiv.org/abs/1712.01815
- `Paper` **MuZero** — Schrittwieser et al. (2019). https://arxiv.org/abs/1911.08265

### Lições
- **MCTS (Monte Carlo Tree Search)** + DL é poderoso.
- Self-play como caminho para superar humanos sem dados humanos.

---

## 17.7 Multi-agent RL e RL em ambientes complexos

- **MARL (Multi-Agent RL)**: cooperação, competição, mixed.
- **PettingZoo** — biblioteca para envs multi-agente. https://pettingzoo.farama.org/
- **Procgen**, **Crafter** — benchmarks para generalização.
- **MineRL**, **Crafter** — Minecraft/sobrevivência.
- **Habitat 2.0**, **Isaac Gym/Lab** — robótica.

---

## 17.8 RLHF, RLAIF e RL em LLMs

Recapitular do mod. [09](09_treinamento_e_alinhamento.mdx) com base teórica agora consolidada.

### Pipeline RLHF
1. SFT base.
2. Reward Model treinado em pares de preferência.
3. PPO contra o RM, com KL-penalty contra SFT inicial.

### RLAIF
RL com feedback de **outro LLM** (constitutional AI).

### Reasoning RL (R1-style)
Recompensa **direta** em corretude (math, code) — sem reward model.
- Usa GRPO frequentemente.
- Emerge "chain-of-thought" longo organicamente.

> Com o formalismo deste módulo, o pipeline RLHF do mod. 09 fica mais claro: o "estado" é o texto gerado até agora, a "ação" é o próximo token, a "recompensa" vem do reward model (ou, em reasoning RL, de um verificador automático de corretude), e PPO/GRPO são os mesmos algoritmos da seção 17.4, aplicados a esse MDP específico onde o "ambiente" é simplesmente o processo de geração de texto token a token.

### Referências
- `Paper` **InstructGPT** — Ouyang et al. (2022). https://arxiv.org/abs/2203.02155
- `Paper` **DPO**, **GRPO**, **KTO**, **ORPO** — revisar mod. [09](09_treinamento_e_alinhamento.mdx).

---

## 17.9 Offline RL

### Definição
Aprender política de dataset fixo, sem interagir com ambiente.

### Por que importa
- Robótica/medicina onde explorar é caro/perigoso.
- Aproveitamento de logs existentes.
- Conexão com LLMs (DPO é, em essência, offline RL).

### Métodos
- **Conservative Q-Learning (CQL)**.
- **BCQ**, **IQL**.
- **Decision Transformer** — RL como modelagem de sequência. `Paper` https://arxiv.org/abs/2106.01345

> **Intuição**: RL online (PPO, DQN) coleta experiência nova a cada passo de treino, interagindo com o ambiente real. Offline RL aprende só do que já está registrado num dataset fixo, sem poder testar ações novas no ambiente real — o desafio central é que a política aprendida pode "superestimar" o valor de ações raras ou ausentes no dataset (nunca foram testadas de verdade, então não há como saber se são boas). Essa é exatamente a situação de DPO (mod. [09](09_treinamento_e_alinhamento.mdx#95-alinhamento-via-preferências)): o modelo aprende de um dataset fixo de pares de preferência, sem gerar e testar novas respostas em tempo real contra um reward model — é offline RL aplicado a alinhamento de LLM, o que explica por que DPO é mais estável (mas também mais limitado pela qualidade/cobertura do dataset de preferências) que RLHF via PPO online.

---

## 17.10 Bibliotecas e ambientes

### Ambientes
- `Ferramenta` **Gymnasium** (sucessor do OpenAI Gym). https://gymnasium.farama.org/
- `Ferramenta` **PettingZoo** (multi-agente).
- `Ferramenta` **MuJoCo** (física continua, hoje free). https://mujoco.org/
- `Ferramenta` **Isaac Lab / Isaac Gym** (NVIDIA, robótica em GPU).
- `Ferramenta` **MineRL**, **Crafter**, **Procgen**, **Atari (ALE)**.

### Frameworks de algoritmos
- `Ferramenta` **Stable-Baselines3** — PPO, SAC, DQN, etc., em PyTorch. https://stable-baselines3.readthedocs.io/
- `Ferramenta` **CleanRL** — implementações single-file pedagógicas. https://docs.cleanrl.dev/
- `Ferramenta` **RLlib (Ray)** — produção, distributed. https://docs.ray.io/en/latest/rllib/
- `Ferramenta` **TorchRL** (PyTorch oficial). https://pytorch.org/rl/
- `Ferramenta` **TRL (Hugging Face)** — RL para LLMs. https://huggingface.co/docs/trl

### Por que CleanRL é especial para aprender
Cada algoritmo em **um único arquivo curto, sem abstrações ocultas**. Excelente didática.

---

## Projetos práticos

### Projeto 17.1 — Q-learning tabular do zero
- Em NumPy puro.
- Frozen Lake, Cliff Walking.
- Plote curvas de aprendizado, visualize política aprendida.

> **Variante guiada**: comece com Value Iteration (o ambiente é totalmente conhecido, mais simples de verificar contra a solução ótima calculável à mão em Gridworld pequeno) antes de Q-learning (que aprende só por interação, sem conhecer o modelo do ambiente) — assim você tem uma referência confiável pra confirmar que o Q-learning convergiu certo.

### Projeto 17.2 — DQN no Atari (ou CartPole)
- Implemente do zero seguindo CleanRL como referência (mas **escreva sozinho**).
- CartPole para validar; Atari Pong/Breakout para experiência real.

### Projeto 17.3 — PPO em ambiente contínuo
- Use Stable-Baselines3 ou CleanRL.
- Ambientes: Pendulum, BipedalWalker, ou MuJoCo Hopper.
- Plote training curves.

### Projeto 17.4 — Reproduzir RLHF mini
- Modelo pequeno (TinyLlama, Phi-3 mini).
- Dataset de preferências (UltraFeedback subset).
- Treine RM + PPO via TRL.
- Compare com DPO no mesmo dataset.

### Projeto 17.5 — Reasoning RL com GRPO
- Use TRL com GRPO.
- Tarefa verificável: aritmética simples ou problemas tipo GSM8K.
- Recompensa: correção da resposta final.
- Observe emergência de chain-of-thought longo.

### Projeto 17.6 — MCTS + UCT em jogo simples
- Implemente MCTS para Tic-Tac-Toe ou Connect Four.
- Sem rede neural — entender o algoritmo.
- Bônus: combinar com DL (mini-AlphaZero).

### Projeto 17.7 — Decision Transformer
- Reproduza experimento do paper em ambiente simples.
- Compare com PPO no mesmo ambiente.

---

## Erros comuns

- **Achar que policy = Q-function aprendida**: confunde value-based com policy-based.
- **Não normalizar advantages** em PPO/A2C — instabilidade.
- **Esquecer de detach value targets** — dupla retropropagação errada.
- **Hiperparâmetros muito sensíveis** — RL é notoriamente difícil de tunar; CleanRL tem boas defaults.
- **Comparar runs com 1 seed** — variância gigante em RL; reporte ≥3.
- **Confundir RLHF/DPO com RL "puro"**. DPO é supervisionado, RLHF é PPO.

---

## Conexão com outros módulos

| Conceito daqui | Aparece em |
|---|---|
| PPO, GRPO | RLHF, reasoning RL (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Reward shaping | Fine-tuning para tasks específicas |
| MCTS + RL | Agentes com planejamento (mod. [13](13_agentes_tools_protocolos.md)) |
| Offline RL / DPO | Alinhamento (mod. [09](09_treinamento_e_alinhamento.mdx)) |
| Decision Transformer | Tópicos avançados (mod. [19](19_topicos_avancados.md)) |

---

## Checklist de saída

- [ ] Implementei Q-learning tabular do zero.
- [ ] Implementei DQN do zero (com replay buffer e target network).
- [ ] Treinei PPO em ambiente contínuo com sucesso.
- [ ] Reproduzi RLHF/DPO ponta-a-ponta em modelo pequeno.
- [ ] Sei explicar diferença entre on-policy, off-policy, offline RL.
- [ ] Entendo as conexões matemáticas entre RLHF, DPO, PPO, GRPO.
