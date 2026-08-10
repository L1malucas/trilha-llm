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

### A2C / A3C
Asynchronous Advantage Actor-Critic.
- `Paper` **Asynchronous Methods for Deep RL** — Mnih et al. (2016). https://arxiv.org/abs/1602.01783

### TRPO
Trust Region Policy Optimization — restrição em KL para estabilidade.
- `Paper` **Trust Region Policy Optimization** — Schulman et al. (2015). https://arxiv.org/abs/1502.05477

### PPO (Proximal Policy Optimization)
**O algoritmo padrão moderno**. Simplificação de TRPO usando clipping. Usado em RLHF, jogos, robótica.
- `Paper` **Proximal Policy Optimization Algorithms** — Schulman et al. (2017). https://arxiv.org/abs/1707.06347

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
