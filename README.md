# Trilha LLM

Currículo de estudo em LLMs, IA e Engenharia de Sistemas Inteligentes — 21 módulos em Markdown, publicados como site com [Docusaurus](https://docusaurus.io/).

Site publicado: https://L1malucas.github.io/trilha-llm/

## Desenvolvimento local

```bash
npm install
npm run start
```

Abre um servidor local com live reload em `http://localhost:3000`.

## Build

```bash
npm run build
```

Gera o site estático em `build/`.

```bash
npm run serve
```

Serve o build gerado localmente, para conferir antes de publicar.

## Deploy

O deploy é automático: a cada push na branch `main`, o workflow `.github/workflows/deploy.yml` builda o site e publica no GitHub Pages via GitHub Actions.
