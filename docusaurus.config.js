// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const GITHUB_USER = 'L1malucas';
const REPO_NAME = 'trilha-llm';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Trilha LLM',
  tagline: 'Currículo de estudo em LLMs, IA e Engenharia de Sistemas Inteligentes',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: `https://${GITHUB_USER}.github.io`,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: `/${REPO_NAME}/`,

  // GitHub pages deployment config.
  organizationName: GITHUB_USER, // Usually your GitHub org/user name.
  projectName: REPO_NAME, // Usually your repo name.

  onBrokenLinks: 'throw',

  // Content is plain Markdown (LaTeX-style math, pseudo-code with `<`/`>`/`{}`),
  // not written for MDX/JSX — parse as CommonMark instead of MDX.
  markdown: {
    format: 'md',
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'pt-BR',
    locales: ['pt-BR'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: `https://github.com/${GITHUB_USER}/${REPO_NAME}/tree/main/docs/`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'Trilha LLM',
        logo: {
          alt: 'Trilha LLM Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Trilha',
          },
          {
            type: 'docsVersionDropdown',
            position: 'left',
          },
          {
            href: `https://github.com/${GITHUB_USER}/${REPO_NAME}`,
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Trilha',
            items: [
              {
                label: 'Início',
                to: '/',
              },
            ],
          },
          {
            title: 'Mais',
            items: [
              {
                label: 'GitHub',
                href: `https://github.com/${GITHUB_USER}/${REPO_NAME}`,
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Trilha LLM. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
