// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    'installation',
    'quick-start',
    'user-stories',
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/analysis-modes',
        'concepts/path-objects',
        'concepts/pathcollection',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/multi-satellite',
        'guides/concordance',
        'guides/streaming',
        'guides/loading-data',
        'guides/semantic-aggregation',
        'guides/network-analysis',
        'guides/uncertainty',
        'guides/consequential-mode',
        'guides/temporal-analysis',
        'guides/loop-analysis',
        'guides/visualization',
        'guides/sector-adjustments',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/spa-class',
        'api/factory-functions',
        'api/data-classes',
        'api/ipf',
      ],
    },
  ],
};

export default sidebars;
