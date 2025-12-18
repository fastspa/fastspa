import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {Highlight, themes} from 'prism-react-renderer';
import styles from './index.module.css';

const FeatureList = [
  {
    title: 'Matrix Analysis',
    description: 'Perform sector-specific or economy wide analysis to identify key contributing sectors and pathways.',
    image: 'img/matrix%20Small.jpeg',
  },
  {
    title: 'Visualisation',
    description: 'Explore results with interactive visualisations with icicle plots.',
    image: 'img/visualisation.gif',
  },
  {
    title: 'Multi-Satellite Support',
    description: 'Analyze multiple environmental flows (GHG, water, energy) simultaneously.',
  },
];

function Feature({title, description, image}) {
  const imageUrl = useBaseUrl(image);
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        {image && <img src={imageUrl} alt={title} style={{maxWidth: '100%', height: 'auto', marginTop: '1rem'}} />}
      </div>
    </div>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const logoUrl = useBaseUrl('/img/logo.webp');
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <img
          src={logoUrl}
          alt="FastSPA Logo"
          className={styles.heroLogo}
        />
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{marginLeft: '1rem'}}
            href="https://pypi.org/project/fastspa/">
            pip install fastspa
          </Link>
        </div>
      </div>
    </header>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CodeExample() {
  const codeExample = `from fastspa import SPA

# Your A-matrix and direct intensities
paths = SPA(A, emissions).analyze(sector=42, depth=8)

# Explore results
for path in paths.top(10):
    print(f"{path.contribution:.2%}: {' → '.join(path.sectors)}")

# Export to DataFrame
df = paths.to_dataframe()`;

  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2" className="text--center">Quick Example</Heading>
            <Highlight theme={themes.nightOwl} code={codeExample} language="python">
              {({className, style, tokens, getLineProps, getTokenProps}) => (
                <pre className={clsx(className, styles.codeBlock)} style={style}>
                  {tokens.map((line, i) => (
                    <div key={i} {...getLineProps({line, key: i})}>
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({token, key})} />
                      ))}
                    </div>
                  ))}
                </pre>
              )}
            </Highlight>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Modern Python SPA for EEIO"
      description={siteConfig.customFields.description}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <CodeExample />
      </main>
    </Layout>
  );
}
