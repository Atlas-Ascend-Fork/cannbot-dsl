# Release and deployment

The target domain is `cannbot-dsl.gitcode.com`, and the source repository is `https://gitcode.com/cann/cannbot-dsl`. Until hosting is configured, preview the generated site on a development machine or server.

## Generate static files

```bash
cd docs
npm install
npm run build
```

The output is written to `docs/.vitepress/dist/`. The deployment system should publish this directory as the site root.

## Preview the production build

```bash
cd docs
npm run preview
```

The preview server listens on `127.0.0.1:4174`. On a remote server, use SSH local port forwarding to inspect the pages in a local browser.

## Deploy under a subpath

The default base is the domain root. Set `DOCS_BASE` when deploying under a temporary subpath:

```bash
DOCS_BASE=/preview/ npm run build
```


## Build channels

The repository keeps one documentation source. The development site is built from `main`; a commercial site is built from a frozen release branch or Git tag.

```bash
# Development preview
DOCS_VERSION=next npm run build

# Example commercial build
DOCS_BASE=/v0.1.0/ DOCS_VERSION=v0.1.0 npm run build
```

Both builds use the same API page paths. The source ref and the site base path identify the version. Never rewrite an already published version; release fixes as a new patch version.
