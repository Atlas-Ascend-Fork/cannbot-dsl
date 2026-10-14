# Documentation versioning

CANNBot-DSL maintains one documentation source and produces two build channels.

## Development preview

- Source: `main`.
- Version label: `next`.
- Purpose: review the latest documentation and in-progress APIs.
- The preview changes over time and is not a commercial reference.

## Commercial release

- Source: a frozen `release/x.y` branch or `vX.Y.Z` tag.
- Version label: the corresponding `vX.Y.Z`.
- Purpose: provide a reproducible, traceable documentation set.
- Published versions are immutable; fixes create a new patch version.

## API paths

The API categories remain Host, Kernel, and AI CPU. A version build changes only the site base path; relative API paths remain stable:

```text
next/zh/api/host/...
v0.1.0/zh/api/host/...
```
