# Contributing

Use Issues and merge requests to participate in the project. Keep the scope of each change clear and add the necessary tests or documentation for behavior changes.

## Workflow

1. Fork the repository and create a branch from the latest target branch.
2. Make the code or documentation change.
3. Run the relevant tests and repository checks.
4. Remove build products, caches, and local configuration from the change.
5. Open a merge request that explains the problem, result, and validation.

## Documentation changes

After changing `docs/`, run `npm install` and `npm run build` there. Start the production preview after a successful build and check navigation, content, and code blocks on desktop and narrow screens.

## Commit messages

Commit messages should describe the actual change. A merge request description should include the motivation, final behavior, and validation result.
