# Contracts

These schemas are the frozen boundary for the first parallel development wave.

- `project.schema.json`: persistent project state.
- `vision-result.schema.json`: common envelope for automated stages.
- `model-manifest.schema.json`: provenance and license contract for every model.

Feature sessions must consume these contracts or use fakes. A contract change
requires explicit approval and a versioned migration.
