# ReliableVision engineering rules

- Work only inside this workspace. Never commit data, weights, secrets, local environments, or generated runs.
- Implement one milestone at a time; obtain user approval before starting the next.
- Keep training, evaluation and serving separate. Use Python 3.11+, types and configuration files.
- Fit the memory bank on nominal training data only. Reserve nominal validation data for thresholds; never tune on final test data.
- Scores are distances, not probabilities. Label synthetic smoke data and simulated traffic explicitly.
- Seed random generators, record configuration, content hashes, environment and model provenance.
- Run relevant tests and lint before committing. Report actual outcomes and external blockers honestly.
- CPU smoke tests must not download datasets or pretrained weights. Full experiments use pretrained features and CUDA when available.
- Respect MVTec licensing. No automatic dataset download or redistribution.
- Update LEARNING.md and milestone reports with five interview questions per milestone.
