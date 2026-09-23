# RCSWX Table 4.2 handoff bundle

Start with **HANDOFF.md**. It supersedes the earlier provisional benchmark handoff.

```bash
python verify_bundle.py
(cd fixtures && python -m unittest -v test_recovery)
```

The latest coauthor ResNet/Mixer definitions are fixed; no reconstruction is required. The four pairings and historical targets are in `targets.json`. `campaign.example.json` is a proposed execution policy, not a runnable configuration for the existing repository.

`fixtures/` preserves the earlier static recovery package. Its README is recovery provenance, not a second execution specification; HANDOFF.md controls the benchmark task.

All four source operation counts match the screenshot (267/499/264/392). Runtime construction, PyTorch execution, Python/Rust equivalence and performance have not been measured in this bundle. Historical algorithm revision, effective YAML/configuration, corner-collapse flag and timing boundary remain unverified.
