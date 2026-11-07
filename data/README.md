# data/

- `raw/<dataset_id>/` downloads and extracted files, plus `.dataset_status.json` (never edited by the pipeline after extraction).
- `manifests/` normalised manifests (`<id>.jsonl`), build metadata (`.meta.json`), validation reports, and `active.json`.
- `processed/` optional resized copies from `scripts.prepare_images`.
- `knowledge/` your own permitted documents for retrieval (`.md`, `.txt`, `.jsonl`).
- `sample/` reserved for small fixtures.

Generated content is git-ignored. Create the synthetic demo with `python -m scripts.generate_demo_data`.
