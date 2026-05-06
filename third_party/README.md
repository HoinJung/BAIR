# Third-Party Repositories

This directory is intentionally ignored by git. Clone external model code here when a model cannot be loaded from plain Hugging Face classes.

Expected optional checkouts:

- `third_party/EarthDial`: required for `akshaydudhane/EarthDial_4B_RGB`
- `third_party/GeoChat`: required by SkySense/GeoChat-style checkpoints
- `third_party/GeoPix`: optional NWPU GeoPix support
- `third_party/RemoteCLIP`: optional RemoteCLIP reference code for rebuilding NWPU retrieval metadata
- `third_party/SkySense-Chat`: optional upstream SkySense reference scripts

Run `scripts/clone_external_repos.sh` from the repository root to create these folders.
