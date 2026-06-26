# .nls format spec

Memory artifact schema (not model weights).

| File | Purpose |
|------|---------|
| [manifest.schema.json](manifest.schema.json) | JSON Schema for manifest header |
| [validate.py](validate.py) | CLI validator |
| [EXAMPLES.md](EXAMPLES.md) | Sample manifests |

Reference implementation: `pri/format.py`.

```bash
python spec/validate.py /data/pri/snapshot/captures/
```
