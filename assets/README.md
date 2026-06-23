# Brand assets

Official Punk Records branding from the hosted product frontend
(`NLS/punk-records/frontend/public/`).

| File | Use |
|------|-----|
| `logo.png` | Hex vinyl/circuit mark — **README header** (matches frontend) |
| `logo-256.png` | Scaled logo for docs |
| `straw-hat.png` | Straw hat illustration (source — for manual design if needed) |
| `social-preview.png` | GitHub social preview (1280×640, hex logo + text) |
| `banner.png` | Alias of `social-preview.png` |
| `favicon.ico` | Browser favicon |

The README uses **`logo.png` only** (no programmatic hat overlay). The frontend does the same; hat compositing in raster assets looked wrong at small sizes.

Regenerate banners:

```bash
pip install pillow
python scripts/build_logo_assets.py
```

### GitHub social preview

GitHub has no API for this — upload manually or use `scripts/upload_social_preview.py`.

**Private repo note:** **Social preview** appears on **public** repos (or private repos that already had one). While private, the Settings section is often hidden.

When public: **Settings** → **Social preview** → upload `assets/social-preview.png`

Verify:

```bash
gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
```
