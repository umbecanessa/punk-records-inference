# Brand assets

Official Punk Records branding.

| File | Use |
|------|-----|
| `logo-composite.png` | Hex + straw hat, **transparent background**, 512×512 — README |
| `logo-composite-256.png` | Scaled composite |
| `logo.png` | Hex mark source |
| `straw-hat.png` | Straw hat illustration (source — for manual design if needed) |
| `social-preview.png` | GitHub social preview (1280×640, hex logo + text) |
| `banner.png` | Alias of `social-preview.png` |
| `favicon.ico` | Browser favicon |

The README uses **`logo-composite.png`** — generated from `logo.png` + `straw-hat.png`.

Regenerate banners:

```bash
pip install pillow
python scripts/build_logo_assets.py
```

### GitHub social preview

GitHub has no API for this — upload manually or use `scripts/upload_social_preview.py`.

**Settings** → **Social preview** → upload `assets/social-preview.png`

Verify after upload:

```bash
gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
```

See [docs/ANNOUNCE.md](../docs/ANNOUNCE.md) for the full public-release checklist.
