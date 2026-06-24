# Brand assets

| File | Use |
|------|-----|
| `logo-composite.png` | Hex + straw hat, transparent background, 512×512 — README |
| `logo-composite-256.png` | Scaled composite |
| `logo.png` | Hex mark source |
| `straw-hat.png` | Straw hat illustration source |
| `social-preview.png` | GitHub social preview (1280×640) |
| `banner.png` | Alias of `social-preview.png` |
| `favicon.ico` | Browser favicon |

Regenerate:

```bash
pip install pillow
python scripts/build_logo_assets.py
```

## GitHub social preview

**Settings** → **Social preview** → upload `assets/social-preview.png`

Verify:

```bash
gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
```
