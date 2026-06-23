# Brand assets

Official Punk Records branding from the hosted product frontend
(`NLS/punk-records/frontend/public/`).

| File | Use |
|------|-----|
| `logo.png` | Hex vinyl/circuit mark (source) |
| `straw-hat.png` | Straw hat overlay (source) |
| `logo-mark.png` | Composite mark (1024×1024) |
| `logo-mark-256.png` | README / docs header |
| `banner.png` | Alias of `social-preview.png` (1280×640) |
| `social-preview.png` | **GitHub social preview** — upload this in repo Settings |
| `favicon.ico` | Browser favicon |

Regenerate composites after updating source PNGs:

```bash
pip install pillow
python scripts/build_logo_assets.py
```

### GitHub social preview

GitHub has no API for this — upload manually or automate with Playwright:

1. **Manual:** Repo → **Settings** → **Social preview** → **Edit** → upload `assets/social-preview.png`
   - Or run: `gh browse --settings -R umbecanessa/punk-records-inference`
2. **Automated:** `python scripts/upload_social_preview.py` (requires Chrome CDP or `--headed`)

Verify:

```bash
gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
```
