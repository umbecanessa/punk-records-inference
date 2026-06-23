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

Regenerate composites after updating source PNGs (hat anchors to detected hex top-left vertex):

```bash
pip install pillow numpy
python scripts/build_logo_assets.py
```

### GitHub social preview

GitHub has no API for this — upload manually or automate with Playwright.

**Private repo note:** GitHub only shows **Social preview** on **public** repositories (or private repos that already had a preview uploaded before). While the repo stays private, the Settings section is often hidden — that is expected. The image is still only used when links are shared **after** the repo is public.

1. **When public (or briefly flip public to set it):** Repo → **Settings** → scroll past **Features** → **Social preview** → **Edit** → upload `assets/social-preview.png`
   - Or run: `gh browse --settings -R umbecanessa/punk-records-inference`
2. **Automated:** `python scripts/upload_social_preview.py` (requires signed-in browser via CDP or `--headed`)

Verify:

```bash
gh api graphql -f query='query { repository(owner:"umbecanessa", name:"punk-records-inference") { openGraphImageUrl } }'
```
