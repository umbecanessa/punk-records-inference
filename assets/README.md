# Brand assets

Official Punk Records branding from the hosted product frontend
(`NLS/punk-records/frontend/public/`).

| File | Use |
|------|-----|
| `logo.png` | Hex vinyl/circuit mark (source) |
| `straw-hat.png` | Straw hat overlay (source) |
| `logo-mark.png` | Composite mark (1024×1024) |
| `logo-mark-256.png` | README / docs header |
| `banner.png` | GitHub social preview (1200×300) |
| `favicon.ico` | Browser favicon |

Regenerate composites after updating source PNGs:

```bash
pip install pillow
python scripts/build_logo_assets.py
```
