"""Build composite logo assets from Punk Records frontend sources."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def main() -> None:
    logo = Image.open(ASSETS / "logo.png").convert("RGBA")
    hat = Image.open(ASSETS / "straw-hat.png").convert("RGBA")

    hat_w = int(logo.width * 0.48)
    hat_h = int(hat.height * (hat_w / hat.width))
    hat = hat.resize((hat_w, hat_h), Image.Resampling.LANCZOS)
    hat = hat.rotate(-22, expand=True, resample=Image.Resampling.BICUBIC)

    out = logo.copy()
    # Perch the straw hat on the top-left corner of the hex mark
    x = int(out.width * 0.04)
    y = int(out.height * 0.01) - int(hat.height * 0.14)
    out.alpha_composite(hat, (x, y))
    out.save(ASSETS / "logo-mark.png", optimize=True)

    out.resize((256, 256), Image.Resampling.LANCZOS).save(
        ASSETS / "logo-mark-256.png", optimize=True
    )

    banner = Image.new("RGBA", (1200, 300), (13, 13, 13, 255))
    mark = out.resize((220, 220), Image.Resampling.LANCZOS)
    banner.paste(mark, (40, 40), mark)
    banner.save(ASSETS / "banner.png", optimize=True)

    print(f"Wrote {ASSETS / 'logo-mark.png'}")
    print(f"Wrote {ASSETS / 'logo-mark-256.png'}")
    print(f"Wrote {ASSETS / 'banner.png'}")


if __name__ == "__main__":
    main()
