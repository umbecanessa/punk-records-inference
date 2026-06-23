"""Build composite logo assets from Punk Records frontend sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

BANNER_SIZE = (1280, 640)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_top_left_anchor(logo: Image.Image) -> tuple[int, int]:
    """Find the top-left vertex of the hex mark from alpha."""
    alpha = np.array(logo.split()[3], dtype=np.uint8).reshape(logo.size[1], logo.size[0])
    ys, xs = np.where(alpha > 96)
    if len(xs) == 0:
        w, h = logo.size
        return int(w * 0.22), int(h * 0.18)

    y_cut = np.percentile(ys, 32)
    top = ys <= y_cut
    # Top-left corner ≈ smallest x+y among upper facet pixels
    score = xs[top].astype(np.int64) + ys[top].astype(np.int64)
    idx = int(np.argmin(score))
    return int(xs[top][idx]), int(ys[top][idx])


def _drop_shadow(size: tuple[int, int], offset: tuple[int, int], blur: int = 6) -> Image.Image:
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse(
        (offset[0], offset[1], offset[0] + 48, offset[1] + 18),
        fill=(0, 0, 0, 90),
    )
    return shadow.filter(ImageFilter.GaussianBlur(blur))


def _compose_mark() -> Image.Image:
    logo = Image.open(ASSETS / "logo.png").convert("RGBA")
    hat_src = Image.open(ASSETS / "straw-hat.png").convert("RGBA")

    anchor_x, anchor_y = _hex_top_left_anchor(logo)

    # Smaller hat; rotate so brim follows the hex top-left facet
    scale = logo.width * 0.34 / hat_src.width
    hat_w = int(hat_src.width * scale)
    hat_h = int(hat_src.height * scale)
    hat = hat_src.resize((hat_w, hat_h), Image.Resampling.LANCZOS)
    hat = hat.rotate(-18, expand=True, resample=Image.Resampling.BICUBIC)

    # Anchor: right side of crown base sits on the top-left hex vertex
    anchor_on_hat_x = hat.width * 0.62
    anchor_on_hat_y = hat.height * 0.68

    x = int(anchor_x - anchor_on_hat_x)
    y = int(anchor_y - anchor_on_hat_y)

    out = logo.copy()
    shadow = _drop_shadow(out.size, (anchor_x - 18, anchor_y + 4))
    out = Image.alpha_composite(out, shadow)
    out.alpha_composite(hat, (x, y))
    return out


def _social_preview_banner(mark: Image.Image) -> Image.Image:
    """GitHub-recommended 1280×640 social preview."""
    banner = Image.new("RGBA", BANNER_SIZE, (13, 13, 13, 255))
    icon = mark.resize((320, 320), Image.Resampling.LANCZOS)
    banner.paste(icon, (72, (BANNER_SIZE[1] - icon.height) // 2), icon)

    draw = ImageDraw.Draw(banner)
    title_font = _load_font(56)
    accent_font = _load_font(30)
    muted_font = _load_font(24)

    draw.text((440, 210), "Punk Records Inference", fill=(255, 255, 255), font=title_font)
    draw.text((440, 290), "KV-state persistence for vLLM", fill=(0, 212, 170), font=accent_font)
    draw.text((440, 340), "capture · resume · .nls · BYOC · local-first", fill=(150, 150, 150), font=muted_font)
    return banner


def main() -> None:
    mark = _compose_mark()
    mark.save(ASSETS / "logo-mark.png", optimize=True)
    mark.resize((256, 256), Image.Resampling.LANCZOS).save(
        ASSETS / "logo-mark-256.png", optimize=True
    )

    social = _social_preview_banner(mark)
    social.save(ASSETS / "social-preview.png", optimize=True)
    social.save(ASSETS / "banner.png", optimize=True)

    print(f"Wrote {ASSETS / 'logo-mark.png'}")
    print(f"Wrote {ASSETS / 'logo-mark-256.png'}")
    print(f"Wrote {ASSETS / 'social-preview.png'} ({BANNER_SIZE[0]}x{BANNER_SIZE[1]})")
    print(f"Wrote {ASSETS / 'banner.png'}")


if __name__ == "__main__":
    main()
