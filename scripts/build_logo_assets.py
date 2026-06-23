"""Build logo and social-preview assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

BANNER_SIZE = (1280, 640)
SOURCE_CANDIDATES = (
    ASSETS / "logo-composite-source.png",
    ROOT.parent.parent / ".cursor/projects/c-Users-umber-Documents-GitHub-NLS/assets/logo-composite-v2.png",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find_source() -> Path:
    if (ASSETS / "logo-composite.png").is_file():
        return ASSETS / "logo-composite.png"
    for path in SOURCE_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError("No logo-composite source found")


def _remove_dark_background(img: Image.Image, *, threshold: int = 28, soft: int = 36) -> Image.Image:
    """Key near-black pixels to transparent with soft edges."""
    rgba = np.array(img.convert("RGBA"), dtype=np.float32)
    rgb = rgba[:, :, :3]
    brightness = np.max(rgb, axis=2)
    alpha = rgba[:, :, 3]

    # Soft matte: dark pixels fade out
    keyed = np.clip((brightness - threshold) * (255.0 / soft), 0, 255)
    new_alpha = np.minimum(alpha, keyed)
    rgba[:, :, 3] = new_alpha
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def _crop_to_content(img: Image.Image, *, pad_ratio: float = 0.06) -> Image.Image:
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    pad = int(max(x1 - x0, y1 - y0) * pad_ratio)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(x1 - x0, y1 - y0) // 2 + pad
    return img.crop((cx - half, cy - half, cx + half, cy + half))


def build_logo_composite(*, size: int = 512) -> Image.Image:
    src = Image.open(_find_source()).convert("RGBA")
    img = _remove_dark_background(src)
    img = _crop_to_content(img)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _social_preview_banner(logo: Image.Image) -> Image.Image:
    banner = Image.new("RGBA", BANNER_SIZE, (13, 13, 13, 255))
    icon = logo.resize((360, 360), Image.Resampling.LANCZOS)
    banner.paste(icon, (56, (BANNER_SIZE[1] - icon.height) // 2), icon)

    draw = ImageDraw.Draw(banner)
    draw.text((440, 210), "Punk Records Inference", fill=(255, 255, 255), font=_load_font(56))
    draw.text((440, 290), "KV-state persistence for vLLM", fill=(0, 212, 170), font=_load_font(30))
    draw.text(
        (440, 340),
        "capture · resume · .nls · BYOC · local-first",
        fill=(150, 150, 150),
        font=_load_font(24),
    )
    return banner


def main() -> None:
    composite = build_logo_composite(size=512)
    composite.save(ASSETS / "logo-composite.png", optimize=True)
    composite.resize((256, 256), Image.Resampling.LANCZOS).save(
        ASSETS / "logo-composite-256.png", optimize=True
    )

    social = _social_preview_banner(composite)
    social.save(ASSETS / "social-preview.png", optimize=True)
    social.save(ASSETS / "banner.png", optimize=True)

    print(f"Wrote {ASSETS / 'logo-composite.png'} (transparent bg, 512px)")
    print(f"Wrote {ASSETS / 'logo-composite-256.png'}")
    print(f"Wrote {ASSETS / 'social-preview.png'} ({BANNER_SIZE[0]}x{BANNER_SIZE[1]})")


if __name__ == "__main__":
    main()
