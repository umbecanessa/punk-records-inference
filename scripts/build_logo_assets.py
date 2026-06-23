"""Build logo and social-preview assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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


def _logo_for_banner() -> Image.Image:
    composite = ASSETS / "logo-composite.png"
    plain = ASSETS / "logo.png"
    path = composite if composite.is_file() else plain
    return Image.open(path).convert("RGBA")


def _social_preview_banner(logo: Image.Image) -> Image.Image:
    banner = Image.new("RGBA", BANNER_SIZE, (13, 13, 13, 255))
    icon = logo.resize((320, 320), Image.Resampling.LANCZOS)
    banner.paste(icon, (72, (BANNER_SIZE[1] - icon.height) // 2), icon)

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
    logo = _logo_for_banner()
    logo.resize((256, 256), Image.Resampling.LANCZOS).save(
        ASSETS / "logo-256.png", optimize=True
    )

    social = _social_preview_banner(logo)
    social.save(ASSETS / "social-preview.png", optimize=True)
    social.save(ASSETS / "banner.png", optimize=True)

    print(f"Wrote {ASSETS / 'social-preview.png'} ({BANNER_SIZE[0]}x{BANNER_SIZE[1]})")
    print(f"Wrote {ASSETS / 'banner.png'}")


if __name__ == "__main__":
    main()
