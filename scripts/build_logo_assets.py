"""Build composite logo assets from Punk Records frontend sources."""

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


def _compose_mark() -> Image.Image:
    logo = Image.open(ASSETS / "logo.png").convert("RGBA")
    hat = Image.open(ASSETS / "straw-hat.png").convert("RGBA")

    hat_w = int(logo.width * 0.48)
    hat_h = int(hat.height * (hat_w / hat.width))
    hat = hat.resize((hat_w, hat_h), Image.Resampling.LANCZOS)
    hat = hat.rotate(-22, expand=True, resample=Image.Resampling.BICUBIC)

    out = logo.copy()
    x = int(out.width * 0.04)
    y = int(out.height * 0.01) - int(hat.height * 0.14)
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
