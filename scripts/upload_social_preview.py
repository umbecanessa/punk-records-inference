#!/usr/bin/env python3
"""Upload repository social preview image via GitHub Settings (Playwright).

GitHub has no public API for social preview upload. This script automates
Settings → Social preview → Upload using a browser that is already signed in.

Usage (Chrome with remote debugging — recommended):

    # Terminal 1: start Chrome logged into GitHub
    chrome.exe --remote-debugging-port=9222

    # Terminal 2: upload
    pip install playwright
    python scripts/upload_social_preview.py

Options:

    --repo owner/name     default: umbecanessa/punk-records-inference
    --image PATH          default: assets/social-preview.png
    --cdp URL             default: http://localhost:9222
    --headed              launch Chromium (sign in manually on first run)

Verify after upload:

    gh api graphql -f query='query { repository(owner:\"OWNER\", name:\"REPO\") { openGraphImageUrl } }'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "umbecanessa/punk-records-inference"
DEFAULT_IMAGE = ROOT / "assets" / "social-preview.png"


def _upload_via_page(page, repo: str, image: Path) -> None:
    settings_url = f"https://github.com/{repo}/settings"
    print(f"Navigating to {settings_url} ...")
    page.goto(settings_url, wait_until="domcontentloaded", timeout=60_000)

    if "login" in page.url:
        raise RuntimeError("Not signed in to GitHub in this browser session.")

    page.wait_for_function(
        "() => [...document.querySelectorAll('h2')]"
        ".some(h => h.textContent.includes('Social preview'))",
        timeout=20_000,
    )
    page.evaluate(
        """() => {
        for (const h of document.querySelectorAll('h2')) {
            if (h.textContent.includes('Social preview')) {
                h.scrollIntoView({ behavior: 'instant', block: 'center' });
                return;
            }
        }
    }"""
    )

    page.locator("summary:has-text('Edit')").first.click()
    with page.expect_file_chooser() as fc_info:
        page.locator("label[for='repo-image-file-input']").click()
    fc_info.value.set_files(str(image.resolve()))

    page.wait_for_function(
        "() => {"
        " const fa = document.querySelector("
        " 'file-attachment.js-upload-repository-image');"
        " return fa && !fa.classList.contains('is-default');"
        "}",
        timeout=20_000,
    )
    print("Upload submitted — waiting for GitHub to process ...")
    page.wait_for_timeout(3000)

    page.goto(f"https://github.com/{repo}", wait_until="domcontentloaded", timeout=30_000)
    og = page.evaluate(
        "() => document.querySelector('meta[property=\"og:image\"]')?.content"
    )
    if og and "repository-images" in (og or ""):
        print(f"Verified og:image: {og}")
    else:
        print(f"Upload done (og:image may lag): {og}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload GitHub repo social preview")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--cdp", default="http://localhost:9222")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        print(f"Image not found: {image}", file=sys.stderr)
        print("Run: python scripts/build_logo_assets.py", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install Playwright: pip install playwright", file=sys.stderr)
        return 1

    with sync_playwright() as p:
        if args.headed:
            profile = ROOT / ".github" / "playwright-profile"
            profile.mkdir(parents=True, exist_ok=True)
            context = p.chromium.launch_persistent_context(
                str(profile),
                headless=False,
                channel="msedge",
            )
            page = context.new_page()
            try:
                _upload_via_page(page, args.repo, image)
            finally:
                context.close()
            return 0

        try:
            browser = p.chromium.connect_over_cdp(args.cdp)
        except Exception as exc:
            print(
                "Could not connect to Chrome CDP. Start Chrome with:\n"
                "  chrome.exe --remote-debugging-port=9222\n"
                "Or run: python scripts/upload_social_preview.py --headed\n",
                file=sys.stderr,
            )
            print(f"CDP error: {exc}", file=sys.stderr)
            return 1

        if not browser.contexts:
            print("No browser contexts on CDP endpoint.", file=sys.stderr)
            browser.close()
            return 1

        page = browser.contexts[0].new_page()
        try:
            _upload_via_page(page, args.repo, image)
        finally:
            page.close()
            browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
