#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "Pillow",
# ]
# ///
"""
Curate a selection of "good" photos from an Immich server into a local directory.

Strategy (in priority order):
  1. "On this day" memories — photos from the same month-day in previous years
  2. Favorites — photos the user has explicitly starred
  3. Random — if neither source yields enough, fill the rest randomly

Each run replaces the contents of the output directory so downstream scripts
always see a fresh set.

Environment variables:
  IMMICH_URL      Base URL of the Immich instance (e.g. https://immich.home.taint.org)
  IMMICH_API_KEY  API key (create one in Immich → User → API Keys)

Usage:
  ./immich-curate.py [--output DIR] [--count N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from io import BytesIO
from math import sqrt
from PIL import Image, ImageStat

# ---------------------------------------------------------------------------
# "Interestingness" filter — skip boring photos
# ---------------------------------------------------------------------------

# Thresholds (0–255 scale). Tuned to reject documents, sketches, whiteboard
# photos, screenshots, and other utility snapshots while keeping real photos.
MIN_MEAN_SATURATION = 35       # HSV saturation; documents/sketches are < 25
MIN_COLOR_STDDEV = 35          # combined RGB std-dev; flat images < 25
MAX_NEAR_WHITE_FRACTION = 0.55 # reject if >55 % of pixels are near-white


def _centerCrop(img: Image.Image) -> Image.Image:
    """Return the centre 60 % of the image (by area).

    This focuses analysis on the actual subject rather than table/desk borders.
    """
    w, h = img.size
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    return img.crop((margin_x, margin_y, w - margin_x, h - margin_y))


def _checkRegion(img: Image.Image) -> tuple[bool, str]:
    """Run interestingness checks on a single image region.

    Returns (pass, reason).
    """
    stat = ImageStat.Stat(img)

    # --- Colour standard deviation (RMS of per-channel stddevs) ---
    colorStd = sqrt(sum(s * s for s in stat.stddev) / 3.0)
    if colorStd < MIN_COLOR_STDDEV:
        return False, f"low color diversity (std={colorStd:.0f})"

    # --- Mean saturation in HSV ---
    hsvImg = img.convert("HSV")
    hsvStat = ImageStat.Stat(hsvImg)
    meanSat = hsvStat.mean[1]
    if meanSat < MIN_MEAN_SATURATION:
        return False, f"low saturation (mean={meanSat:.0f})"

    # --- Near-white pixel fraction ---
    pixels = img.getdata()
    sampleStep = max(1, len(pixels) // 10000)
    sampled = 0
    whiteCount = 0
    # Loop invariant: whiteCount <= sampled
    for i in range(0, len(pixels), sampleStep):
        r, g, b = pixels[i]
        sampled += 1
        if r > 220 and g > 220 and b > 220:
            whiteCount += 1
    whiteFrac = whiteCount / sampled if sampled else 0.0
    if whiteFrac > MAX_NEAR_WHITE_FRACTION:
        return False, f"too much white ({whiteFrac:.0%} near-white)"

    return True, "ok"


def isInteresting(imageBytes: bytes) -> tuple[bool, str]:
    """Analyse a thumbnail and decide whether the photo is 'interesting'.

    Checks both the full image and its centre crop.  A photo of a document on
    a colourful desk might pass the full-image check but fail the centre check
    (where the actual paper/whiteboard dominates).

    Returns (pass, reason).  reason explains why it was rejected (or "ok").
    """
    # Pre-condition: imageBytes is a valid image buffer.
    try:
        img = Image.open(BytesIO(imageBytes)).convert("RGB")
    except Exception as exc:
        return False, f"cannot decode image: {exc}"

    # Check full image first.
    ok, reason = _checkRegion(img)
    if not ok:
        return False, reason

    # Check the centre crop — catches documents on colourful backgrounds.
    centre = _centerCrop(img)
    ok, reason = _checkRegion(centre)
    if not ok:
        return False, f"centre: {reason}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Immich API helpers
# ---------------------------------------------------------------------------

def makeClient(baseUrl: str, apiKey: str) -> httpx.Client:
    """Create an httpx client pre-configured for the Immich API."""
    return httpx.Client(
        base_url=baseUrl.rstrip("/"),
        headers={"x-api-key": apiKey, "Accept": "application/json"},
        timeout=30.0,
        verify=False,  # common for home-lab TLS
    )


def getMemories(client: httpx.Client, today: date) -> list[dict[str, Any]]:
    """Fetch 'on this day' memory assets for *today*'s month/day."""
    resp = client.get("/api/memories")
    resp.raise_for_status()
    memories = resp.json()

    # Memories are grouped; extract individual asset IDs whose memoryAt
    # matches today's month and day.
    assets: list[dict[str, Any]] = []
    for mem in memories:
        for asset in mem.get("assets", []):
            # memoryAt or the asset's own date — check month/day match
            rawDate = asset.get("localDateTime") or asset.get("createdAt", "")
            try:
                dt = datetime.fromisoformat(rawDate.replace("Z", "+00:00"))
                if dt.month == today.month and dt.day == today.day:
                    assets.append(asset)
            except (ValueError, AttributeError):
                # If we can't parse the date, still include — the server
                # already filtered for relevance.
                assets.append(asset)
    return assets


def getFavorites(client: httpx.Client, count: int) -> list[dict[str, Any]]:
    """Fetch up to *count* favorited assets via metadata search."""
    resp = client.post(
        "/api/search/metadata",
        json={"isFavorite": True, "size": count, "page": 1},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("assets", {}).get("items", [])


def getRandomAssets(client: httpx.Client, count: int) -> list[dict[str, Any]]:
    """Fetch *count* random assets."""
    resp = client.get("/api/assets/random", params={"count": count})
    resp.raise_for_status()
    return resp.json()


def getThumbnail(client: httpx.Client, assetId: str) -> bytes | None:
    """Download thumbnail bytes for an asset.  Returns None on failure."""
    resp = client.get(f"/api/assets/{assetId}/thumbnail", follow_redirects=True)
    if resp.status_code == 200:
        return resp.content
    return None


# Map Content-Type to file extension for cases where Immich transcodes.
_CONTENT_TYPE_EXT: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


def downloadAsset(client: httpx.Client, assetId: str, outPath: Path) -> tuple[bool, Path]:
    """Download the original file for *assetId*.

    The extension of *outPath* may be corrected based on the response
    Content-Type (e.g. Immich transcodes .mov → .mp4).

    Returns (success, actualPath).
    """
    # Pre-condition: outPath's parent directory exists.
    assert outPath.parent.is_dir(), f"Parent directory does not exist: {outPath.parent}"

    resp = client.get(f"/api/assets/{assetId}/original", follow_redirects=True)
    if resp.status_code != 200:
        print(f"  ⚠ Could not download asset {assetId}: HTTP {resp.status_code}", file=sys.stderr)
        return False, outPath

    # Fix extension if the Content-Type disagrees with the filename.
    contentType = resp.headers.get("content-type", "").split(";")[0].strip()
    correctExt = _CONTENT_TYPE_EXT.get(contentType)
    if correctExt and outPath.suffix.lower() != correctExt:
        outPath = outPath.with_suffix(correctExt)

    outPath.write_bytes(resp.content)
    return True, outPath


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def writeSidecar(outPath: Path, asset: dict[str, Any]) -> None:
    """Write a small JSON sidecar next to the image with useful metadata."""
    info = {
        "assetId": asset.get("id"),
        "originalFileName": asset.get("originalFileName"),
        "localDateTime": asset.get("localDateTime"),
        "city": asset.get("exifInfo", {}).get("city"),
        "description": asset.get("exifInfo", {}).get("description"),
        "isFavorite": asset.get("isFavorite"),
    }
    sidecar = outPath.with_suffix(".json")
    sidecar.write_text(json.dumps(info, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def curate(client: httpx.Client, count: int) -> list[dict[str, Any]]:
    """Build a deduplicated candidate list from a mix of sources.

    Target mix (of the final *count* photos, before interestingness filtering):
      ~40 % "on this day" memories
      ~20 % favorites
      ~40 % random from the whole library

    Each source is over-fetched to compensate for the interestingness filter
    rejecting some.  The candidates are shuffled so the download loop picks
    from a varied pool rather than exhausting one source before the next.
    """
    overFetch = 3  # fetch 3× what we need from each bucket
    seen: set[str] = set()

    def isScreenshot(a: dict[str, Any]) -> bool:
        """Detect screenshots by filename pattern."""
        name = (a.get("originalFileName") or "").lower()
        return "screenshot" in name or "screen shot" in name

    def dedup(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return image-only, non-screenshot assets not already seen."""
        out: list[dict[str, Any]] = []
        # Loop invariant: out contains only unique IMAGE assets whose IDs
        # are in seen, with no screenshots.
        for a in assets:
            aid = a.get("id")
            if aid and aid not in seen and not isScreenshot(a):
                seen.add(aid)
                out.append(a)
        return out

    today = date.today()

    # --- Memories (hard cap: 10 from "on this day") ---
    memTarget = min(10, count // 2) * overFetch
    print(f"Looking for 'on this day' memories ({today.strftime('%B %d')})…")
    rawMemories = getMemories(client, today)
    random.shuffle(rawMemories)
    memories = dedup(rawMemories)[:memTarget]
    print(f"  Found {len(rawMemories)}, using {len(memories)} candidate(s).")

    # --- Random (fills the rest) ---
    randTarget = (count - min(10, count // 2)) * overFetch
    print(f"Fetching random photos…")
    rawRandom = getRandomAssets(client, max(randTarget, 50))
    randoms = dedup(rawRandom)[:randTarget]
    print(f"  Got {len(randoms)} random candidate(s).")

    # --- Favorites (sprinkle in a few) ---
    favTarget = count * overFetch // 6
    print(f"Fetching favorites…")
    rawFavs = getFavorites(client, max(favTarget, 50))
    random.shuffle(rawFavs)
    favs = dedup(rawFavs)[:favTarget]
    print(f"  Found {len(rawFavs)}, using {len(favs)} candidate(s).")

    # Interleave: round-robin from the three buckets, then append leftovers.
    # This ensures variety even if the download loop stops early.
    interleaved: list[dict[str, Any]] = []
    buckets = [memories, randoms, favs]  # memories and randoms alternate, favs fill in
    maxLen = max((len(b) for b in buckets), default=0)
    # Loop invariant: interleaved contains at most i * len(buckets) items.
    for i in range(maxLen):
        for bucket in buckets:
            if i < len(bucket):
                interleaved.append(bucket[i])

    return interleaved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate good photos from Immich into a local directory.",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.expanduser("~/Pictures/immich-curated"),
        help="Output directory (default: ~/Pictures/immich-curated)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=20,
        help="Number of photos to download (default: 20)",
    )
    args = parser.parse_args()

    # Pre-conditions: environment variables must be set.
    baseUrl = os.environ.get("IMMICH_URL")
    apiKey = os.environ.get("IMMICH_API_KEY")
    if not baseUrl or not apiKey:
        print(
            "Error: IMMICH_URL and IMMICH_API_KEY environment variables are required.\n"
            "  IMMICH_URL     — e.g. https://immich.home.taint.org\n"
            "  IMMICH_API_KEY — create one in Immich → User icon → API Keys",
            file=sys.stderr,
        )
        sys.exit(1)

    outDir = Path(args.output)

    client = makeClient(baseUrl, apiKey)

    # Verify connectivity.
    try:
        resp = client.get("/api/server/ping")
        resp.raise_for_status()
        print(f"Connected to Immich at {baseUrl}")
    except httpx.HTTPError as exc:
        print(
            f"Error: Cannot reach Immich at {baseUrl}/api/server/ping — {exc}\n"
            "Check IMMICH_URL and network connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)

    assets = curate(client, args.count)
    if not assets:
        print("No assets found — nothing to download.", file=sys.stderr)
        sys.exit(1)

    # Prepare output directory: clear old contents, keep the directory itself.
    outDir.mkdir(parents=True, exist_ok=True)
    for f in outDir.iterdir():
        f.unlink()

    targetCount = args.count
    print(f"\nFiltering {len(assets)} candidate(s) and downloading up to {targetCount} to {outDir}/ …")
    downloaded = 0
    skippedBoring = 0
    skippedError = 0
    # Loop invariant: downloaded + skippedBoring + skippedError == number of
    # candidates processed so far; downloaded <= targetCount.
    for asset in assets:
        if downloaded >= targetCount:
            break

        assetId = asset["id"]
        origName = asset.get("originalFileName", f"photo-{downloaded + 1}")

        # --- Interestingness check via thumbnail (images only) ---
        isVideo = asset.get("type") == "VIDEO"
        if not isVideo:
            thumb = getThumbnail(client, assetId)
            if thumb is None:
                print(f"  SKIP {origName} — could not fetch thumbnail")
                skippedError += 1
                continue

            interesting, reason = isInteresting(thumb)
            if not interesting:
                print(f"  SKIP {origName} — {reason}")
                skippedBoring += 1
                continue

        # --- Download the full-resolution image ---
        idx = downloaded + 1
        stem = Path(origName).stem
        suffix = Path(origName).suffix or ".jpg"
        outPath = outDir / f"{idx:02d}-{stem}{suffix}"

        ok, outPath = downloadAsset(client, assetId, outPath)
        if ok:
            writeSidecar(outPath, asset)
            downloaded += 1
            size = outPath.stat().st_size
            print(f"  [{downloaded}/{targetCount}] {outPath.name}  ({size / 1024:.0f} KB)")
        else:
            skippedError += 1

    print(f"\nDone — {downloaded} photo(s) saved to {outDir}/")
    if skippedBoring:
        print(f"  ({skippedBoring} boring photo(s) filtered out)")


if __name__ == "__main__":
    main()
