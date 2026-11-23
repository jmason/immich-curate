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

Sources (proportions tunable via CLI flags):
  - Random — random photos from the whole library
  - "On this day" memories — photos from the same month-day in previous years
  - Favorites — photos the user has explicitly starred

The default mix is 90 % random, 8 % memories, 2 % favorites.  Override with
--random-pct, --memory-pct, --favorite-pct (must sum to 100).

Each run replaces the contents of the output directory so downstream scripts
always see a fresh set.

Environment variables:
  IMMICH_URL      Base URL of the Immich instance (e.g. https://immich.home.taint.org)
  IMMICH_API_KEY  API key (create one in Immich → User → API Keys)

Usage:
  ./immich-curate.py [--output DIR] [--count N]
                     [--random-pct P] [--memory-pct P] [--favorite-pct P]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

    # The memories endpoint omits people, so fetch each full asset record.
    full: list[dict[str, Any]] = []
    for asset in assets:
        resp = client.get(f"/api/assets/{asset['id']}")
        if resp.status_code == 200:
            full.append(resp.json())
    return full


# Search filters shared by the favorites and random queries.  "timeline"
# excludes archived assets and the hidden motion clips of live photos.
_SEARCH_FILTERS: dict[str, Any] = {"visibility": "timeline", "withPeople": True}


def getFavorites(client: httpx.Client, count: int) -> list[dict[str, Any]]:
    """Fetch up to *count* favorited assets via metadata search."""
    resp = client.post(
        "/api/search/metadata",
        json={"isFavorite": True, "size": count, "page": 1, **_SEARCH_FILTERS},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("assets", {}).get("items", [])


def getRandomAssets(client: httpx.Client, count: int) -> list[dict[str, Any]]:
    """Fetch *count* random assets (the server caps one call at 1000)."""
    resp = client.post(
        "/api/search/random", json={"size": min(count, 1000), **_SEARCH_FILTERS}
    )
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
    """Write a JSON sidecar at <outPath>.json with useful metadata.

    The sidecar appends ``.json`` rather than replacing the suffix so that
    e.g. ``001.jpg`` and ``001.mov`` get separate sidecars
    (``001.jpg.json`` and ``001.mov.json``) instead of clobbering one
    another.
    """
    info = {
        "assetId": asset.get("id"),
        "originalFileName": asset.get("originalFileName"),
        "localDateTime": asset.get("localDateTime"),
        "city": asset.get("exifInfo", {}).get("city"),
        "description": asset.get("exifInfo", {}).get("description"),
        "isFavorite": asset.get("isFavorite"),
    }
    sidecar = outPath.with_name(outPath.name + ".json")
    sidecar.write_text(json.dumps(info, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------

MIN_IMAGE_PIXELS = 1_000_000

# WhatsApp names its files e.g. IMG-20200319-WA0013.jpg.
_WHATSAPP_NAME = re.compile(r"^(img|vid)-\d{8}-wa\d+", re.IGNORECASE)


def rejectReason(asset: dict[str, Any]) -> str | None:
    """Return why *asset* should be skipped, based on its Immich metadata.

    Returns None if the asset is acceptable.  Pre-condition: *asset* came
    from a request that included people (withPeople, or a full asset fetch).

    >>> rejectReason({"visibility": "hidden", "type": "VIDEO"})
    'not on timeline (hidden)'
    """
    visibility = asset.get("visibility", "timeline")
    if visibility != "timeline":
        return f"not on timeline ({visibility})"

    name = (asset.get("originalFileName") or "").lower()
    if "screenshot" in name or "screen shot" in name:
        return "screenshot"
    if _WHATSAPP_NAME.match(name):
        return "WhatsApp"

    if asset.get("type") != "IMAGE":
        return None

    pixels = (asset.get("width") or 0) * (asset.get("height") or 0)
    if pixels < MIN_IMAGE_PIXELS:
        return f"low resolution ({pixels / 1e6:.1f} MP)"
    if not any(p.get("name") for p in asset.get("people") or []):
        return "no named person"
    return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def curate(
    client: httpx.Client,
    count: int,
    randomPct: float,
    memoryPct: float,
    favoritePct: float,
) -> list[dict[str, Any]]:
    """Build a deduplicated candidate list from a mix of sources.

    The three percentages give the target share of *count* photos drawn from
    each source (random, "on this day" memories, favorites).  They must sum
    to 100.  Each source is over-fetched to compensate for the
    interestingness filter rejecting some, and shuffled so the download loop
    picks from a varied pool rather than exhausting one source before the
    next.
    """
    # Pre-condition: the three percentages sum to 100.
    assert abs((randomPct + memoryPct + favoritePct) - 100.0) < 0.01, \
        "percentages must sum to 100"

    overFetch = 3  # fetch 3× what we need from each bucket
    seen: set[str] = set()
    seenDuplicateGroups: set[str] = set()

    def dedup(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return assets that pass rejectReason and are not already seen.

        Only one asset per Immich duplicate group is kept.
        """
        out: list[dict[str, Any]] = []
        # Loop invariant: every asset in out has an unseen id and an unseen
        # duplicateId (if any), and passed rejectReason.
        for a in assets:
            aid = a.get("id")
            dupId = a.get("duplicateId")
            if not aid or aid in seen or (dupId and dupId in seenDuplicateGroups):
                continue
            if rejectReason(a):
                continue
            seen.add(aid)
            if dupId:
                seenDuplicateGroups.add(dupId)
            out.append(a)
        return out

    def slice_(pct: float) -> int:
        """Convert a percentage of *count* into a fetch target (with over-fetch).

        Always fetch at least 1 if pct > 0, so a small percentage on a small
        count doesn't round to zero.
        """
        share = max(1, round(count * pct / 100.0)) if pct > 0 else 0
        return share * overFetch

    today = date.today()

    # --- Memories ---
    memTarget = slice_(memoryPct)
    if memTarget:
        print(f"Looking for 'on this day' memories ({today.strftime('%B %d')})…")
        rawMemories = getMemories(client, today)
        random.shuffle(rawMemories)
        memories = dedup(rawMemories)[:memTarget]
        print(f"  Found {len(rawMemories)}, using {len(memories)} candidate(s).")
    else:
        memories = []

    # --- Random ---
    randTarget = slice_(randomPct)
    if randTarget:
        print(f"Fetching random photos…")
        randoms: list[dict[str, Any]] = []
        # Most random assets fail the metadata filters, so keep asking.
        # ponytail: fixed round limit; a small library may not fill the target.
        for _ in range(20):
            randoms += dedup(getRandomAssets(client, 1000))
            if len(randoms) >= randTarget:
                break
        randoms = randoms[:randTarget]
        print(f"  Got {len(randoms)} random candidate(s).")
    else:
        randoms = []

    # --- Favorites ---
    favTarget = slice_(favoritePct)
    if favTarget:
        print(f"Fetching favorites…")
        rawFavs = getFavorites(client, 1000)
        random.shuffle(rawFavs)
        favs = dedup(rawFavs)[:favTarget]
        print(f"  Found {len(rawFavs)}, using {len(favs)} candidate(s).")
    else:
        favs = []

    # Interleave: round-robin from the three buckets, then append leftovers.
    # This ensures variety even if the download loop stops early.
    interleaved: list[dict[str, Any]] = []
    buckets = [randoms, memories, favs]
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
    parser.add_argument(
        "--random-pct",
        type=float,
        default=90.0,
        help="Percentage of photos drawn at random (default: 90)",
    )
    parser.add_argument(
        "--memory-pct",
        type=float,
        default=8.0,
        help="Percentage drawn from 'on this day' memories (default: 8)",
    )
    parser.add_argument(
        "--favorite-pct",
        type=float,
        default=2.0,
        help="Percentage drawn from favorited photos (default: 2)",
    )
    args = parser.parse_args()

    # Pre-condition: the three percentages must sum to 100 (within rounding).
    pctSum = args.random_pct + args.memory_pct + args.favorite_pct
    if abs(pctSum - 100.0) > 0.01:
        print(
            f"Error: --random-pct + --memory-pct + --favorite-pct must sum to 100; "
            f"got {pctSum:g} ({args.random_pct:g} + {args.memory_pct:g} + {args.favorite_pct:g}).",
            file=sys.stderr,
        )
        sys.exit(1)

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

    assets = curate(
        client,
        args.count,
        randomPct=args.random_pct,
        memoryPct=args.memory_pct,
        favoritePct=args.favorite_pct,
    )
    if not assets:
        print("No assets found — nothing to download.", file=sys.stderr)
        sys.exit(1)

    # Output directory: re-runs overwrite previous output in place so cached
    # file lists on the display side keep working (the underlying photo at
    # e.g. 001.jpg may change, but the filename stays the same).
    outDir.mkdir(parents=True, exist_ok=True)

    targetCount = args.count
    print(f"\nFiltering {len(assets)} candidate(s) and downloading up to {targetCount} to {outDir}/ …")
    downloaded = 0
    skippedBoring = 0
    skippedError = 0
    # Per-extension sequential counter, starting at 0 (000.jpg, 001.jpg, …
    # and independently 000.mov, 001.mov, …).  Key is the lowercase suffix
    # including the leading dot.
    extCounters: dict[str, int] = {}
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

        # --- Download to a temp path, then rename to its sequential name ---
        # Immich may correct the extension based on Content-Type, so we don't
        # know the final ext until after the download completes.
        tempSuffix = Path(origName).suffix or ".jpg"
        tempPath = outDir / f".tmp-{assetId}{tempSuffix}"
        ok, downloadPath = downloadAsset(client, assetId, tempPath)
        if not ok:
            # downloadAsset already printed an error; clean up if a partial
            # file was created.
            if tempPath.exists():
                tempPath.unlink()
            skippedError += 1
            continue

        ext = downloadPath.suffix.lower() or ".jpg"
        idx = extCounters.get(ext, 0)
        extCounters[ext] = idx + 1
        finalPath = outDir / f"{idx:03d}{ext}"
        # Path.replace is atomic on POSIX — overwrites any existing file at
        # finalPath, including its sidecar siblings on the next line.
        downloadPath.replace(finalPath)

        writeSidecar(finalPath, asset)
        downloaded += 1
        size = finalPath.stat().st_size
        print(f"  [{downloaded}/{targetCount}] {finalPath.name}  ({size / 1024:.0f} KB)")

    print(f"\nDone — {downloaded} photo(s) saved to {outDir}/")
    if skippedBoring:
        print(f"  ({skippedBoring} boring photo(s) filtered out)")


if __name__ == "__main__":
    main()
