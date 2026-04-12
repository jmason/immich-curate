Curate a selection of "good" photos from an Immich server into a local directory.

Strategy (in priority order):
  1. "On this day" memories — photos from the same month-day in previous years
  2. Favorites — photos the user has explicitly starred
  3. Random — if neither source yields enough, fill the rest randomly

Each run replaces the contents of the output directory so downstream scripts
always see a fresh set.

Environment variables:
- IMMICH_URL      Base URL of the Immich instance (e.g. https://immich.home.taint.org)
- IMMICH_API_KEY  API key (create one in Immich → User → API Keys)

Usage:
  ./immich-curate.py [--output DIR] [--count N]
