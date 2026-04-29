"""Scrape raw HTML from ufcstats.com into a local cache.

This script ONLY fetches and saves HTML. Parsing happens in a separate step
so you can iterate on parsers without re-hitting the network.

Cache layout:
    data/ufcstats/raw_html/
        events_index.html
        events/{event_hash}.html
        fights/{fight_hash}.html
        fighters/{fighter_hash}.html      (optional; Kaggle already has bio)

Resume: every fetch checks disk first. Crash + re-run = picks up where it left.

Mode 2 (active roster) usage:
    # One-time bulk (~15 min for 3 years of events):
    python scripts/scrape_ufcstats.py all --years 3

    # Or step-by-step (recommended first time, inspect between steps):
    python scripts/scrape_ufcstats.py events-index
    python scripts/scrape_ufcstats.py events --years 3
    python scripts/scrape_ufcstats.py fights

    # Weekly incremental (only new events):
    python scripts/scrape_ufcstats.py all --years 3

On-demand (one fighter + their recent fights):
    python scripts/scrape_ufcstats.py fighter "Ilia Topuria"
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import typer
from bs4 import BeautifulSoup

app = typer.Typer(add_completion=False, help="ufcstats.com raw-HTML scraper")

BASE = "http://ufcstats.com"
CACHE_ROOT = Path("data/ufcstats/raw_html")
EVENTS_INDEX_URL = f"{BASE}/statistics/events/completed?page=all"

# Polite defaults. 2 req/sec = ~0.5s sleep.
DEFAULT_SLEEP = 0.5

_HASH_RE = re.compile(r"/([a-f0-9]{16})(?:/|$)")

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": (
                    "sports-predictor/0.1 (research; contact: kuznetsovilas@gmail.com)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        _session = s
    return _session


def _hash_from_url(url: str) -> str | None:
    m = _HASH_RE.search(url)
    return m.group(1) if m else None


def fetch(url: str, cache_path: Path, sleep: float, force: bool = False) -> bytes:
    """GET url and cache body. Skip network if cache exists (unless force)."""
    if cache_path.exists() and not force:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Retry with backoff on transient errors.
    for attempt in range(3):
        try:
            r = session().get(url, timeout=20)
            if r.status_code == 200:
                cache_path.write_bytes(r.content)
                time.sleep(sleep)
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                typer.echo(f"  [{r.status_code}] {url} — retry in {wait}s", err=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            wait = 2 ** attempt
            typer.echo(f"  [error] {e} — retry in {wait}s", err=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url} after 3 attempts")


# --- phase 1: events index ------------------------------------------------


@app.command("events-index")
def cmd_events_index(
    sleep: float = typer.Option(DEFAULT_SLEEP, help="Seconds between requests."),
    force: bool = typer.Option(False, help="Re-fetch even if cached."),
) -> None:
    """Fetch the full events index (all completed UFC events, one HTML page)."""
    path = CACHE_ROOT / "events_index.html"
    typer.echo(f"fetching events index -> {path}")
    fetch(EVENTS_INDEX_URL, path, sleep=sleep, force=force)
    typer.echo(f"ok ({path.stat().st_size // 1024} KB)")


def _parse_events_index() -> list[tuple[str, str, date]]:
    """Return [(event_hash, url, event_date), ...] sorted by date desc."""
    path = CACHE_ROOT / "events_index.html"
    if not path.exists():
        typer.echo("events_index.html not cached — run 'events-index' first", err=True)
        raise typer.Exit(1)

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    rows = soup.select("tr.b-statistics__table-row")
    events: list[tuple[str, str, date]] = []
    for tr in rows:
        a = tr.select_one("a.b-link.b-link_style_black")
        if a is None:
            continue
        href = a.get("href", "")
        h = _hash_from_url(href)
        if not h:
            continue
        # Date cell is the sibling <span class="b-statistics__date"> inside the same <i>.
        date_span = tr.select_one("span.b-statistics__date")
        if date_span is None:
            continue
        try:
            event_date = datetime.strptime(date_span.text.strip(), "%B %d, %Y").date()
        except ValueError:
            continue
        events.append((h, href, event_date))
    events.sort(key=lambda t: t[2], reverse=True)
    return events


# --- phase 2: event pages -------------------------------------------------


@app.command("events")
def cmd_events(
    years: float = typer.Option(3.0, help="Fetch events within the last N years."),
    sleep: float = typer.Option(DEFAULT_SLEEP),
    force: bool = typer.Option(False),
) -> None:
    """Fetch each event-details page within the window."""
    events = _parse_events_index()
    cutoff = date.today() - timedelta(days=int(years * 365.25))
    targets = [(h, url, d) for (h, url, d) in events if d >= cutoff]
    typer.echo(f"{len(targets)} events in last {years} years (cutoff {cutoff})")

    for i, (h, url, d) in enumerate(targets, 1):
        path = CACHE_ROOT / "events" / f"{h}.html"
        was_cached = path.exists()
        fetch(url, path, sleep=sleep, force=force)
        tag = "cache" if was_cached and not force else "fetch"
        if i % 25 == 0 or i == len(targets):
            typer.echo(f"  [{i}/{len(targets)}] {tag} {d} {h}")
    typer.echo("events done")


def _parse_event_page(path: Path) -> list[tuple[str, str]]:
    """Return [(fight_hash, fight_url), ...] from a cached event page."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    out: list[tuple[str, str]] = []
    for tr in soup.select("tr.b-fight-details__table-row"):
        href = tr.get("data-link") or ""
        if not href:
            a = tr.select_one("a[href*='fight-details']")
            if a:
                href = a.get("href", "")
        h = _hash_from_url(href)
        if h:
            out.append((h, href))
    return out


# --- phase 3: fight pages -------------------------------------------------


@app.command("fights")
def cmd_fights(
    sleep: float = typer.Option(DEFAULT_SLEEP),
    force: bool = typer.Option(False),
) -> None:
    """Fetch every fight-details page linked from cached event pages."""
    event_dir = CACHE_ROOT / "events"
    if not event_dir.exists():
        typer.echo("no cached events — run 'events' first", err=True)
        raise typer.Exit(1)

    # Collect unique (hash, url) across all cached event pages.
    seen: dict[str, str] = {}
    for ev_path in sorted(event_dir.glob("*.html")):
        for h, url in _parse_event_page(ev_path):
            seen.setdefault(h, url)

    typer.echo(f"{len(seen)} unique fights linked from cached events")
    fight_dir = CACHE_ROOT / "fights"

    fetched = skipped = 0
    total = len(seen)
    for i, (h, url) in enumerate(seen.items(), 1):
        path = fight_dir / f"{h}.html"
        was_cached = path.exists()
        fetch(url, path, sleep=sleep, force=force)
        if was_cached and not force:
            skipped += 1
        else:
            fetched += 1
        if i % 100 == 0 or i == total:
            typer.echo(f"  [{i}/{total}] fetched={fetched} cached={skipped}")
    typer.echo(f"fights done: fetched={fetched} cached={skipped}")


# --- phase 4 (optional): fighter pages ------------------------------------


def _parse_fight_page_fighters(path: Path) -> list[tuple[str, str]]:
    """Return the two fighters linked on a fight-details page."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    out: list[tuple[str, str]] = []
    for a in soup.select("a.b-fight-details__person-link, a.b-link[href*='fighter-details']"):
        href = a.get("href", "")
        h = _hash_from_url(href)
        if h and (h, href) not in out:
            out.append((h, href))
    return out


@app.command("fighters")
def cmd_fighters(
    sleep: float = typer.Option(DEFAULT_SLEEP),
    force: bool = typer.Option(False),
) -> None:
    """(Optional) Fetch fighter-details pages for every fighter in cached fights."""
    fight_dir = CACHE_ROOT / "fights"
    if not fight_dir.exists():
        typer.echo("no cached fights — run 'fights' first", err=True)
        raise typer.Exit(1)

    seen: dict[str, str] = {}
    for fp in fight_dir.glob("*.html"):
        for h, url in _parse_fight_page_fighters(fp):
            seen.setdefault(h, url)

    typer.echo(f"{len(seen)} unique fighters referenced")
    fdir = CACHE_ROOT / "fighters"

    for i, (h, url) in enumerate(seen.items(), 1):
        path = fdir / f"{h}.html"
        fetch(url, path, sleep=sleep, force=force)
        if i % 100 == 0 or i == len(seen):
            typer.echo(f"  [{i}/{len(seen)}]")
    typer.echo("fighters done")


# --- convenience: run all phases -----------------------------------------


@app.command("all")
def cmd_all(
    years: float = typer.Option(3.0, help="Event window in years."),
    sleep: float = typer.Option(DEFAULT_SLEEP),
    include_fighters: bool = typer.Option(
        False, help="Also scrape fighter-details (skip — Kaggle has bio data)."
    ),
    force: bool = typer.Option(False),
) -> None:
    """Run events-index -> events -> fights (-> fighters) end-to-end."""
    cmd_events_index(sleep=sleep, force=force)
    cmd_events(years=years, sleep=sleep, force=force)
    cmd_fights(sleep=sleep, force=force)
    if include_fighters:
        cmd_fighters(sleep=sleep, force=force)


# --- on-demand: single fighter -------------------------------------------


def _search_fighter(name: str, sleep: float) -> str | None:
    """Find fighter_hash by scanning the first-letter index page."""
    first = name.strip()[0].lower()
    if not first.isalpha():
        return None
    url = f"{BASE}/statistics/fighters?char={first}&page=all"
    path = CACHE_ROOT / "fighters_index" / f"{first}.html"
    html = fetch(url, path, sleep=sleep)
    soup = BeautifulSoup(html, "lxml")

    target = re.sub(r"\s+", " ", name.strip().lower())
    for tr in soup.select("tr.b-statistics__table-row"):
        tds = tr.select("td.b-statistics__table-col")
        if len(tds) < 2:
            continue
        first_td = tds[0].get_text(strip=True).lower()
        last_td = tds[1].get_text(strip=True).lower()
        candidate = f"{first_td} {last_td}".strip()
        if candidate == target:
            a = tr.select_one("a[href*='fighter-details']")
            if a:
                return _hash_from_url(a.get("href", ""))
    return None


@app.command("fighter")
def cmd_fighter(
    name: str = typer.Argument(..., help="Fighter full name, e.g. 'Ilia Topuria'"),
    last_n: int = typer.Option(15, help="Also fetch last N fight-details pages."),
    sleep: float = typer.Option(DEFAULT_SLEEP),
) -> None:
    """On-demand: fetch one fighter's detail page + their last-N fight pages."""
    h = _search_fighter(name, sleep=sleep)
    if h is None:
        typer.echo(f"fighter '{name}' not found in ufcstats index", err=True)
        raise typer.Exit(1)
    typer.echo(f"fighter hash: {h}")

    fpath = CACHE_ROOT / "fighters" / f"{h}.html"
    fhtml = fetch(f"{BASE}/fighter-details/{h}", fpath, sleep=sleep)

    soup = BeautifulSoup(fhtml, "lxml")
    fight_urls: list[str] = []
    for a in soup.select("a.b-flag[href*='fight-details'], a[href*='fight-details']"):
        href = a.get("href", "")
        if href and href not in fight_urls:
            fight_urls.append(href)
    fight_urls = fight_urls[:last_n]

    typer.echo(f"fetching {len(fight_urls)} recent fight pages")
    for url in fight_urls:
        fh = _hash_from_url(url)
        if not fh:
            continue
        fetch(url, CACHE_ROOT / "fights" / f"{fh}.html", sleep=sleep)
    typer.echo("done")


if __name__ == "__main__":
    # Make relative cache paths resolve from the project root, not the CWD,
    # so invoking the script from anywhere still writes to data/ufcstats/.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    import os
    os.chdir(project_root)
    sys.exit(app() or 0)
