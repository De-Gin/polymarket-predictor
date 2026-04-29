"""Parse cached ufcstats.com HTML into parquet.

Input:  data/ufcstats/raw_html/events/*.html, fights/*.html
Output: data/ufcstats/fights.parquet  (one row PER FIGHTER per fight)

Schema (fighter-centric — each bout becomes 2 rows, one per corner):
    fight_id, event_id, event_name, fight_date, title_fight,
    fighter_hash, fighter_name, opponent_hash, opponent_name,
    result,                         # W / L / D / NC
    method, end_round, end_time_sec,
    # this fighter's stats this fight
    kd, sig_str_landed, sig_str_att,
    total_str_landed, total_str_att,
    td_landed, td_att, sub_att, rev, ctrl_sec,
    head_landed, head_att, body_landed, body_att, leg_landed, leg_att,
    distance_landed, distance_att, clinch_landed, clinch_att, ground_landed, ground_att,
    # opponent mirror (so we can derive sapm / str_def without a second lookup)
    opp_sig_str_landed, opp_sig_str_att,
    opp_td_landed, opp_td_att,
    # fight duration for rate stats
    total_fight_time_sec

Usage:
    python scripts/parse_ufcstats.py
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

CACHE_ROOT = Path("data/ufcstats/raw_html")
OUT_PATH = Path("data/ufcstats/fights.parquet")

_HASH_RE = re.compile(r"/([a-f0-9]{16})(?:/|$)")
_OF_RE = re.compile(r"(\d+)\s+of\s+(\d+)")
_TIME_RE = re.compile(r"(\d+):(\d+)")


def _hash_from_url(url: str) -> str | None:
    m = _HASH_RE.search(url or "")
    return m.group(1) if m else None


def _ptexts(td: Tag) -> list[str]:
    """Return the two <p> stripped texts inside a cell."""
    return [p.get_text(strip=True) for p in td.find_all("p", class_="b-fight-details__table-text")]


def _parse_of(s: str) -> tuple[int, int]:
    """'17 of 42' -> (17, 42). '---' -> (0, 0)."""
    m = _OF_RE.search(s or "")
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _parse_time_sec(s: str) -> int:
    m = _TIME_RE.search(s or "")
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def _parse_int(s: str) -> int:
    s = (s or "").strip()
    try:
        return int(s)
    except ValueError:
        return 0


# --- event pages: date + fight list --------------------------------------


def parse_event(path: Path) -> dict:
    """Return {event_id, event_name, event_date, fight_hashes}."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    event_id = path.stem

    title = soup.select_one("h2.b-content__title span.b-content__title-highlight")
    event_name = title.get_text(strip=True) if title else ""

    event_date: date | None = None
    for li in soup.select("li.b-list__box-list-item"):
        label = li.select_one("i.b-list__box-item-title")
        if label and "Date" in label.get_text():
            text = li.get_text(" ", strip=True)
            text = text.replace(label.get_text(strip=True), "").strip()
            try:
                event_date = datetime.strptime(text, "%B %d, %Y").date()
            except ValueError:
                pass
            break

    fight_hashes: list[str] = []
    for tr in soup.select("tr.b-fight-details__table-row"):
        href = tr.get("data-link") or ""
        h = _hash_from_url(href)
        if h:
            fight_hashes.append(h)

    return {
        "event_id": event_id,
        "event_name": event_name,
        "event_date": event_date,
        "fight_hashes": fight_hashes,
    }


def build_event_index() -> dict[str, dict]:
    """Map event_id -> event metadata for every cached event page."""
    out: dict[str, dict] = {}
    for p in (CACHE_ROOT / "events").glob("*.html"):
        info = parse_event(p)
        out[info["event_id"]] = info
    return out


# --- fight page ----------------------------------------------------------


def _find_outer_totals_table(soup: BeautifulSoup) -> Tag | None:
    """The summary Totals table — the first <table> that's NOT inside a js-fight-table per-round one.

    Page layout: there are 4 tables.
      0: totals summary        (what we want)
      1: totals per round      (class includes js-fight-table)
      2: sig strikes summary   (what we want)
      3: sig strikes per round (class includes js-fight-table)
    """
    tables = soup.select("table")
    non_perround = [t for t in tables if "js-fight-table" not in (t.get("class") or [])]
    return non_perround[0] if len(non_perround) >= 1 else None


def _find_sig_strikes_table(soup: BeautifulSoup) -> Tag | None:
    tables = soup.select("table")
    non_perround = [t for t in tables if "js-fight-table" not in (t.get("class") or [])]
    return non_perround[1] if len(non_perround) >= 2 else None


def _extract_fight_head(soup: BeautifulSoup) -> dict:
    """Event link, fighters, W/L status, method, end round, end time, title-bout flag."""
    out: dict = {}

    event_a = soup.select_one("h2.b-content__title a")
    out["event_id"] = _hash_from_url(event_a.get("href", "") if event_a else "") or ""

    persons = soup.select("div.b-fight-details__person")
    fighters: list[dict] = []
    for p in persons:
        status = p.select_one("i.b-fight-details__person-status")
        link = p.select_one("a.b-fight-details__person-link")
        fighters.append(
            {
                "status": status.get_text(strip=True) if status else "",
                "hash": _hash_from_url(link.get("href", "") if link else ""),
                "name": link.get_text(strip=True) if link else "",
            }
        )
    out["fighters"] = fighters

    # Title bout detection — belt image OR "Title" in the fight title.
    title_tag = soup.select_one("i.b-fight-details__fight-title")
    title_txt = title_tag.get_text(" ", strip=True).lower() if title_tag else ""
    has_belt = bool(title_tag and title_tag.find("img", src=re.compile("belt")))
    out["title_fight"] = has_belt or "title" in title_txt

    # Method / round / time — scan the first <p> labels.
    method = ""
    end_round = 0
    end_time = "0:00"
    p_text = soup.select_one("p.b-fight-details__text")
    if p_text:
        for item in p_text.find_all("i", recursive=False):
            label_tag = item.find("i", class_="b-fight-details__label")
            if not label_tag:
                continue
            label = label_tag.get_text(strip=True).rstrip(":")
            value = item.get_text(" ", strip=True).replace(label_tag.get_text(strip=True), "").strip()
            if label == "Method":
                method = value
            elif label == "Round":
                end_round = _parse_int(value)
            elif label == "Time":
                end_time = value
    out["method"] = method
    out["end_round"] = end_round
    out["end_time_sec"] = _parse_time_sec(end_time)

    return out


def _extract_totals(table: Tag) -> list[dict]:
    """Return [{fighter_hash, kd, sig_str_l, sig_str_a, total_str_l, total_str_a,
                td_l, td_a, sub_att, rev, ctrl_sec}, ...] — one per fighter."""
    if table is None:
        return []
    row = table.select_one("tbody tr.b-fight-details__table-row")
    if row is None:
        return []
    cells = row.find_all("td", recursive=False)
    if len(cells) < 10:
        return []

    # Cell 0: fighter links (two <a>); Cells 1..9: stats
    links = cells[0].select("a[href*='fighter-details']")
    hashes = [_hash_from_url(a.get("href", "")) for a in links][:2]
    if len(hashes) < 2:
        return []

    # Parse each stat cell — two values each (one per fighter).
    kd = [_parse_int(v) for v in _ptexts(cells[1])]
    sig_str = [_parse_of(v) for v in _ptexts(cells[2])]
    # cells[3] is sig% — derivable, skip
    total_str = [_parse_of(v) for v in _ptexts(cells[4])]
    td = [_parse_of(v) for v in _ptexts(cells[5])]
    # cells[6] is td% — derivable
    sub_att = [_parse_int(v) for v in _ptexts(cells[7])]
    rev = [_parse_int(v) for v in _ptexts(cells[8])]
    ctrl = [_parse_time_sec(v) for v in _ptexts(cells[9])]

    out = []
    for i in range(2):
        out.append(
            {
                "fighter_hash": hashes[i],
                "kd": kd[i] if i < len(kd) else 0,
                "sig_str_landed": sig_str[i][0] if i < len(sig_str) else 0,
                "sig_str_att": sig_str[i][1] if i < len(sig_str) else 0,
                "total_str_landed": total_str[i][0] if i < len(total_str) else 0,
                "total_str_att": total_str[i][1] if i < len(total_str) else 0,
                "td_landed": td[i][0] if i < len(td) else 0,
                "td_att": td[i][1] if i < len(td) else 0,
                "sub_att": sub_att[i] if i < len(sub_att) else 0,
                "rev": rev[i] if i < len(rev) else 0,
                "ctrl_sec": ctrl[i] if i < len(ctrl) else 0,
            }
        )
    return out


def _extract_sig_breakdown(table: Tag) -> list[dict]:
    """Head/Body/Leg/Distance/Clinch/Ground — landed/attempted per fighter."""
    if table is None:
        return [{}, {}]
    row = table.select_one("tbody tr.b-fight-details__table-row")
    if row is None:
        return [{}, {}]
    cells = row.find_all("td", recursive=False)
    if len(cells) < 9:
        return [{}, {}]

    # Columns: Fighter, Sig.str, Sig.str%, Head, Body, Leg, Distance, Clinch, Ground
    head = [_parse_of(v) for v in _ptexts(cells[3])]
    body = [_parse_of(v) for v in _ptexts(cells[4])]
    leg = [_parse_of(v) for v in _ptexts(cells[5])]
    distance = [_parse_of(v) for v in _ptexts(cells[6])]
    clinch = [_parse_of(v) for v in _ptexts(cells[7])]
    ground = [_parse_of(v) for v in _ptexts(cells[8])]

    out = []
    for i in range(2):
        out.append(
            {
                "head_landed": head[i][0] if i < len(head) else 0,
                "head_att": head[i][1] if i < len(head) else 0,
                "body_landed": body[i][0] if i < len(body) else 0,
                "body_att": body[i][1] if i < len(body) else 0,
                "leg_landed": leg[i][0] if i < len(leg) else 0,
                "leg_att": leg[i][1] if i < len(leg) else 0,
                "distance_landed": distance[i][0] if i < len(distance) else 0,
                "distance_att": distance[i][1] if i < len(distance) else 0,
                "clinch_landed": clinch[i][0] if i < len(clinch) else 0,
                "clinch_att": clinch[i][1] if i < len(clinch) else 0,
                "ground_landed": ground[i][0] if i < len(ground) else 0,
                "ground_att": ground[i][1] if i < len(ground) else 0,
            }
        )
    return out


def _fight_time_sec(end_round: int, end_time_sec: int) -> int:
    """Total fight duration — completed rounds × 5 min + final round time."""
    if end_round <= 0:
        return 0
    return (end_round - 1) * 300 + end_time_sec


def parse_fight(path: Path, event_index: dict[str, dict]) -> list[dict]:
    """One fight page -> two rows (one per fighter)."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    fight_id = path.stem
    head = _extract_fight_head(soup)
    fighters = head.get("fighters") or []
    if len(fighters) != 2 or not fighters[0].get("hash") or not fighters[1].get("hash"):
        return []

    event = event_index.get(head["event_id"], {})
    event_date = event.get("event_date")
    event_name = event.get("event_name", "")

    totals = _extract_totals(_find_outer_totals_table(soup))
    if len(totals) != 2:
        return []
    sig = _extract_sig_breakdown(_find_sig_strikes_table(soup))

    dur_sec = _fight_time_sec(head["end_round"], head["end_time_sec"])

    # Status → result code (W / L / D / NC).
    def _result_code(s: str) -> str:
        s = s.strip().upper()
        if s.startswith("W"):
            return "W"
        if s.startswith("L"):
            return "L"
        if s.startswith("D"):
            return "D"
        if s.startswith("N"):
            return "NC"
        return s or "?"

    rows = []
    for i in range(2):
        me = fighters[i]
        opp = fighters[1 - i]
        row: dict = {
            "fight_id": fight_id,
            "event_id": head["event_id"],
            "event_name": event_name,
            "fight_date": event_date,
            "title_fight": head["title_fight"],
            "fighter_hash": me["hash"],
            "fighter_name": me["name"],
            "opponent_hash": opp["hash"],
            "opponent_name": opp["name"],
            "result": _result_code(me["status"]),
            "method": head["method"],
            "end_round": head["end_round"],
            "end_time_sec": head["end_time_sec"],
            "total_fight_time_sec": dur_sec,
            "opp_sig_str_landed": totals[1 - i]["sig_str_landed"],
            "opp_sig_str_att": totals[1 - i]["sig_str_att"],
            "opp_td_landed": totals[1 - i]["td_landed"],
            "opp_td_att": totals[1 - i]["td_att"],
        }
        row.update(totals[i])
        if i < len(sig):
            row.update(sig[i])
        rows.append(row)
    return rows


# --- driver --------------------------------------------------------------


def main() -> int:
    # Run from project root so relative paths work regardless of cwd.
    script_dir = Path(__file__).resolve().parent
    import os
    os.chdir(script_dir.parent)

    if not (CACHE_ROOT / "events").exists():
        print("no events cached — run scrape_ufcstats.py first", file=sys.stderr)
        return 1

    print("indexing events...")
    ei = build_event_index()
    print(f"  {len(ei)} events")

    fight_files = sorted((CACHE_ROOT / "fights").glob("*.html"))
    print(f"parsing {len(fight_files)} fight pages...")

    all_rows: list[dict] = []
    skipped = 0
    for i, fp in enumerate(fight_files, 1):
        rows = parse_fight(fp, ei)
        if not rows:
            skipped += 1
        else:
            all_rows.extend(rows)
        if i % 250 == 0:
            print(f"  [{i}/{len(fight_files)}] rows={len(all_rows)} skipped={skipped}")

    df = pd.DataFrame(all_rows)
    print(f"parsed {len(df)} rows from {len(fight_files) - skipped} fights (skipped {skipped})")

    # Sanity: date coverage and null checks.
    if "fight_date" in df.columns:
        missing_date = df["fight_date"].isna().sum()
        if missing_date:
            print(f"  warn: {missing_date} rows with no fight_date (event not cached?)")
        dated = df.dropna(subset=["fight_date"])
        if len(dated):
            print(f"  date range: {dated['fight_date'].min()} .. {dated['fight_date'].max()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
