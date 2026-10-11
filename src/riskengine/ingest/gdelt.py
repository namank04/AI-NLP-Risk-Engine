"""GDELT 2.0 Global Knowledge Graph: global news in 15-minute files (title, URL, tone, themes)."""

from __future__ import annotations

import csv
import html
import io
import logging
import re
import zipfile
from datetime import UTC, datetime, timedelta

from riskengine.ingest.base import HttpClient, Source
from riskengine.schema import RawItem, stable_id

log = logging.getLogger(__name__)
GKG_BASE = "https://data.gdeltproject.org/gdeltv2"
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
# Themes worth keeping as features / weak labels for the event and impact models
FIN_THEME = re.compile(r"^(ECON_|EPU_|SANCTIONS|TAX_FNCACT_BANK|WB_\d+_(FINANC|BANK|DEBT|TRADE|ENERGY|MONETARY)|"
                       r"ARMEDCONFLICT|MILITARY|CYBER_ATTACK|STRIKE|PROTEST|NATURAL_DISASTER|CRISISLEX)")
csv.field_size_limit(2**31 - 1)


def _slots(since: datetime, until: datetime) -> list[datetime]:
    t = since.replace(minute=since.minute - since.minute % 15, second=0, microsecond=0)
    out = []
    while t <= until:
        out.append(t)
        t += timedelta(minutes=15)
    return out


class GdeltGkgSource(Source):
    """Every GKG record in the window; the engine's entity linker decides relevance."""

    name = "gdelt"

    def __init__(self):
        self.http = HttpClient(min_interval_s=0.5)

    def _file(self, slot: datetime) -> list[list[str]]:
        url = f"{GKG_BASE}/{slot:%Y%m%d%H%M%S}.gkg.csv.zip"
        resp = self.http.client.get(url)
        if resp.status_code != 200:            # recent slots appear with a lag; old gaps exist
            return []
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        with z.open(z.namelist()[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            return list(csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE))

    def fetch(self, since: datetime, until: datetime | None = None) -> list[RawItem]:
        until = until or datetime.now(UTC)
        items: dict[str, RawItem] = {}
        for slot in _slots(since, until):
            try:
                rows = self._file(slot)
            except Exception as exc:
                log.warning("GKG %s failed: %s", slot, exc)
                continue
            for r in rows:
                if len(r) < 27:
                    continue
                m = _TITLE.search(r[26])
                title = html.unescape(m.group(1)).strip() if m else ""
                url = r[4]
                if not title or not url.startswith("http"):
                    continue
                themes = sorted({t.split(",")[0] for t in r[8].split(";") if t and FIN_THEME.match(t)})
                item_id = stable_id("gdelt", url)
                items[item_id] = RawItem(
                    item_id=item_id, source=self.name, source_type="news", title=title, url=url,
                    publisher=r[3] or None,
                    published_at=datetime.strptime(r[1], "%Y%m%d%H%M%S").replace(tzinfo=UTC),
                    labels_hint={"gdelt_tone": r[15].split(",")[0], "gdelt_themes": ";".join(themes[:12])},
                )
        return sorted(items.values(), key=lambda i: i.published_at)
