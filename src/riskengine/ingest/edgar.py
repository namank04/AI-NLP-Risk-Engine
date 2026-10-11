"""SEC EDGAR 8-K filings: authoritative, timestamped corporate events, free.

8-K item codes are free event labels (see config/taxonomy.yaml: edgar_8k_items).
SEC fair-access policy: <= 10 requests/second and a User-Agent with contact info.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime

from bs4 import BeautifulSoup

from riskengine.config import RAW_DIR, universe
from riskengine.ingest.base import HttpClient, Source
from riskengine.schema import RawItem, stable_id

ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
# Items that carry no event information on their own
BOILERPLATE_ITEMS = {"9.01", "7.01", "5.07", "3.02", "5.03"}


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "RiskEngine hackathon prototype contact@example.com")


class EdgarSource(Source):
    name = "edgar"

    def __init__(self, fetch_body: bool = True, max_body_chars: int = 4000):
        self.http = HttpClient(min_interval_s=0.15, user_agent=_user_agent())
        self.fetch_body = fetch_body
        self.max_body_chars = max_body_chars
        self._ciks: dict[str, int] | None = None

    def ciks(self) -> dict[str, int]:
        if self._ciks is None:
            cache = RAW_DIR / "sec_company_tickers.json"
            if cache.exists():
                data = json.loads(cache.read_text(encoding="utf-8"))
            else:
                data = self.http.get("https://www.sec.gov/files/company_tickers.json").json()
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data), encoding="utf-8")
            by_ticker = {row["ticker"]: int(row["cik_str"]) for row in data.values()}
            self._ciks = {t: by_ticker[t] for t in universe().tickers if t in by_ticker}
        return self._ciks

    def _document_text(self, cik: int, accession: str, primary_doc: str) -> str:
        """Prefer the EX-99.1 press release; fall back to the 8-K body."""
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        target = f"{base}/{primary_doc}"
        try:
            index = self.http.get(f"{base}/index.json").json()
            names = [f["name"] for f in index.get("directory", {}).get("item", [])]
            ex99 = [n for n in names if re.search(r"ex-?99", n, re.I) and n.lower().endswith((".htm", ".html"))]
            if ex99:
                target = f"{base}/{sorted(ex99)[0]}"
        except Exception:
            pass
        html = self.http.get(target).text
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)[: self.max_body_chars]

    def fetch(self, since: datetime) -> list[RawItem]:
        items: list[RawItem] = []
        for ticker, cik in self.ciks().items():
            sub = self.http.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
            recent = sub["filings"]["recent"]
            for i, form in enumerate(recent["form"]):
                if form not in ("8-K", "8-K/A"):
                    continue
                accepted = datetime.fromisoformat(recent["acceptanceDateTime"][i].replace("Z", "+00:00"))
                if accepted < since:
                    continue
                codes = [c.strip() for c in recent["items"][i].split(",") if c.strip()]
                informative = [c for c in codes if c not in BOILERPLATE_ITEMS] or codes
                title = f"{sub['name']} 8-K: " + "; ".join(ITEM_TITLES.get(c, f"Item {c}") for c in informative)
                accession = recent["accessionNumber"][i]
                body = ""
                if self.fetch_body:
                    try:
                        body = self._document_text(cik, accession, recent["primaryDocument"][i])
                    except Exception:
                        body = ""
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{accession.replace('-', '')}/{recent['primaryDocument'][i]}")
                items.append(RawItem(
                    item_id=stable_id("edgar", accession),
                    source=self.name,
                    source_type="filing",
                    title=title,
                    body=body,
                    url=url,
                    publisher="SEC EDGAR",
                    published_at=accepted.astimezone(UTC),
                    tickers_hint=[ticker],
                    labels_hint={"8k_items": ",".join(codes)},
                ))
        return sorted(items, key=lambda i: i.published_at)
