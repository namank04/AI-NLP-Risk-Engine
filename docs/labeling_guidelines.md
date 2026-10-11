# Gold-set labeling guidelines

The gold set is the ground truth every model in this project is measured against.
Labels must be consistent, so when in doubt, follow these rules rather than intuition.

Each item is one piece of text (headline, article lead, filing or post). For each
**entity** the engine linked (a ticker, or `MARKET` for market-wide news) you give:

1. **Relevant?** Is this entity a real subject of the text?
2. **Sentiment** toward that entity, on a 5-point scale.

For the item as a whole you give the **event class(es)** and, optionally, a subtype.

---

## 1. Relevance

Mark **not relevant** when the link is wrong or incidental:

| Text | Entity | Relevant? | Why |
|---|---|---|---|
| "Russian scientists raise funds to rebuild Tesla Tower" | TSLA | No | Nikola Tesla, not the company |
| "Lowest price ever on this book at Amazon" | AMZN | No | Shopping deal, no information about the company |
| "Micron is now worth more than Coca-Cola, Disney and Nike" | KO | No | KO is a size yardstick, not the subject |
| "Amazon signs 20-year nuclear power deal" | AMZN | Yes | A corporate action by Amazon |
| "Romania Q2 GDP contracts" | MARKET | Yes, but see note | Market-wide macro news, foreign scope |

`MARKET` is relevant when the news can move broad US/global asset prices: central
banks, inflation, growth, rates, credit conditions, geopolitics, commodities, major
policy. Purely local news ("Iowa agriculture supports 1 in 4 jobs") is **not** relevant.

## 2. Sentiment (toward the entity, for its investors)

| Score | Meaning | Examples |
|---|---|---|
| **-2** | Clearly bad news | default, downgrade to junk, guidance cut, fraud probe, recall, big miss |
| **-1** | Mildly bad | small miss, analyst trims target, lawsuit filed, cautious outlook |
| **0** | Neutral / factual / mixed | scheduled events, option-activity lists, balanced news |
| **+1** | Mildly good | small beat, target raised, new contract |
| **+2** | Clearly good | big beat and raise, approval of a key product, accretive deal at a premium |

Rules:

- **Judge from the investor's point of view of *that* entity.** "JPMorgan agrees to
  acquire First Republic after regulators seize the lender" is +1 for JPM (cheap
  deposit franchise), even though the story is about a bank failure.
- **Expectation, not level.** "Loss narrows" is positive; "profit rises but misses
  estimates" is negative; "guidance maintained" amid weakness is mildly negative.
- **For `MARKET`, think about broad equities and credit.** A surprise rate hike or
  hotter inflation is negative; easing inflation or a ceasefire is positive.
- **Social posts:** rate the poster's stance ("$TSLA to the moon" = +1, not +2, since
  hype carries little information). Sarcasm counts as what it means.
- Opinion pieces count only when they convey a concrete view; generic listicles = 0.

## 3. Event class (multi-label)

Pick **every** class that applies; the first one you pick is the primary.
Choose **None/Noise** only when nothing below applies.

| Class | Use when the text reports... | Not when... |
|---|---|---|
| **Geopolitical** | war, sanctions, tariffs, export controls, elections, coups | a company merely operates abroad |
| **Macroeconomic** | central banks, inflation, jobs, GDP, yields, fiscal policy, sovereign debt | a company comments on "the economy" in an earnings call (that is Earnings) |
| **Credit Event** | bankruptcy, missed payment, restructuring, downgrade or negative outlook, covenant breach, liquidity stress, bank runs | a company issues new bonds normally |
| **Merger/Acquisition** | announced/completed/terminated deals, takeover bids, divestitures, spin-offs, activist stakes | partnerships or supply contracts (use Product or None) |
| **Earnings** | results, beats/misses, guidance, pre-announcements, **analyst rating/target changes** | — |
| **Product Launch** | new products/services, regulatory product approvals (FDA), recalls of products | marketing fluff with no new product |
| **Regulatory/Legal** | lawsuits, settlements, fines, antitrust, investigations, new rules | — |
| **Management Change** | CEO/CFO/director departures or appointments, governance fights | routine hires below C-suite |
| **Operational/Supply Chain** | outages, accidents, cyber incidents, strikes, supply disruptions, layoffs, disasters | — |
| **None/Noise** | price-action chatter, "stocks moving today" lists, promotion, generic commentary | — |

Subtypes are optional but valuable for Credit Event (`bankruptcy`, `failure_to_pay`,
`restructuring`, `rating_action`, `covenant_breach`, `liquidity_stress`).

## 4. Edge cases decided once

- **Rating-agency actions** on a company are Credit Event (`rating_action`) regardless of direction; sentiment carries the direction.
- **Analyst upgrades/downgrades of a stock** (Buy/Sell) are Earnings (`analyst_rating`), *not* Credit Event.
- **Bank failures** are Credit Event for the failing bank and for `MARKET` when systemic.
- **"Stocks to buy" listicles** are None/Noise with sentiment 0 unless the item gives a concrete company-specific reason.
- **8-K filings:** label from the press-release text, not the item-code boilerplate.

## 5. Quality control

- Label at a steady pace (~30-40 items per sitting).
- After a week, re-label a random 50 items blind; agreement with your earlier labels
  (intra-annotator Cohen's kappa) is reported alongside model results.
- Use **Skip** for unreadable or non-English items rather than guessing.
