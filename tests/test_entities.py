from riskengine.nlp.entities import EntityLinker
from riskengine.schema import MARKET_ENTITY


def tickers(mentions):
    return {m.entity for m in mentions}


linker = EntityLinker()


def test_cashtag_and_alias():
    m = linker.link("$AAPL rallies after Microsoft cloud revenue beat")
    assert {"AAPL", "MSFT"} <= tickers(m)
    by = {x.entity: x for x in m}
    assert by["AAPL"].confidence == 1.0 and by["AAPL"].evidence == "cashtag"
    assert by["MSFT"].evidence == "alias"


def test_ambiguous_alias_needs_finance_context():
    assert "AAPL" not in tickers(linker.link("I baked an apple pie with fresh apples"))
    assert "AAPL" not in tickers(linker.link("Apple orchards had a great harvest this autumn"))
    assert "AAPL" in tickers(linker.link("Apple shares fell after the company cut its revenue outlook"))


def test_ambiguous_lowercase_never_links():
    assert "META" not in tickers(linker.link("the meta discussion about earnings was long"))


def test_product_is_weak_and_only_without_company():
    m = {x.entity: x for x in linker.link("iPhone sales slump in China")}
    assert m["AAPL"].confidence == 0.5
    m = {x.entity: x for x in linker.link("Apple Inc says iPhone sales slump")}
    assert m["AAPL"].confidence == 0.9


def test_entity_sentences_are_targeted():
    text = "Apple Inc beat estimates on strong iPhone demand. Tesla missed badly as margins collapsed."
    m = {x.entity: x for x in linker.link(text)}
    assert m["AAPL"].sentences == ["Apple Inc beat estimates on strong iPhone demand."]
    assert m["TSLA"].sentences == ["Tesla missed badly as margins collapsed."]


def test_shared_sentence_is_split_into_entity_clauses():
    m = {x.entity: x for x in linker.link(
        "Apple beats quarterly estimates on record iPhone demand, while Tesla misses badly as margins collapse")}
    assert m["AAPL"].sentences == ["Apple beats quarterly estimates on record iPhone demand"]
    assert m["TSLA"].sentences == ["Tesla misses badly as margins collapse"]


def test_source_hint_forces_link():
    m = {x.entity: x for x in linker.link("Shares slide after guidance cut", tickers_hint=["XOM"])}
    assert m["XOM"].confidence == 1.0


def test_macro_falls_back_to_market_entity():
    m = linker.link("Federal Reserve signals two more rate hikes as inflation stays hot")
    assert tickers(m) == {MARKET_ENTITY}


def test_short_uppercase_ticker_alias_is_case_sensitive():
    assert "BA" not in tickers(linker.link("She has a BA in economics and works at a startup"))
    assert "GS" not in tickers(linker.link("gs shares"))


def test_scope_extraction():
    s = linker.scope("U.S. sanctions on Russia push Brent oil and the ruble lower; banks exposed")
    assert "US" in s.countries and "RU" in s.countries
    assert "crude_oil" in s.commodities
    assert "RUB" in s.currencies
    assert "Financials" in s.sectors and "Energy" in s.sectors


def test_rating_action_needs_credit_context():
    from riskengine.nlp.events import EventClassifier

    ec = EventClassifier(use_zero_shot=False, student=False)
    assert "credit_event" not in ec.classify(["The Microsoft 365 storage downgrade shows the perils of cloud services"])[0].labels
    hit = ec.classify(["Moody's downgrades Boeing to Baa3, outlook negative"])[0]
    assert hit.primary == "credit_event" and hit.subtype == "rating_action"


def test_index_cut_is_not_a_rating_action():
    from riskengine.nlp.events import EventClassifier

    ec = EventClassifier(use_zero_shot=False, student=False)
    out = ec.classify(["XLG Cuts the S&P 500 Down to 50 Stocks. The Result Is an Index Fund Built Entirely on Mega Caps"])[0]
    assert "credit_event" not in out.labels
    for t in ["Fitch cuts Italy's rating to BBB-", "Ford downgraded to junk by Moody's", "S&P Global Ratings downgrades Evergrande"]:
        assert ec.classify([t])[0].subtype == "rating_action", t


def test_tricky_new_aliases():
    lk = EntityLinker()
    assert "DOW" not in tickers(lk.link("Dow Jones falls 500 points as bank stocks slide"))
    assert "DOW" not in tickers(lk.link("The Dow drops 2% while investors flee stocks"))
    assert "DOW" not in tickers(lk.link("US stocks fall sharply as Dow sinks on Credit Suisse fears"))
    assert "DOW" in tickers(lk.link("Dow Inc to close plants in Europe"))
    assert "MRK" not in tickers(lk.link("Merck KGaA shares rise on strong earnings"))
    assert "MRK" in tickers(lk.link("Merck shares jump after Keytruda trial success"))
    assert "AMT" not in tickers(lk.link("Congress debates AMT changes for corporate taxpayers"))
    assert "AMT" in tickers(lk.link("American Tower raises dividend after strong quarter"))


def test_armed_conflict_outranks_macro():
    from riskengine.nlp.events import EventClassifier

    ec = EventClassifier(use_zero_shot=False, student=False)
    war = ec.classify(["Russian troops begin full-scale invasion of Ukraine; oil and inflation fears grip markets"])[0]
    assert war.primary == "geopolitical" and war.subtype == "armed_conflict"
    both = ec.classify(["U.S. imposes sanctions on Russian banks after the invasion of Ukraine"])[0]
    assert both.primary == "geopolitical" and both.subtype == "armed_conflict"
