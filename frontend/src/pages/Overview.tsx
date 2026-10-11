import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { CapitalGauge } from '../components/CapitalGauge'
import { Chart } from '../components/Chart'
import { Counter, EASE, Grow, Reveal, STATIC } from '../components/motion'
import { EventTag, ImpactMeter } from '../components/ui'
import { useStream } from '../hooks/useStream'
import { compact, money, signed, titleCase } from '../lib/format'
import { DEEMPH, divergingColor, INK, POS } from '../lib/theme'

const SAMPLES = [
  'Apple beats quarterly estimates on record iPhone demand, while Tesla misses badly as margins collapse.',
  "Boeing downgraded to junk by Moody's as 737 MAX crisis deepens and cash burn accelerates.",
  'Silicon Valley Bank shares plunge as depositors pull funds; regulators weigh options amid a liquidity crisis.',
  'Exxon Mobil agrees to buy Pioneer Natural Resources in a $59.5 billion all-stock deal.',
]

function useTyped(text: string, cps = 55) {
  const [n, setN] = useState(STATIC ? text.length : 0)
  useEffect(() => {
    if (STATIC) return
    setN(0)
    const id = setInterval(() => setN((k) => (k >= text.length ? (clearInterval(id), k) : k + 1)), 1000 / cps)
    return () => clearInterval(id)
  }, [text, cps])
  return { shown: text.slice(0, n), done: n >= text.length }
}

function EngineDemo() {
  const [k, setK] = useState(0)
  const text = SAMPLES[k]
  const { shown, done: typed } = useTyped(text)
  const { data, isError } = useQuery({ queryKey: ['demo', text], queryFn: () => api.analyze({ text }), staleTime: Infinity, retry: 2 })
  useQuery({ queryKey: ['demo', SAMPLES[(k + 1) % SAMPLES.length]], queryFn: () => api.analyze({ text: SAMPLES[(k + 1) % SAMPLES.length] }),
    staleTime: Infinity, enabled: !!data })
  const done = typed && !!data
  useEffect(() => {
    if (!done || STATIC) return
    const id = setTimeout(() => setK((x) => (x + 1) % SAMPLES.length), 4800)
    return () => clearTimeout(id)
  }, [done])
  return (
    <div className="relative rounded-2xl border border-line-strong bg-panel/90 shadow-2xl shadow-black/40 backdrop-blur">
      <div className="flex items-center justify-between border-b border-line px-5 py-3 text-xs">
        <span className="flex items-center gap-2 font-medium text-ink-2">
          <span className={clsx('h-1.5 w-1.5 rounded-full', data ? 'live-dot bg-good' : 'animate-pulse bg-warning')} />
          Live engine · POST /api/analyze
        </span>
        <span className="flex gap-1">{SAMPLES.map((_, i) => (
          <button key={i} onClick={() => setK(i)} aria-label={`Example ${i + 1}`}
            className={clsx('h-1.5 rounded-full transition-all', i === k ? 'w-5 bg-accent' : 'w-1.5 bg-[#2b3240] hover:bg-ink-3')} />))}
        </span>
      </div>
      <div className="min-h-[92px] px-5 pt-4 text-[15px] leading-relaxed text-ink">
        {shown}<span className={clsx('ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-accent', done ? 'animate-pulse' : '')} />
      </div>
      <div className="min-h-[190px] px-5 pt-2 pb-5">
        {typed && !data && <div className="text-xs text-ink-3">{isError ? 'API offline: start the backend to score this text.' : 'Scoring… (first call loads FinBERT and the event models)'}</div>}
        <AnimatePresence mode="wait">
          {done && (
            <motion.div key={k} initial="hide" animate="show" exit={{ opacity: 0, transition: { duration: 0.2 } }}
              variants={{ show: { transition: { staggerChildren: 0.12 } } }} className="space-y-2.5">
              {data!.map((sig) => ({ e: sig.entity === 'MARKET' ? 'Market' : sig.entity, s: sig.sentiment.score, ev: sig.event.primary,
                sub: sig.event.subtype, i: sig.impact.score, ch: sig.channels })).map((o) => (
                <motion.div key={o.e} variants={{ hide: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } } }}
                  className="rounded-xl border border-line bg-page/60 px-4 py-3">
                  <div className="flex items-center gap-3">
                    <span className="w-14 text-sm font-semibold">{o.e}</span>
                    <EventTag event={o.ev} />
                    {o.sub && <span className="text-[11px] text-ink-3">{titleCase(o.sub)}</span>}
                    <span className="ml-auto"><ImpactMeter score={o.i} /></span>
                  </div>
                  <div className="mt-2.5 grid grid-cols-[1fr_auto] items-center gap-3">
                    <div className="relative h-1.5 rounded-full bg-raised">
                      <span className="absolute top-[-3px] left-1/2 h-3 w-px bg-[#4a5262]" />
                      <div className={clsx('absolute top-0 h-1.5 w-1/2', o.s >= 0 ? 'left-1/2' : 'right-1/2')}>
                        <motion.span className="block h-1.5 rounded-full" style={{ background: divergingColor(o.s), transformOrigin: o.s >= 0 ? 'left' : 'right' }}
                          initial={{ scaleX: 0 }} animate={{ scaleX: Math.abs(o.s) }} transition={{ duration: 0.9, delay: 0.15, ease: EASE }} />
                      </div>
                    </div>
                    <span className="tnum w-12 text-right text-sm font-semibold" style={{ color: divergingColor(o.s) }}>
                      <Counter value={o.s} format={(v) => signed(v)} duration={0.9} /></span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {o.ch.map((c) => <span key={c} className="rounded-md bg-raised px-1.5 py-px text-[10px] text-ink-3">{titleCase(c)}</span>)}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

const NODE = 'fill-[#11151c] stroke-[#2b3240]'

function Node({ x, y, w = 170, h = 46, title, sub, accent }: { x: number; y: number; w?: number; h?: number; title: string; sub: string[]; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} className={NODE} strokeWidth={1} stroke={accent ? '#3987e5' : undefined} />
      <text x={x + 14} y={y + 20} fontSize={12.5} fontWeight={600} fill={INK.primary}>{title}</text>
      {sub.map((s, i) => <text key={s} x={x + 14} y={y + 36 + i * 15} fontSize={10.5} fill={INK.muted}>{s}</text>)}
    </g>
  )
}

function Pipeline() {
  const src = [['GDELT news', '15-minute global feed'], ['SEC EDGAR 8-K', 'material corporate events'], ['FNSPID', 'financial news, 2009–2023'], ['Stock tweets', 'social chatter']]
  const nlp = [['Entity sentiment', 'FinBERT, clause-targeted', 34], ['Event classification', '10 classes · student + NLI', 127], ['Impact 1–10', 'tested by event study', 220]] as const
  const paths = [
    ...src.map((_, i) => `M190,${43 + i * 70} C235,${43 + i * 70} 235,150 280,150`),
    ...nlp.map(([, , y]) => `M450,150 C495,150 495,${y + 23} 540,${y + 23}`),
    ...nlp.map(([, , y]) => `M710,${y + 23} C755,${y + 23} 755,150 800,150`),
    'M950,150 C990,150 990,73 1030,73', 'M950,150 C990,150 990,223 1030,223',
  ]
  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 1210 300" className="min-w-[900px]" role="img" aria-label="Data flow from sources through the NLP engine to Modules A and B">
        {paths.map((d, i) => <path key={i} d={d} fill="none" stroke="#2b3240" strokeWidth={1.5} />)}
        {paths.map((d, i) => <path key={`f${i}`} d={d} fill="none" stroke="#3987e5" strokeOpacity={0.55} strokeWidth={1.5} className="flow" />)}
        {!STATIC && paths.map((d, i) => (
          <circle key={`p${i}`} r={3} fill="#8fbdf2" style={{ filter: 'drop-shadow(0 0 4px #3987e5)' }}>
            <animateMotion dur={`${2.2 + (i % 3) * 0.4}s`} repeatCount="indefinite" begin={`${(i * 0.37) % 2}s`} path={d} />
          </circle>
        ))}
        {src.map(([t, s], i) => <Node key={t} x={20} y={20 + i * 70} title={t} sub={[s]} />)}
        <Node x={280} y={100} h={100} title="Ingestion" sub={['entity linking + relevance', 'story clustering, novelty', 'source credibility']} />
        {nlp.map(([t, s, y]) => <Node key={t} x={540} y={y} title={t} sub={[s]} />)}
        <Node x={800} y={112} w={150} h={76} title="Signal API" sub={['DuckDB store', 'REST + live SSE']} accent />
        <Node x={1030} y={50} title="Module A · Index" sub={['Black-Litterman rebalancer']} />
        <Node x={1030} y={200} title="Module B · Stress" sub={['IFRS 9 · IRB · CET1']} />
      </svg>
    </div>
  )
}

const CRISES = [
  { key: 'svb_2023', title: 'SVB collapse', when: 'March 2023',
    text: 'A deposit run sinks Silicon Valley Bank, contagion reaches Credit Suisse and First Republic. Watch whether the engine fires a systemic credit stress test, and when.' },
  { key: 'ukraine_2022', title: 'Russia–Ukraine', when: 'February 2022',
    text: 'Invasion, sanctions on Russian banks, the SWIFT cut-off and an oil spike. See how tone, the index tilt and bank capital respond day by day.' },
  { key: 'covid_2020', title: 'COVID crash', when: 'March 2020',
    text: 'Lockdowns, an oil price collapse and emergency Fed cuts, through company news and GDELT global coverage of the fastest bear market on record.' },
]

function Stat({ value, format, label, delay }: { value: number; format: (n: number) => string; label: string; delay: number }) {
  return (
    <Reveal delay={delay} className="border-l border-line pl-5">
      <div className="text-4xl font-semibold tracking-tight text-ink"><Counter value={value} format={format} duration={1.6} /></div>
      <div className="mt-1 text-[13px] text-ink-3">{label}</div>
    </Reveal>
  )
}

export default function Overview() {
  const nav = useNavigate()
  const { clear } = useStream()
  const [starting, setStarting] = useState<string | null>(null)
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 10000 })
  const { data: cmp } = useQuery({ queryKey: ['compare', 1], queryFn: () => api.stressCompare(1) })
  const { data: bt } = useQuery({ queryKey: ['backtest'], queryFn: async () => (await fetch('/api/gold/backtest')).json() as Promise<{ nav: Record<string, number | string>[] } | null> })
  const { data: universe } = useQuery({ queryKey: ['universe'], queryFn: api.universe })
  const { data: taxonomy } = useQuery({ queryKey: ['taxonomy'], queryFn: api.taxonomy })
  const { data: pf } = useQuery({ queryKey: ['portfolio'], queryFn: api.portfolio })
  const { data: curve } = useQuery({ queryKey: ['curve', 'credit_event', ''], queryFn: () => api.stressCurve('credit_event') })
  const { data: metrics } = useQuery({ queryKey: ['backtestMetrics'], queryFn: async () =>
    (await fetch('/api/gold/backtest')).json() as Promise<{ metrics: Record<string, Record<string, number>>; config: { bl?: { te_budget: number } } } | null> })
  const credit = cmp?.find((c) => c.key === 'credit_event')
  const sectors = new Set((universe ?? []).map((u) => u.sector)).size
  const te = metrics?.metrics.black_litterman?.tracking_error

  const replay = async (name: string) => {
    setStarting(name)
    try {
      clear()
      await api.startReplay({ name, speed: 21600, reset: true })
      nav('/desk')
    } finally { setStarting(null) }
  }

  return (
    <div className="-mt-8">
      <section className="grid min-h-[min(calc(100vh-110px),780px)] items-center gap-12 py-12 lg:grid-cols-[1.05fr_1fr]">
        <div>
          <motion.div initial={STATIC ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE }}
            className="inline-flex items-center gap-2 rounded-full border border-line bg-panel/70 px-3 py-1 text-xs text-ink-2">
            <span className="live-dot h-1.5 w-1.5 rounded-full bg-good" />
            S&amp;P Global × Crisil Campus Hackathon 2026 · AI/NLP Risk Engine
          </motion.div>
          <motion.h1 initial={STATIC ? false : { opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.08, ease: EASE }}
            className="mt-6 text-5xl leading-[1.04] font-semibold tracking-[-0.03em] md:text-6xl xl:text-7xl">
            Headlines in.<br />
            <span className="bg-gradient-to-r from-[#6da7ec] via-[#3987e5] to-[#2bb59a] bg-clip-text text-transparent">Capital impact out.</span>
          </motion.h1>
          <motion.p initial={STATIC ? false : { opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.16, ease: EASE }}
            className="mt-6 max-w-xl text-[17px] leading-relaxed text-ink-2">
            Signal Risk reads news, SEC filings and social posts, scores every company they mention, and streams the result into a
            sentiment-driven index and a bank stress test, live from GDELT and the SEC or replayed through past crises.
          </motion.p>
          <motion.div initial={STATIC ? false : { opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.24, ease: EASE }}
            className="mt-8 flex flex-wrap items-center gap-3">
            <button onClick={() => replay('svb_2023')} disabled={!!starting}
              className="group inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-[#3987e5]/25 transition hover:bg-[#4a94ea] disabled:opacity-60">
              <Play size={15} className="transition-transform group-hover:scale-110" />
              {starting === 'svb_2023' ? 'Loading models…' : 'Replay the SVB collapse'}
            </button>
            <Link to="/analyze" className="inline-flex items-center gap-2 rounded-xl border border-line-strong px-5 py-3 text-sm font-medium text-ink-2 transition hover:border-ink-3 hover:text-ink">
              Score your own text <ArrowRight size={15} />
            </Link>
          </motion.div>
          <motion.div initial={STATIC ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1, delay: 0.5 }}
            className="mt-10 flex flex-wrap gap-x-6 gap-y-2 text-[13px] text-ink-3">
            {universe && <span>{universe.length} S&amp;P 100 companies · {sectors} sectors</span>}
            {taxonomy && <span>{Object.keys(taxonomy).length} event classes</span>}
            {pf && <span>{pf.n_positions}-position bank book</span>}
            <span>GDELT · SEC EDGAR · FNSPID · tweets</span>
          </motion.div>
        </div>
        <motion.div initial={STATIC ? false : { opacity: 0, y: 24, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.9, delay: 0.2, ease: EASE }}>
          <EngineDemo />
        </motion.div>
      </section>

      <section className="py-10">
        <Reveal>
          <div className="mb-6 max-w-2xl">
            <div className="text-xs font-semibold tracking-wider text-accent uppercase">How it works</div>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight">One signal stream, two risk applications</h2>
            <p className="mt-3 text-ink-2">Every item is linked to companies, clustered with its copies, then scored per entity. The same signal
              that tilts the index can trigger a stress test.</p>
          </div>
        </Reveal>
        <Reveal delay={0.1} className="rounded-2xl border border-line bg-panel/60 p-4"><Pipeline /></Reveal>
      </section>

      <section className="grid gap-8 py-14 sm:grid-cols-2 lg:grid-cols-4">
        <Stat value={health?.raw_items ?? 0} format={(v) => compact(Math.round(v))} label="text items ingested into the store" delay={0} />
        <Stat value={health?.signals ?? 0} format={(v) => compact(Math.round(v))} label="entity-level signals produced from them" delay={0.08} />
        {curve?.breach_buffer && <Stat value={100 * curve.breach_buffer.severity} format={(v) => `${Math.round(v)}%`}
          label={`of Lehman-2008 severity takes CET1 below its ${(100 * curve.requirement).toFixed(1)}% requirement`} delay={0.16} />}
        {te !== undefined && <Stat value={100 * te} format={(v) => `${v.toFixed(1)}%`}
          label={`realised tracking error, 2022–2023 walk-forward (budget ${(100 * (metrics?.config.bl?.te_budget ?? 0.02)).toFixed(0)}%)`} delay={0.24} />}
      </section>

      <section className="py-10">
        <Reveal>
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="text-xs font-semibold tracking-wider text-accent uppercase">Crisis replays</div>
              <h2 className="mt-2 text-3xl font-semibold tracking-tight">Relive the news as it broke</h2>
            </div>
            <p className="max-w-md text-sm text-ink-3">Archived news replays at six hours a second through the live engine; every score, rebalance and stress run is computed as you watch.</p>
          </div>
        </Reveal>
        <div className="grid gap-5 md:grid-cols-3">
          {CRISES.map((c, i) => (
            <Reveal key={c.key} delay={i * 0.08}>
              <button onClick={() => replay(c.key)} disabled={!!starting}
                className="group flex h-full w-full flex-col rounded-2xl border border-line bg-panel p-6 text-left transition hover:-translate-y-1 hover:border-accent/50 hover:shadow-xl hover:shadow-[#3987e5]/10">
                <div className="text-xs text-ink-3">{c.when}</div>
                <div className="mt-1 text-xl font-semibold">{c.title}</div>
                <p className="mt-3 flex-1 text-[13px] leading-relaxed text-ink-2">{c.text}</p>
                <span className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-accent">
                  <span className="grid h-7 w-7 place-items-center rounded-full bg-accent-soft transition group-hover:bg-accent group-hover:text-white"><Play size={12} /></span>
                  {starting === c.key ? 'Loading…' : 'Replay'}
                </span>
              </button>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="grid gap-5 py-10 lg:grid-cols-2">
        <Reveal>
          <Link to="/index" className="group block h-full rounded-2xl border border-line bg-panel p-6 transition hover:border-accent/50">
            <div className="text-xs font-semibold tracking-wider text-accent uppercase">Module A</div>
            <h3 className="mt-2 text-2xl font-semibold">Sentiment-driven index</h3>
            <p className="mt-2 text-sm text-ink-2">Decayed, evidence-shrunk sentiment becomes Black-Litterman views, sized by Grinold’s rule and held to a
              2% tracking-error budget.</p>
            <div className="mt-4">
              {bt?.nav ? (
                <Chart height={150} option={{
                  grid: { left: 0, right: 0, top: 8, bottom: 0, containLabel: true },
                  legend: { top: 0, right: 0, data: ['Black-Litterman', 'Benchmark'] },
                  xAxis: { type: 'time', splitLine: { show: false }, axisLabel: { show: false }, axisLine: { show: false } },
                  yAxis: { type: 'value', scale: true, splitNumber: 2, axisLabel: { formatter: (v: number) => `$${v.toFixed(2)}` } },
                  tooltip: { trigger: 'axis', valueFormatter: (v) => `$${Number(v).toFixed(3)}` },
                  animationDuration: 2200,
                  series: [
                    { name: 'Benchmark', type: 'line', symbol: 'none', data: bt.nav.map((r) => [r.date, r.cap_weight]), lineStyle: { width: 1.2, color: DEEMPH }, itemStyle: { color: DEEMPH } },
                    { name: 'Black-Litterman', type: 'line', symbol: 'none', data: bt.nav.map((r) => [r.date, r.black_litterman]), lineStyle: { width: 2, color: POS }, itemStyle: { color: POS },
                      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(57,135,229,.25)' }, { offset: 1, color: 'rgba(57,135,229,0)' }] } } },
                  ],
                }} />
              ) : <div className="h-[150px]" />}
            </div>
            <span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-accent">Open the index <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" /></span>
          </Link>
        </Reveal>
        <Reveal delay={0.08}>
          <Link to="/stress" className="group block h-full rounded-2xl border border-line bg-panel p-6 transition hover:border-accent/50">
            <div className="text-xs font-semibold tracking-wider text-accent uppercase">Module B</div>
            <h3 className="mt-2 text-2xl font-semibold">Event-triggered stress test</h3>
            <p className="mt-2 text-sm text-ink-2">Corroborated high-impact events trigger scenarios calibrated to 2008, 2020 and 2022, applied to a
              synthetic wholesale bank through IFRS 9, Basel IRB and CET1.</p>
            <div className="mt-4 flex flex-wrap items-center gap-8">
              {credit ? <CapitalGauge id="overview" base={credit.summary.cet1_ratio_base} stressed={credit.summary.cet1_ratio_stressed}
                requirement={credit.summary.cet1_requirement} minimum={credit.summary.cet1_minimum} /> : <div className="h-[150px]" />}
              <div className="space-y-3 text-sm">
                <div className="text-ink-3">Lehman-2008 analogue, full severity</div>
                {credit && <>
                  <Bar label="IFRS 9 credit losses" v={credit.summary.ecl_increase} max={credit.summary.total_loss} />
                  <Bar label="Mark-to-market" v={credit.summary.mtm_loss} max={credit.summary.total_loss} />
                </>}
              </div>
            </div>
            <span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-accent">Open the stress test <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" /></span>
          </Link>
        </Reveal>
      </section>

      <footer className="mt-10 flex flex-wrap justify-between gap-3 border-t border-line pt-6 text-xs text-ink-3">
        <span>Naman Karwa · Indian Institute of Technology Guwahati</span>
        <span>Public and synthetic data only · MIT licence</span>
      </footer>
    </div>
  )
}

function Bar({ label, v, max }: { label: string; v: number; max: number }) {
  return (
    <div className="w-56">
      <div className="flex justify-between text-xs"><span className="text-ink-2">{label}</span><span className="tnum text-ink-3">{money(v)}</span></div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-raised"><Grow to={max ? v / max : 0} color={INK.secondary} className="h-1.5 w-full rounded-full" /></div>
    </div>
  )
}
