import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ChevronRight, ExternalLink } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Signal } from '../api/types'
import { barStyle, Chart } from '../components/Chart'
import { PageTitle } from '../components/Layout'
import { Counter } from '../components/motion'
import { Empty, EventTag, Hero, ImpactMeter, Panel, Segmented, Select, SourceTag, ToneBar } from '../components/ui'
import { useStream } from '../hooks/useStream'
import { compact, day, EVENT_LABEL, eventLabel, signed, titleCase, when } from '../lib/format'
import { DIVERGING_STOPS, INK, NEG, POS, SERIES } from '../lib/theme'

interface Story {
  id: string
  headline: string
  url: string | null
  at: string
  event: string
  impact: number
  outlets: number
  sourceType: string
  signals: Signal[]
}

function groupStories(signals: Signal[]): Story[] {
  const by = new Map<string, Story>()
  for (const s of signals) {
    const st = by.get(s.cluster_id)
    if (!st) {
      by.set(s.cluster_id, { id: s.cluster_id, headline: s.evidence.headline, url: s.evidence.url, at: s.published_at,
        event: s.event.primary, impact: s.impact.score, outlets: s.n_sources, sourceType: s.source_type, signals: [s] })
    } else {
      st.impact = Math.max(st.impact, s.impact.score)
      st.outlets = Math.max(st.outlets, s.n_sources)
      if (!st.signals.some((x) => x.entity === s.entity)) st.signals.push(s)
    }
  }
  return [...by.values()]
}

const weight = (s: Signal) => s.source_weight * (0.4 + 0.6 * s.novelty)

function StoryDetail({ s }: { s: Signal }) {
  const probs = Object.entries(s.event.probs).filter(([k]) => k !== 'none').sort((a, b) => b[1] - a[1]).slice(0, 5)
  return (
    <div className="grid gap-6 border-t border-line bg-page/40 px-5 py-4 md:grid-cols-[1.2fr_1fr]">
      <div className="space-y-3 text-[13px]">
        <p className="leading-relaxed text-ink-2">{s.evidence.span ?? s.evidence.headline}</p>
        <dl className="grid grid-cols-3 gap-x-4 gap-y-2 text-xs">
          {[['Sentiment confidence', s.sentiment.confidence.toFixed(2)], ['Novelty', s.novelty.toFixed(2)],
            ['Source weight', s.source_weight.toFixed(2)], ['Channels', s.channels.map(titleCase).join(', ') || '—'],
            ['Scope', [...s.scope.countries, ...s.scope.commodities].join(', ') || '—'], ['Model', s.model_version]]
            .map(([k, v]) => <div key={k}><dt className="text-ink-3">{k}</dt><dd className="mt-0.5 text-ink-2">{v}</dd></div>)}
        </dl>
      </div>
      <div>
        <div className="mb-2 text-xs text-ink-3">Event probabilities</div>
        <div className="space-y-1.5">
          {probs.map(([k, v], i) => (
            <div key={k} className="grid grid-cols-[120px_1fr_36px] items-center gap-2 text-xs">
              <span className={i === 0 ? 'text-ink' : 'text-ink-3'}>{eventLabel(k)}</span>
              <span className="h-1.5 rounded-full bg-raised">
                <span className="block h-1.5 rounded-full" style={{ width: `${v * 100}%`, background: i === 0 ? SERIES[0] : '#3a4250' }} />
              </span>
              <span className="tnum text-right text-ink-3">{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Stream({ stories }: { stories: Story[] }) {
  const [mode, setMode] = useState<'all' | 'high' | 'filing'>('all')
  const [event, setEvent] = useState('')
  const [open, setOpen] = useState<string | null>(null)
  const rows = stories.filter((s) => (mode !== 'high' || s.impact >= 7) && (mode !== 'filing' || s.sourceType === 'filing')
    && (!event || s.event === event)).slice(0, 120)
  return (
    <Panel title="Stream" meta="Stories, with every company each one touches" flush
      actions={<>
        <Segmented size="sm" value={mode} onChange={setMode}
          options={[{ value: 'all', label: 'All' }, { value: 'high', label: 'Impact ≥ 7' }, { value: 'filing', label: 'Filings' }]} />
        <Select label="Event class" value={event} onChange={setEvent}>
          <option value="">All events</option>
          {Object.keys(EVENT_LABEL).map((k) => <option key={k} value={k}>{eventLabel(k)}</option>)}
        </Select>
      </>}>
      {rows.length === 0 ? <Empty tall>Press play to stream a replay, or score your own text under Analyze.</Empty> : (
        <ul className="max-h-[720px] divide-y divide-line overflow-y-auto">
          {rows.map((st) => {
            const isOpen = open === st.id
            return (
              <li key={st.id} className="arrive">
                <button onClick={() => setOpen(isOpen ? null : st.id)} aria-expanded={isOpen}
                  className="grid w-full grid-cols-[1fr_auto] gap-4 px-5 py-3 text-left hover:bg-hover/50">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-[11px] text-ink-3">
                      <span className="tnum">{when(st.at)}</span><span>·</span><SourceTag type={st.sourceType} />
                      {st.outlets > 1 && <><span>·</span><span>{st.outlets} outlets</span></>}
                    </div>
                    <div className="mt-1 line-clamp-2 text-[14px] leading-snug text-ink">{st.headline}</div>
                    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
                      {st.signals.slice(0, 4).map((s) => (
                        <span key={s.entity} className="flex items-center gap-2 text-xs">
                          <span className="font-semibold text-ink-2">{s.entity === 'MARKET' ? 'Market' : s.entity}</span>
                          <ToneBar value={s.sentiment.score} width={44} />
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-col items-end justify-between gap-2">
                    <EventTag event={st.event} />
                    <span className="flex items-center gap-2"><ImpactMeter score={st.impact} />
                      <ChevronRight size={14} className={clsx('text-ink-3 transition-transform', isOpen && 'rotate-90')} /></span>
                  </div>
                </button>
                {isOpen && (
                  <>
                    {st.signals.map((s) => <StoryDetail key={s.signal_id} s={s} />)}
                    {st.url && <a href={st.url} target="_blank" rel="noreferrer"
                      className="flex items-center gap-1 border-t border-line px-5 py-2 text-xs text-accent hover:underline">
                      Open source <ExternalLink size={12} /></a>}
                  </>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}

export default function SignalDesk() {
  const stream = useStream()
  const { data: stored } = useQuery({ queryKey: ['signals'], queryFn: () => api.signals({ limit: 400 }), refetchInterval: 20000 })
  const signals = stream.signals.length ? stream.signals : stored ?? []
  const stories = useMemo(() => groupStories(signals), [signals])

  const tone = useMemo(() => {
    let num = 0, den = 0
    signals.forEach((s) => { num += weight(s) * s.sentiment.score; den += weight(s) })
    return den ? num / den : 0
  }, [signals])

  // tone over stream time, in ~24 buckets, for the hero sparkline
  const spark = useMemo(() => {
    if (signals.length < 10) return []
    const sorted = [...signals].sort((a, b) => a.published_at.localeCompare(b.published_at))
    const size = Math.max(1, Math.ceil(sorted.length / 24))
    const out: [string, number][] = []
    for (let i = 0; i < sorted.length; i += size) {
      const g = sorted.slice(i, i + size)
      const w = g.reduce((a, s) => a + weight(s), 0)
      out.push([g[g.length - 1].published_at, w ? g.reduce((a, s) => a + weight(s) * s.sentiment.score, 0) / w : 0])
    }
    return out
  }, [signals])

  const byCompany = useMemo(() => {
    const m = new Map<string, { sum: number; w: number; n: number }>()
    signals.filter((s) => s.entity_type === 'company').forEach((s) => {
      const e = m.get(s.entity) ?? { sum: 0, w: 0, n: 0 }
      e.sum += weight(s) * s.sentiment.score; e.w += weight(s); e.n += 1
      m.set(s.entity, e)
    })
    return [...m.entries()].map(([t, e]) => ({ t, v: e.w ? e.sum / e.w : 0, n: e.n })).sort((a, b) => b.v - a.v)
  }, [signals])

  const mix = useMemo(() => {
    const m = new Map<string, number>()
    stories.forEach((s) => s.event !== 'none' && m.set(s.event, (m.get(s.event) ?? 0) + 1))
    return [...m.entries()].sort((a, b) => b[1] - a[1])
  }, [stories])

  const alerts = stories.filter((s) => s.impact >= 7).slice(0, 6)

  const heat = useMemo(() => {
    const comp = signals.filter((s) => s.entity_type === 'company')
    const days = [...new Set(comp.map((s) => s.published_at.slice(0, 10)))].sort().slice(-21)
    if (days.length < 2) return null
    const tickers = byCompany.map((b) => b.t)
    const cell = new Map<string, { s: number; n: number }>()
    comp.forEach((s) => {
      const k = `${s.entity}|${s.published_at.slice(0, 10)}`
      const c = cell.get(k) ?? { s: 0, n: 0 }
      c.s += s.sentiment.score; c.n += 1
      cell.set(k, c)
    })
    const data: [number, number, number][] = []
    tickers.forEach((t, y) => days.forEach((d, x) => { const c = cell.get(`${t}|${d}`); if (c) data.push([x, y, +(c.s / c.n).toFixed(3)]) }))
    return { days, tickers, data }
  }, [signals, byCompany])

  return (
    <>
      <PageTitle title="Desk">Unstructured text in, structured risk signals out.</PageTitle>
      <div className="mb-6 rounded-2xl border border-line bg-panel px-6 py-6">
        <Hero label="Market tone, credibility- and novelty-weighted" value={signals.length ? <Counter value={tone} format={(v) => signed(v)} duration={0.6} /> : '—'}
          detail={signals.length
            ? <>Across <b className="text-ink"><Counter value={stories.length} format={(v) => compact(Math.round(v))} /></b> stories and <b className="text-ink"><Counter value={stream.counter || signals.length} format={(v) => compact(Math.round(v))} /></b> entity signals;
              <b className="text-ink"> {alerts.length}</b> high-impact {alerts.length === 1 ? 'story' : 'stories'} in view.</>
            : 'Waiting for the first signals.'}
          aside={spark.length > 1 && (
            <div className="w-full max-w-[420px]">
              <Chart height={92} option={{
                grid: { left: 0, right: 0, top: 6, bottom: 0 },
                xAxis: { type: 'time', show: false }, yAxis: { type: 'value', min: -1, max: 1, show: false },
                tooltip: { trigger: 'axis', valueFormatter: (v) => signed(Number(v)) },
                series: [{ type: 'line', data: spark, symbol: 'none', smooth: 0.3, lineStyle: { width: 2, color: POS },
                  areaStyle: { color: POS, opacity: 0.1 },
                  markLine: { silent: true, symbol: 'none', lineStyle: { color: '#3a4250', width: 1, type: 'solid' }, label: { show: false }, data: [{ yAxis: 0 }] } }],
              }} />
            </div>
          )} />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Stream stories={stories} />
        <div className="space-y-6">
          <Panel title="Tone by company">
            {byCompany.length ? (
              <Chart height={Math.max(160, byCompany.length * 22 + 24)} option={{
                grid: { left: 4, right: 12, top: 4, bottom: 4, containLabel: true },
                xAxis: { type: 'value', min: -1, max: 1, splitNumber: 4, axisLabel: { formatter: (v: number) => signed(v, 1) } },
                yAxis: { type: 'category', inverse: true, data: byCompany.map((d) => d.t), axisLabel: { color: INK.secondary } },
                tooltip: { trigger: 'axis', formatter: (p: unknown) => { const d = byCompany[(p as { dataIndex: number }[])[0].dataIndex]; return `<b>${d.t}</b> ${signed(d.v)} · ${d.n} signals` } },
                series: [{ type: 'bar', ...barStyle(true), barMaxWidth: 12,
                  data: byCompany.map((d) => ({ value: +d.v.toFixed(3), itemStyle: { color: d.v >= 0 ? POS : NEG, borderRadius: d.v >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })) }],
              }} />
            ) : <Empty>No company signals yet</Empty>}
          </Panel>
          <Panel title="Event mix" meta="Stories by primary class">
            {mix.length ? (
              <Chart height={mix.length * 26 + 16} option={{
                grid: { left: 4, right: 28, top: 0, bottom: 0, containLabel: true },
                xAxis: { type: 'value', show: false },
                yAxis: { type: 'category', inverse: true, data: mix.map(([k]) => eventLabel(k)), axisLine: { show: false }, axisLabel: { color: INK.secondary } },
                tooltip: { trigger: 'axis' },
                series: [{ type: 'bar', ...barStyle(true), barMaxWidth: 10, data: mix.map(([, v]) => v), itemStyle: { color: SERIES[0], borderRadius: [0, 4, 4, 0] },
                  label: { show: true, position: 'right', color: INK.muted, fontSize: 11 } }],
              }} />
            ) : <Empty>—</Empty>}
          </Panel>
          <Panel title="High-impact stories" flush>
            {alerts.length ? (
              <ul className="divide-y divide-line">
                {alerts.map((a) => (
                  <li key={a.id} className="px-5 py-3">
                    <div className="flex items-center justify-between text-[11px] text-ink-3"><span>{when(a.at)}</span><ImpactMeter score={a.impact} /></div>
                    <div className="mt-1 line-clamp-2 text-[13px] text-ink">{a.headline}</div>
                    <div className="mt-1 text-xs text-ink-3">{eventLabel(a.event)} · {a.signals.map((s) => s.entity === 'MARKET' ? 'Market' : s.entity).join(', ')}</div>
                  </li>
                ))}
              </ul>
            ) : <Empty>Nothing at impact 7 or above.</Empty>}
          </Panel>
        </div>
      </div>

      {heat && (
        <Panel title="Tone by day" meta="Average entity sentiment; blue positive, red negative" className="mt-6">
          <Chart height={Math.max(220, heat.tickers.length * 20 + 70)} option={{
            grid: { left: 4, right: 8, top: 4, bottom: 40, containLabel: true },
            xAxis: { type: 'category', data: heat.days.map(day), splitLine: { show: false } },
            yAxis: { type: 'category', data: heat.tickers, inverse: true, splitLine: { show: false } },
            visualMap: { min: -1, max: 1, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 10, itemHeight: 160,
              inRange: { color: DIVERGING_STOPS }, textStyle: { color: INK.muted, fontSize: 11 }, text: ['positive', 'negative'] },
            tooltip: { formatter: (p: unknown) => { const v = (p as { value: [number, number, number] }).value; return `${heat.tickers[v[1]]} · ${heat.days[v[0]]}<br/>tone ${signed(v[2])}` } },
            series: [{ type: 'heatmap', data: heat.data, itemStyle: { borderColor: '#11151c', borderWidth: 2, borderRadius: 3 } }],
          }} />
        </Panel>
      )}
    </>
  )
}
