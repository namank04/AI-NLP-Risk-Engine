import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { IndexSnapshot } from '../api/types'
import { Chart } from '../components/Chart'
import { PageTitle } from '../components/Layout'
import { Counter, STATIC } from '../components/motion'
import { Empty, Fact, Hero, Panel, td, th, ToneBar } from '../components/ui'
import { useStream } from '../hooks/useStream'
import { pct, pp, signed, when } from '../lib/format'
import { DEEMPH, DIVERGING_STOPS, divergingColor, INK, POS, SERIES } from '../lib/theme'

const SECTOR_ORDER = ['Information Technology', 'Communication Services', 'Consumer Discretionary', 'Financials', 'Energy',
  'Health Care', 'Industrials', 'Consumer Staples', 'Utilities', 'Materials', 'Real Estate']

interface Backtest {
  metrics: Record<string, Record<string, number>>
  ic: Record<string, { mean_ic: number; ic_t_stat: number; hit_rate: number; n_days: number }>
  nav: Record<string, number | string>[]
  config: { te_budget?: number; cost_bps: number; band: number; bl?: { te_budget: number; ic: number } }
}
interface ImpactEval { sentiment_direction: { spearman_sentiment_vs_signed_scar: number } }

const STRATEGY: Record<string, string> = {
  black_litterman: 'Black-Litterman', engine_tilt: 'Exponential tilt', naive_tilt: 'Naive document sentiment',
  equal_weight: 'Equal weight', cap_weight: 'Cap-weight benchmark',
}

function TeMeter({ te, budget }: { te: number; budget: number }) {
  const f = Math.min(1, te / budget)
  return (
    <div className="w-60">
      <div className="flex justify-between text-xs"><span className="text-ink-3">Ex-ante tracking error</span>
        <span className="tnum text-ink-2">{pct(te, 2)} of {pct(budget, 0)}</span></div>
      <div className="mt-2 h-2 rounded-full bg-[#1b2a42]">
        <div className="h-2 rounded-full bg-accent" style={{ width: `${f * 100}%` }} />
      </div>
    </div>
  )
}

function Evidence() {
  const { data: bt } = useQuery({ queryKey: ['backtest'], queryFn: async () => (await fetch('/api/gold/backtest')).json() as Promise<Backtest | null> })
  const { data: ie } = useQuery({ queryKey: ['impactEval'], queryFn: async () => (await fetch('/api/gold/impact')).json() as Promise<ImpactEval | null> })
  if (!bt) return null
  const order = ['black_litterman', 'cap_weight', 'engine_tilt', 'naive_tilt', 'equal_weight']
  const color: Record<string, string> = { black_litterman: POS, cap_weight: INK.secondary }
  const bl = bt.metrics.black_litterman
  const contemporaneous = ie?.sentiment_direction.spearman_sentiment_vs_signed_scar
  return (
    <section className="mt-10">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold">Evidence · walk-forward 2022–2023</h2>
        <span className="text-xs text-ink-3">formed at close D from signals dated ≤ D−1 · {bt.config.cost_bps} bp costs · {pct(bt.config.band, 0)} no-trade band</span>
      </div>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_400px]">
        <Panel title="Growth of $1" meta="Black-Litterman and the benchmark highlighted">
          <Chart height={320} option={{
            legend: { top: 0, left: 0, data: order.map((s) => STRATEGY[s]) },
            grid: { left: 4, right: 12, top: 34, bottom: 4, containLabel: true },
            tooltip: { trigger: 'axis', order: 'valueDesc', valueFormatter: (v) => `$${Number(v).toFixed(3)}` },
            xAxis: { type: 'time', splitLine: { show: false } },
            yAxis: { type: 'value', scale: true, axisLabel: { formatter: (v: number) => `$${v.toFixed(2)}` } },
            series: [...order].reverse().map((s) => ({
              name: STRATEGY[s], type: 'line' as const, symbol: 'none', z: color[s] ? 3 : 1,
              lineStyle: { width: color[s] ? 2 : 1.2, color: color[s] ?? DEEMPH }, itemStyle: { color: color[s] ?? DEEMPH },
              data: bt.nav.map((r) => [r.date, r[s]]),
            })),
          }} />
        </Panel>
        <Panel title="What the test shows">
          <ol className="space-y-4 text-[13px] leading-relaxed text-ink-2">
            <li><b className="text-ink">The risk model holds its budget.</b> Realised tracking error was {pct(bl.tracking_error, 1)} against a {pct(bt.config.bl?.te_budget ?? 0.02, 0)} ex-ante
              budget, with {pct(bl.avg_daily_turnover, 1)} daily turnover.</li>
            <li><b className="text-ink">Active return was small and not significant.</b> {signed(100 * bl.active_return, 1)}% a year, information ratio {bl.information_ratio.toFixed(2)}
              (t = {bl.active_t_stat.toFixed(2)}).</li>
            <li><b className="text-ink">News is priced within a day.</b> Sentiment agrees with the same-day abnormal move
              {contemporaneous !== undefined && <> (ρ = {signed(contemporaneous, 3)})</>}, but traded a day later it slightly reverses
              (IC {signed(bt.ic.engine.mean_ic, 3)}, t = {bt.ic.engine.ic_t_stat.toFixed(1)}). The live index therefore rebalances on 15-minute timestamps, not daily.</li>
          </ol>
        </Panel>
      </div>
      <Panel title="Strategy statistics" flush className="mt-6">
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead><tr className="border-b border-line">
              {['Strategy', 'Return p.a.', 'Volatility', 'Sharpe', 'Max drawdown', 'Active p.a.', 'Tracking error', 'Info. ratio', 'Turnover/day']
                .map((h, i) => <th key={h} className={clsx(th, i && 'text-right')}>{h}</th>)}
            </tr></thead>
            <tbody className="tnum">
              {order.map((s) => { const m = bt.metrics[s]; return (
                <tr key={s} className="border-b border-line last:border-0">
                  <td className={clsx(td, 'text-ink')}>
                    <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: color[s] ?? DEEMPH }} />{STRATEGY[s]}</td>
                  <td className={clsx(td, 'text-right')}>{pct(m.ann_return)}</td>
                  <td className={clsx(td, 'text-right text-ink-2')}>{pct(m.ann_vol)}</td>
                  <td className={clsx(td, 'text-right')}>{m.sharpe.toFixed(2)}</td>
                  <td className={clsx(td, 'text-right text-ink-2')}>{pct(m.max_drawdown)}</td>
                  <td className={clsx(td, 'text-right')}>{m.active_return === undefined || m.active_return === null ? '—' : pct(m.active_return, 2)}</td>
                  <td className={clsx(td, 'text-right text-ink-2')}>{m.tracking_error ? pct(m.tracking_error, 1) : '—'}</td>
                  <td className={clsx(td, 'text-right')}>{m.information_ratio !== undefined && m.information_ratio !== null ? m.information_ratio.toFixed(2) : '—'}</td>
                  <td className={clsx(td, 'text-right text-ink-2')}>{m.avg_daily_turnover ? pct(m.avg_daily_turnover, 1) : '—'}</td>
                </tr>) })}
            </tbody>
          </table>
        </div>
      </Panel>
    </section>
  )
}

function Focus({ ticker, history }: { ticker: string; history: IndexSnapshot[] }) {
  const w = history.map((h) => [h.ts, +(100 * (h.weights[ticker] ?? 0)).toFixed(3)])
  const b = history.map((h) => [h.ts, +(100 * (h.benchmark[ticker] ?? 0)).toFixed(3)])
  const x = history.map((h) => [h.ts, +(h.signals[ticker]?.tilt ?? 0).toFixed(3)])
  return (
    <Panel title={`${ticker}: weight against benchmark`} meta="The signal driving the tilt is plotted underneath on its own scale">
      <Chart height={190} option={{
        legend: { top: 0, right: 0, data: ['Portfolio', 'Benchmark'] },
        grid: { left: 4, right: 12, top: 28, bottom: 4, containLabel: true },
        tooltip: { trigger: 'axis', valueFormatter: (v) => `${Number(v).toFixed(2)}%` },
        xAxis: { type: 'time', splitLine: { show: false } },
        yAxis: { type: 'value', scale: true, axisLabel: { formatter: '{value}%' } },
        series: [
          { name: 'Benchmark', type: 'line', step: 'end', symbol: 'none', data: b, lineStyle: { width: 1.5, color: INK.muted }, itemStyle: { color: INK.muted } },
          { name: 'Portfolio', type: 'line', step: 'end', symbol: 'none', data: w, lineStyle: { width: 2, color: POS }, itemStyle: { color: POS } },
        ],
      }} />
      <div className="mt-2 text-xs text-ink-3">Sentiment state (shrunk, decayed)</div>
      <Chart height={110} option={{
        grid: { left: 4, right: 12, top: 8, bottom: 4, containLabel: true },
        tooltip: { trigger: 'axis', valueFormatter: (v) => signed(Number(v), 3) },
        xAxis: { type: 'time', splitLine: { show: false } },
        yAxis: { type: 'value', min: -1, max: 1, splitNumber: 2 },
        series: [{ type: 'line', step: 'end', symbol: 'none', data: x, lineStyle: { width: 2, color: SERIES[1] },
          areaStyle: { color: SERIES[1], opacity: 0.1 } }],
      }} />
    </Panel>
  )
}

export default function IndexPage() {
  const { status } = useStream()
  const { data: history } = useQuery({ queryKey: ['indexHistory'], queryFn: () => api.indexHistory(5000),
    refetchInterval: status?.running ? 2500 : 20000 })
  const { data: universe } = useQuery({ queryKey: ['universe'], queryFn: api.universe })
  const [focus, setFocus] = useState('XOM')

  const sector = useMemo(() => Object.fromEntries((universe ?? []).map((u) => [u.ticker, u.sector])), [universe])
  const tickers = useMemo(() => [...(universe ?? [])].sort((a, b) => SECTOR_ORDER.indexOf(a.sector) - SECTOR_ORDER.indexOf(b.sector)
    || b.benchmark_weight - a.benchmark_weight).map((u) => u.ticker), [universe])
  const last = history?.[history.length - 1]
  const rebalances = (history ?? []).filter((h) => h.reason !== 'inception')

  const rows = useMemo(() => last ? tickers.map((t) => ({
    t, w: last.weights[t] ?? 0, b: last.benchmark[t] ?? 0, a: (last.weights[t] ?? 0) - (last.benchmark[t] ?? 0),
    s: last.signals[t]?.sentiment ?? 0, n: last.signals[t]?.n ?? 0, alpha: last.alpha_bps?.[t],
  })) : [], [last, tickers])
  const race = STATIC ? [...rows].sort((x, y) => Math.abs(y.a) - Math.abs(x.a)) : rows
  const activeShare = rows.reduce((a, r) => a + Math.abs(r.a), 0) / 2
  const maxActive = Math.max(0.005, ...rows.map((r) => Math.abs(r.a)))

  const heat = useMemo(() => {
    if (!history || history.length < 2) return null
    const snaps = history.slice(-120)
    const data: [number, number, number][] = []
    let mx = 0.002
    snaps.forEach((h, x) => tickers.forEach((t, y) => {
      const a = (h.weights[t] ?? 0) - (h.benchmark[t] ?? 0)
      mx = Math.max(mx, Math.abs(a))
      data.push([x, y, +(100 * a).toFixed(3)])
    }))
    return { snaps, data, max: +(100 * mx).toFixed(2) }
  }, [history, tickers])

  return (
    <>
      <PageTitle title="Index">Module A · sentiment-tilted index over 20 S&amp;P 100 names</PageTitle>
      <div className="mb-6 rounded-2xl border border-line bg-panel px-6 py-6">
        <Hero label="Active share against the cap-weighted benchmark" value={last ? <Counter value={100 * activeShare} format={(v) => `${v.toFixed(1)}%`} /> : '—'}
          detail={last ? <>{last.method === 'black_litterman' ? 'Black-Litterman with Grinold-sized views' : 'Exponential tilt'} ·
            last rebalanced {when(last.ts)}{last.reason !== 'inception' && <> · {pct(last.turnover, 1)} one-way turnover</>}</>
            : 'Start a replay to stream sentiment into the index.'}
          aside={<div className="flex items-end gap-8">
            {last?.ex_ante_te !== null && last?.ex_ante_te !== undefined && <TeMeter te={last.ex_ante_te} budget={0.02} />}
            <Fact label="Rebalances" value={rebalances.length} />
          </div>} />
      </div>

      <div className="mb-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel title="Active weights now" meta={last ? `Portfolio minus benchmark, largest bets first, re-sorted at every rebalance · ${when(last.ts)}` : undefined}>
          {race.length ? (
            <Chart merge height={tickers.length * 19 + 30} onClick={(p) => { const n = (p as { name?: string }).name; if (n) setFocus(n) }} option={{
              grid: { left: 4, right: 56, top: 4, bottom: 4, containLabel: true },
              xAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${v.toFixed(1)}pp` }, splitNumber: 4 },
              yAxis: { type: 'category', inverse: true, data: race.map((r) => r.t), animationDuration: 300, animationDurationUpdate: 600,
                axisLabel: { color: INK.secondary }, splitLine: { show: false } },
              tooltip: { trigger: 'axis', formatter: (p: unknown) => { const d = race[(p as { dataIndex: number }[])[0].dataIndex]
                return `<b>${d.t}</b> active ${signed(100 * d.a, 2)} pp<br/>sentiment ${signed(d.s)} · ${d.n} signals` } },
              animationDuration: 0, animationDurationUpdate: 700, animationEasingUpdate: 'cubicInOut',
              series: [{ type: 'bar', realtimeSort: true, barMaxWidth: 11,
                data: race.map((r) => ({ value: +(100 * r.a).toFixed(3), itemStyle: { color: divergingColor(r.a, maxActive), borderRadius: r.a >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })),
                label: { show: true, position: 'right', color: INK.muted, fontSize: 10, valueAnimation: true, formatter: (p: unknown) => `${signed((p as { value: number }).value, 2)}` } }],
            }} />
          ) : <Empty tall>Start a replay; the index rebalances as sentiment arrives.</Empty>}
        </Panel>
        {history && history.length > 1 ? <Focus ticker={focus} history={history} /> : <Panel title="Focus"><Empty tall>Pick a name once rebalances arrive.</Empty></Panel>}
      </div>

      <div>
        <Panel title="Active weight over time" meta={heat ? `Last ${heat.snaps.length} rebalances · rows grouped by sector · click a row to focus` : undefined}>
          {heat ? (
            <Chart height={tickers.length * 20 + 80} onClick={(p) => { const v = (p as { value?: number[] }).value; if (v) setFocus(tickers[v[1]]) }} option={{
              grid: { left: 4, right: 8, top: 4, bottom: 44, containLabel: true },
              xAxis: { type: 'category', data: heat.snaps.map((h) => when(h.ts)), axisLabel: { show: false }, splitLine: { show: false }, axisLine: { show: false } },
              yAxis: { type: 'category', data: tickers, inverse: true, splitLine: { show: false },
                axisLabel: { color: INK.secondary, formatter: (t: string) => (t === focus ? `{f|${t}}` : t), rich: { f: { color: INK.primary, fontWeight: 600 } } } },
              visualMap: { min: -heat.max, max: heat.max, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 10, itemHeight: 160,
                inRange: { color: DIVERGING_STOPS }, textStyle: { color: INK.muted, fontSize: 11 }, text: ['overweight', 'underweight'], formatter: (v: unknown) => `${Number(v).toFixed(1)}pp` },
              tooltip: { formatter: (p: unknown) => { const v = (p as { value: number[] }).value; return `${tickers[v[1]]} · ${when(heat.snaps[v[0]].ts)}<br/>active ${signed(v[2], 2)} pp` } },
              series: [{ type: 'heatmap', data: heat.data, itemStyle: { borderColor: '#11151c', borderWidth: 1 }, progressive: 0 }],
            }} />
          ) : <Empty tall>No rebalances yet.</Empty>}
        </Panel>
      </div>

      <Panel title="Positioning" meta="Current weights; select a row to focus it" flush className="mt-6">
        {rows.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead><tr className="border-b border-line">
                {['Name', 'Sector', 'Benchmark', 'Portfolio', 'Active', 'Sentiment state', 'View α (bp/yr)', 'Signals'].map((h, i) =>
                  <th key={h} className={clsx(th, i > 1 && i !== 5 && 'text-right')}>{h}</th>)}
              </tr></thead>
              <tbody className="tnum">
                {rows.map((r) => (
                  <tr key={r.t} onClick={() => setFocus(r.t)} aria-selected={r.t === focus}
                    className={clsx('cursor-pointer border-b border-line last:border-0 hover:bg-hover/50', r.t === focus && 'bg-accent-soft')}>
                    <td className={clsx(td, 'font-semibold text-ink')}>{r.t}</td>
                    <td className={clsx(td, 'text-ink-3')}>{sector[r.t]}</td>
                    <td className={clsx(td, 'text-right text-ink-2')}>{pct(r.b, 2)}</td>
                    <td className={clsx(td, 'text-right')}>{pct(r.w, 2)}</td>
                    <td className={clsx(td, 'text-right')}>
                      <span className="inline-flex items-center justify-end gap-2">
                        <svg width="60" height="8" aria-hidden>
                          <rect x="0" y="3" width="60" height="2" rx="1" fill="#262c37" />
                          <rect x={r.a >= 0 ? 30 : 30 - (Math.abs(r.a) / maxActive) * 30} y="1" width={Math.max(1.5, (Math.abs(r.a) / maxActive) * 30)} height="6" rx="2" fill={divergingColor(r.a, maxActive)} />
                          <rect x="29.5" y="0" width="1" height="8" fill="#4a5262" />
                        </svg>
                        <span className="w-16">{pp(r.a, 2)}</span>
                      </span>
                    </td>
                    <td className={td}><ToneBar value={r.s} /></td>
                    <td className={clsx(td, 'text-right text-ink-2')}>{r.alpha === undefined ? '—' : signed(r.alpha, 0)}</td>
                    <td className={clsx(td, 'text-right text-ink-3')}>{r.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty>—</Empty>}
      </Panel>

      <Evidence />
    </>
  )
}
