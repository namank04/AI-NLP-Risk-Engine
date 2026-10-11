import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { useCallback, useState } from 'react'
import { barStyle, Chart } from '../components/Chart'
import { PageTitle } from '../components/Layout'
import { Empty, Hero, Panel, Segmented, td, th } from '../components/ui'
import { eventLabel, pct, signed } from '../lib/format'
import { DEEMPH, INK, SEQUENTIAL_STOPS, SERIES } from '../lib/theme'
import LabelPage, { type GoldStats } from './LabelPage'

interface Decile { decile: number; n: number; median_abs_car_pct: number; mean_abs_scar: number; share_2sigma: number }
interface ImpactEval {
  n_train: number; n_test: number; test_start: string
  spearman: Record<string, number>; spearman_se: number
  bootstrap_learned_minus_rule: { mean_diff: number; ci95: [number, number] }
  adopted: string
  sentiment_direction: { spearman_sentiment_vs_signed_scar: number; hit_rate_strong: number }
  deciles_learned: Decile[]; deciles_rule_v0: Decile[]
  feature_importance: Record<string, number>
}
interface ClassMetric { precision: number; recall: number; f1: number; support: number }
interface EvalResult {
  n_items: number; n_gold: number
  linking?: { precision: number | null; missed_entities: number }
  sentiment?: { macro_f1_3class: number; exact_5pt: number; spearman: number }
  events?: { primary_accuracy: number; micro_f1: number; macro_f1: number | null; per_class: Record<string, ClassMetric>; confusion: Record<string, Record<string, number>> }
}

const PREDICTOR_LABEL: Record<string, string> = {
  rule_v0: 'Impact score (production)', learned: 'Gradient-boosted model', p_earnings: 'Earnings probability',
  news_count: 'Number of stories', max_abs_sentiment: 'Strongest sentiment',
}

function ImpactValidation() {
  const { data } = useQuery({ queryKey: ['impactEval'], queryFn: async () => (await fetch('/api/gold/impact')).json() as Promise<ImpactEval | null> })
  if (!data) return <Panel><Empty tall>Run python scripts/train_impact_model.py</Empty></Panel>
  const rho = data.spearman.rule_v0
  const b = data.bootstrap_learned_minus_rule
  const quint = (d: Decile[]) => [0, 2, 4, 6, 8].map((i) => (d[i].share_2sigma * d[i].n + d[i + 1].share_2sigma * d[i + 1].n) / (d[i].n + d[i + 1].n))
  const preds = Object.entries(data.spearman).sort((a, c) => c[1] - a[1])
  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-line bg-panel px-6 py-6">
        <Hero label="Rank correlation of the impact score with the realised abnormal move" value={signed(rho, 3)}
          detail={<>On {data.n_test.toLocaleString()} held-out stock-days from {data.test_start}: about {(rho / data.spearman_se).toFixed(1)} standard errors from zero.
            A gradient-boosted model scored {signed(data.spearman.learned, 3)}, but its gain ({signed(b.mean_diff, 3)}, 95% CI {signed(b.ci95[0], 3)} to {signed(b.ci95[1], 3)})
            is not significant, so production keeps the transparent score.</>} />
      </div>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Panel title="Share of stock-days with a move beyond 2σ, by predicted quintile" meta="Label: |2-day abnormal return| vs a SPY market model, in units of the stock's own residual volatility">
          <Chart height={280} option={{
            legend: { top: 0, left: 0, data: ['Impact score (production)', 'Gradient-boosted model'] },
            grid: { left: 4, right: 8, top: 34, bottom: 4, containLabel: true },
            xAxis: { type: 'category', data: ['Lowest', 'Q2', 'Q3', 'Q4', 'Highest'], splitLine: { show: false } },
            yAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${(100 * v).toFixed(0)}%` } },
            tooltip: { trigger: 'axis', valueFormatter: (v) => pct(Number(v), 1) },
            series: [
              { name: 'Impact score (production)', type: 'bar', ...barStyle(false), barGap: '20%', data: quint(data.deciles_rule_v0).map((v) => +v.toFixed(4)), itemStyle: { color: SERIES[0], borderRadius: [4, 4, 0, 0] } },
              { name: 'Gradient-boosted model', type: 'bar', ...barStyle(false), data: quint(data.deciles_learned).map((v) => +v.toFixed(4)), itemStyle: { color: DEEMPH, borderRadius: [4, 4, 0, 0] } },
            ],
          }} />
        </Panel>
        <Panel title="Every predictor tried" flush>
          <table className="w-full text-[13px]">
            <thead><tr className="border-b border-line"><th className={th}>Predictor</th><th className={clsx(th, 'text-right')}>Spearman ρ</th></tr></thead>
            <tbody className="tnum">
              {preds.map(([k, v]) => (
                <tr key={k} className="border-b border-line last:border-0">
                  <td className={clsx(td, k === 'rule_v0' ? 'text-ink' : 'text-ink-2')}>{PREDICTOR_LABEL[k] ?? k}</td>
                  <td className={clsx(td, 'text-right')}>{signed(v, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-line px-5 py-3 text-xs text-ink-3">Standard error ≈ {data.spearman_se.toFixed(3)}. Direction: day-average sentiment vs the signed move ρ = {signed(data.sentiment_direction.spearman_sentiment_vs_signed_scar, 3)}.</p>
        </Panel>
      </div>
    </div>
  )
}

function ModelQuality({ stats }: { stats: GoldStats | null }) {
  const { data: evals } = useQuery({ queryKey: ['evals'], queryFn: async () => (await fetch('/api/gold/eval')).json() as Promise<Record<string, EvalResult>> })
  const names = Object.keys(evals ?? {})
  const [sel, setSel] = useState<string | null>(null)
  const cur = sel ?? names[names.length - 1]
  const res = cur ? evals?.[cur] : undefined
  if (!names.length) {
    return (
      <Panel>
        <Empty tall>
          No scored predictors yet. Label items in the Labeling tab{stats ? ` (${stats.labeled} of ${stats.pool} done)` : ''}, then run
          python scripts/evaluate_models.py baseline teacher
        </Empty>
      </Panel>
    )
  }
  const classes = res?.events ? Object.entries(res.events.per_class).filter(([, m]) => m.support > 0).sort((a, b) => b[1].support - a[1].support) : []
  const conf = res?.events?.confusion
  const keys = conf ? Object.keys(conf).filter((k) => Object.values(conf[k]).some((v) => v) || Object.values(conf).some((r) => r[k])) : []
  const cells: [number, number, number][] = []
  keys.forEach((g, y) => keys.forEach((p, x) => { const v = conf![g][p]; if (v) cells.push([x, y, v]) }))
  return (
    <div className="space-y-6">
      <Panel title="Predictors on the gold set" flush>
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-line">
            {['Predictor', 'Items', 'Link precision', 'Sentiment F1', 'Sentiment ρ', 'Event accuracy', 'Event macro-F1'].map((h, i) => <th key={h} className={clsx(th, i && 'text-right')}>{h}</th>)}
          </tr></thead>
          <tbody className="tnum">
            {names.map((n) => { const r = evals![n]; return (
              <tr key={n} onClick={() => setSel(n)} className={clsx('cursor-pointer border-b border-line last:border-0 hover:bg-hover/50', n === cur && 'bg-accent-soft')}>
                <td className={clsx(td, 'text-ink')}>{n}</td><td className={clsx(td, 'text-right')}>{r.n_items}</td>
                <td className={clsx(td, 'text-right')}>{pct(r.linking?.precision)}</td><td className={clsx(td, 'text-right')}>{pct(r.sentiment?.macro_f1_3class)}</td>
                <td className={clsx(td, 'text-right')}>{r.sentiment ? signed(r.sentiment.spearman, 3) : '—'}</td>
                <td className={clsx(td, 'text-right')}>{pct(r.events?.primary_accuracy)}</td><td className={clsx(td, 'text-right')}>{pct(r.events?.macro_f1)}</td>
              </tr>) })}
          </tbody>
        </table>
      </Panel>
      {res?.events && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title={`F1 by event class · ${cur}`}>
            <Chart height={classes.length * 28 + 24} option={{
              grid: { left: 4, right: 40, top: 4, bottom: 4, containLabel: true },
              xAxis: { type: 'value', max: 1, axisLabel: { formatter: (v: number) => `${(100 * v).toFixed(0)}%` } },
              yAxis: { type: 'category', inverse: true, data: classes.map(([k]) => eventLabel(k)), axisLabel: { color: INK.secondary } },
              tooltip: { trigger: 'axis', formatter: (p: unknown) => { const [k, m] = classes[(p as { dataIndex: number }[])[0].dataIndex]; return `<b>${eventLabel(k)}</b><br/>P ${pct(m.precision)} · R ${pct(m.recall)} · F1 ${pct(m.f1)} · n ${m.support}` } },
              series: [{ type: 'bar', ...barStyle(true), barMaxWidth: 12, data: classes.map(([, m]) => m.f1), itemStyle: { color: SERIES[0], borderRadius: [0, 4, 4, 0] } }],
            }} />
          </Panel>
          <Panel title="Primary-class confusion" meta="Rows: gold label · columns: predicted">
            <Chart height={keys.length * 28 + 90} option={{
              grid: { left: 4, right: 8, top: 4, bottom: 4, containLabel: true },
              xAxis: { type: 'category', data: keys.map(eventLabel), axisLabel: { rotate: 35 }, splitLine: { show: false } },
              yAxis: { type: 'category', data: keys.map(eventLabel), inverse: true, splitLine: { show: false } },
              visualMap: { show: false, min: 0, max: Math.max(1, ...cells.map((c) => c[2])), inRange: { color: SEQUENTIAL_STOPS } },
              tooltip: { formatter: (p: unknown) => { const v = (p as { value: number[] }).value; return `${eventLabel(keys[v[1]])} → ${eventLabel(keys[v[0]])}: ${v[2]}` } },
              series: [{ type: 'heatmap', data: cells, label: { show: true, color: INK.primary, fontSize: 10 }, itemStyle: { borderColor: '#11151c', borderWidth: 2, borderRadius: 3 } }],
            }} />
          </Panel>
        </div>
      )}
    </div>
  )
}

export default function EvaluationPage() {
  const [tab, setTab] = useState<'impact' | 'quality' | 'label'>('impact')
  const [stats, setStats] = useState<GoldStats | null>(null)
  const { data: initial } = useQuery({ queryKey: ['goldStats'], queryFn: async () => (await fetch('/api/gold/stats')).json() as Promise<GoldStats> })
  const s = stats ?? initial ?? null
  const onStats = useCallback((x: GoldStats) => setStats(x), [])
  return (
    <>
      <PageTitle title="Evaluation">
        {s && <span className="tnum">Gold set: {s.labeled} labeled · {s.remaining} to go</span>}
      </PageTitle>
      <div className="mb-6">
        <Segmented value={tab} onChange={setTab}
          options={[{ value: 'impact', label: 'Impact validation' }, { value: 'quality', label: 'Model quality' }, { value: 'label', label: 'Labeling' }]} />
      </div>
      {tab === 'impact' && <ImpactValidation />}
      {tab === 'quality' && <ModelQuality stats={s} />}
      {tab === 'label' && <LabelPage onStats={onStats} />}
    </>
  )
}
