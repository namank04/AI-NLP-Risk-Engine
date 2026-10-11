import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ExternalLink } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { StressRun, StressSummary } from '../api/types'
import { CapitalGauge } from '../components/CapitalGauge'
import { barStyle, Chart } from '../components/Chart'
import { Counter } from '../components/motion'
import { PageTitle } from '../components/Layout'
import { Button, Empty, EventTag, Fact, ImpactMeter, Panel, Segmented, Select, StatusPill, td, th } from '../components/ui'
import { useStream } from '../hooks/useStream'
import { instrumentLabel, money, pct, titleCase, when } from '../lib/format'
import { DEEMPH, INK, NEG, POS, SEQUENTIAL_STOPS, SERIES, STATUS } from '../lib/theme'

const RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
const COUNTRIES = ['EU', 'GB', 'JP', 'CN', 'IN', 'BR', 'MX', 'SA', 'TW']
const FACTORS: [string, string][] = [['rates', 'Rates'], ['spreads', 'Credit spreads'], ['equity', 'Equity'], ['fx', 'FX'],
  ['credit', 'Credit losses'], ['interaction', 'Interaction']]

const compliance = (s: StressSummary) =>
  s.breaches_minimum ? { level: 'critical' as const, text: 'Below the 4.5% minimum' }
    : s.breaches_buffer ? { level: 'warning' as const, text: 'Inside the capital buffers' }
      : { level: 'good' as const, text: 'Above requirement' }

function Builder({ onRun }: { onRun: (id: string) => void }) {
  const qc = useQueryClient()
  const { data: sc } = useQuery({ queryKey: ['scenarios'], queryFn: api.scenarios })
  const [kind, setKind] = useState<'systemic' | 'idiosyncratic'>('systemic')
  const [cls, setCls] = useState('credit_event')
  const [severity, setSeverity] = useState(1.0)
  const [scope, setScope] = useState<string[]>([])
  const [cp, setCp] = useState('')
  const [subtype, setSubtype] = useState('rating_action')
  useEffect(() => { if (!cp && sc?.counterparties.length) setCp(sc.counterparties.find((c) => c.includes('Boeing')) ?? sc.counterparties[0]) }, [sc, cp])
  const { data: curve } = useQuery({ queryKey: ['curve', cls, scope.join(',')], enabled: kind === 'systemic',
    queryFn: () => api.stressCurve(cls, scope) })
  const projected = useMemo(() => {
    if (!curve || kind !== 'systemic') return null
    const p = curve.points
    const j = p.findIndex((x) => x.severity >= severity)
    if (j <= 0) return p[Math.max(0, j)]?.cet1_ratio ?? p[p.length - 1].cet1_ratio
    const t = (severity - p[j - 1].severity) / (p[j].severity - p[j - 1].severity)
    return p[j - 1].cet1_ratio + t * (p[j].cet1_ratio - p[j - 1].cet1_ratio)
  }, [curve, severity, kind])
  const m = useMutation({
    mutationFn: () => api.runStress(kind === 'systemic' ? { kind, event_class: cls, severity, scope_countries: scope } : { kind, counterparty: cp, severity, subtype }),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ['stressRuns'] }); onRun(r.run_id) },
  })
  return (
    <Panel title="Run a scenario">
      <div className="space-y-3">
        <Segmented size="sm" value={kind} onChange={setKind} options={[{ value: 'systemic', label: 'Systemic' }, { value: 'idiosyncratic', label: 'Single name' }]} />
        {kind === 'systemic' ? (
          <>
            <Select label="Scenario" value={cls} onChange={setCls} className="w-full">
              {Object.entries(sc?.systemic ?? {}).map(([k, v]) => <option key={k} value={k}>{v.name}</option>)}
            </Select>
            <div>
              <div className="mb-1.5 text-xs text-ink-3">Countries in scope</div>
              <div className="flex flex-wrap gap-1">
                {COUNTRIES.map((c) => (
                  <button key={c} aria-pressed={scope.includes(c)} onClick={() => setScope((s) => (s.includes(c) ? s.filter((x) => x !== c) : [...s, c]))}
                    className={clsx('rounded-md border px-2 py-0.5 text-xs', scope.includes(c) ? 'border-accent bg-accent-soft text-ink' : 'border-line text-ink-3 hover:text-ink-2')}>{c}</button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <>
            <Select label="Counterparty" value={cp} onChange={setCp} className="w-full">
              {(sc?.counterparties ?? []).filter((c) => !c.startsWith('Government') && !c.startsWith('ALM')).map((c) => <option key={c} value={c}>{c}</option>)}
            </Select>
            <Select label="Event" value={subtype} onChange={setSubtype} className="w-full">
              {['rating_action', 'liquidity_stress', 'covenant_breach', 'restructuring', 'failure_to_pay', 'bankruptcy'].map((s) => <option key={s} value={s}>{titleCase(s)}</option>)}
            </Select>
          </>
        )}
        <label className="block">
          <span className="flex justify-between text-xs"><span className="text-ink-3">Severity</span><span className="tnum text-ink-2">{Math.round(severity * 100)}% of the episode</span></span>
          <input type="range" min={0.1} max={1.5} step={0.05} value={severity} onChange={(e) => setSeverity(Number(e.target.value))} className="mt-2 w-full accent-[#3987e5]" />
        </label>
        {projected !== null && curve && (
          <div className="rounded-xl border border-line bg-page/60 px-3 py-2.5">
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-ink-3">Projected CET1</span>
              <span className={clsx('text-base font-semibold', projected < curve.minimum ? 'text-critical' : projected < curve.requirement ? 'text-warning' : 'text-ink')}>
                <Counter value={100 * projected} start={100 * curve.points[0].cet1_ratio} duration={0.35} format={(v) => `${v.toFixed(2)}%`} />
              </span>
            </div>
            <div className="relative mt-2 h-1.5 rounded-full bg-raised">
              <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-300"
                style={{ width: `${Math.min(100, (projected / 0.2) * 100)}%`, background: projected < curve.requirement ? STATUS.warning : POS }} />
              <span className="absolute -top-1 h-3.5 w-px bg-ink-3" style={{ left: `${(curve.requirement / 0.2) * 100}%` }} title="Requirement" />
            </div>
            <div className="mt-1.5 text-[11px] text-ink-3">From the model's severity curve; run to see the full breakdown.</div>
          </div>
        )}
        <Button onClick={() => m.mutate()} disabled={m.isPending} className="w-full">{m.isPending ? 'Running…' : 'Run stress test'}</Button>
      </div>
    </Panel>
  )
}

function Runs({ runs, selected, onSelect }: { runs: StressRun[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <Panel title="Runs" meta="Triggered by the engine or run by hand" flush>
      {runs.length === 0 ? <Empty>No runs yet</Empty> : (
        <ul className="max-h-[440px] divide-y divide-line overflow-y-auto">
          {runs.map((r) => {
            const trig = r.scenario.trigger as Record<string, string | boolean>
            const c = compliance(r.summary)
            return (
              <li key={r.run_id}>
                <button onClick={() => onSelect(r.run_id)} aria-current={selected === r.run_id}
                  className={clsx('w-full px-5 py-3 text-left hover:bg-hover/50', selected === r.run_id && 'bg-accent-soft')}>
                  <div className="flex items-center justify-between text-[11px] text-ink-3">
                    <span>{trig.manual ? 'Manual' : when(String(trig.published_at ?? r.at))}</span>
                    <span className="tnum">{pct(r.summary.cet1_ratio_stressed, 1)} CET1</span>
                  </div>
                  <div className="mt-0.5 text-[13px] text-ink">{r.scenario.name.replace(/ \(.*\)$/, '')}</div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-xs text-ink-3">
                    {c.level !== 'good' && <span className="h-1.5 w-1.5 rounded-full" style={{ background: STATUS[c.level] }} />}
                    {money(r.summary.total_loss)} loss · {Math.round(r.scenario.severity * 100)}% severity
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}

function Waterfall({ run }: { run: StressRun }) {
  const steps = FACTORS.map(([k, label]) => ({ label, v: -(run.attribution[k] ?? 0) }))
  const start = run.summary.portfolio_value
  const labels = ['Before', ...steps.map((s) => s.label), 'After']
  let level = start
  const base: number[] = [0], up: (number | '-')[] = [start], down: (number | '-')[] = ['-']
  steps.forEach((s) => {
    if (s.v >= 0) { base.push(level); up.push(s.v); down.push('-') } else { base.push(level + s.v); up.push('-'); down.push(-s.v) }
    level += s.v
  })
  base.push(0); up.push(level); down.push('-')
  const floor = Math.min(level, ...base.filter((b, i) => i > 0 && i < labels.length - 1 && b > 0)) * 0.99
  return (
    <Chart height={280} option={{
      grid: { left: 4, right: 8, top: 12, bottom: 4, containLabel: true },
      xAxis: { type: 'category', data: labels, axisLabel: { interval: 0, fontSize: 11 }, splitLine: { show: false } },
      yAxis: { type: 'value', min: Math.floor(floor), splitNumber: 4, axisLabel: { formatter: (v: number) => money(v, 1) } },
      tooltip: { trigger: 'axis', formatter: (p: unknown) => {
        const i = (p as { dataIndex: number }[])[0].dataIndex
        if (i === 0 || i === labels.length - 1) return `${labels[i]}: ${money(Number(up[i]), 2)}`
        return `${labels[i]}: ${steps[i - 1].v >= 0 ? '+' : ''}${money(steps[i - 1].v, 1)}`
      } },
      series: [
        { type: 'bar', stack: 'w', silent: true, itemStyle: { color: 'transparent' }, data: base },
        { type: 'bar', stack: 'w', name: 'gain', barMaxWidth: 34, itemStyle: { borderRadius: [4, 4, 0, 0] },
          data: up.map((v, i) => (v === '-' ? '-' : { value: v, itemStyle: { color: i === 0 || i === labels.length - 1 ? '#4a5262' : POS } })) },
        { type: 'bar', stack: 'w', name: 'loss', barMaxWidth: 34, itemStyle: { color: NEG, borderRadius: [0, 0, 4, 4] }, data: down },
      ],
    }} />
  )
}

function BreakingPoint({ run }: { run: StressRun }) {
  const systemic = run.scenario.kind === 'systemic'
  const { data: curve } = useQuery({ queryKey: ['curve', run.scenario.key, run.scenario.scope_countries.join(',')], enabled: systemic,
    queryFn: () => api.stressCurve(run.scenario.key, run.scenario.scope_countries) })
  if (!systemic) return <Empty tall>Breaking-point analysis applies to systemic scenarios.</Empty>
  if (!curve) return <Empty tall>Computing…</Empty>
  const pts = curve.points.map((p) => [Math.round(p.severity * 100), +(100 * p.cet1_ratio).toFixed(2)])
  const here = [Math.round(run.scenario.severity * 100), +(100 * run.summary.cet1_ratio_stressed).toFixed(2)]
  return (
    <>
      <Chart height={240} option={{
        grid: { left: 4, right: 12, top: 14, bottom: 22, containLabel: true },
        xAxis: { type: 'value', min: 0, max: 150, name: 'severity, % of the historical episode', nameLocation: 'middle', nameGap: 26,
          nameTextStyle: { color: INK.muted, fontSize: 11 }, axisLabel: { formatter: '{value}%' }, splitLine: { show: false } },
        yAxis: { type: 'value', min: 0, axisLabel: { formatter: '{value}%' } },
        tooltip: { trigger: 'axis', formatter: (p: unknown) => { const v = (p as { value: number[] }[])[0].value; return `severity ${v[0]}%<br/>CET1 ${v[1].toFixed(2)}%` } },
        series: [
          { type: 'line', data: pts, symbol: 'none', lineStyle: { width: 2, color: POS },
            markLine: { silent: true, symbol: 'none', animation: false, label: { position: 'insideEndTop', distance: [0, 4], color: INK.secondary, fontSize: 11 },
              data: [
                { yAxis: 100 * curve.requirement, lineStyle: { color: STATUS.warning, width: 1, type: 'solid' }, label: { formatter: `Requirement ${(100 * curve.requirement).toFixed(1)}%` } },
                { yAxis: 100 * curve.minimum, lineStyle: { color: STATUS.critical, width: 1, type: 'solid' }, label: { formatter: `Minimum ${(100 * curve.minimum).toFixed(1)}%` } },
              ] } },
          { type: 'scatter', data: [here], symbolSize: 11, itemStyle: { color: POS, borderColor: '#11151c', borderWidth: 2 }, z: 5 },
        ],
      }} />
      <p className="mt-1 text-xs text-ink-2">
        {curve.breach_buffer
          ? <>Capital falls below requirement at <b className="text-ink">{Math.round(curve.breach_buffer.severity * 100)}%</b> of the episode, about engine impact {curve.breach_buffer.impact_equivalent.toFixed(1)}.</>
          : <>The requirement holds even at 150% of the episode.</>}
        {curve.breach_minimum && <> The 4.5% minimum breaks at {Math.round(curve.breach_minimum.severity * 100)}%.</>}
      </p>
    </>
  )
}

function LossBars({ rows, keyName }: { rows: Record<string, string | number>[]; keyName: string }) {
  const label = keyName === 'instrument' ? instrumentLabel : titleCase
  const r = rows.filter((x) => Math.abs(Number(x.loss)) > 0.05).slice(0, 9)
  if (!r.length) return <Empty>No losses here</Empty>
  return (
    <Chart height={r.length * 28 + 24} option={{
      grid: { left: 4, right: 40, top: 4, bottom: 4, containLabel: true },
      xAxis: { type: 'value', splitNumber: 2, axisLabel: { formatter: (v: number) => money(v, 0), hideOverlap: true } },
      yAxis: { type: 'category', inverse: true, data: r.map((x) => label(String(x[keyName]))), axisLabel: { color: INK.secondary } },
      tooltip: { trigger: 'axis', formatter: (p: unknown) => { const x = r[(p as { dataIndex: number }[])[0].dataIndex]
        return `<b>${label(String(x[keyName]))}</b><br/>loss ${money(Number(x.loss), 1)} · MTM ${money(Number(x.mtm_pnl), 1)} · ECL +${money(Number(x.ecl_increase), 1)}` } },
      series: [{ type: 'bar', ...barStyle(true), barMaxWidth: 12,
        data: r.map((x) => ({ value: +Number(x.loss).toFixed(2), itemStyle: { color: Number(x.loss) >= 0 ? NEG : POS, borderRadius: Number(x.loss) >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })) }],
    }} />
  )
}

function Migration({ run }: { run: StressRun }) {
  const data = (run.by_rating_migration ?? []).map((m) => [RATINGS.indexOf(m.rating_stressed), RATINGS.indexOf(m.rating_orig), Math.round(m.exposure)])
  const max = Math.max(1, ...data.map((d) => d[2]))
  return (
    <Chart height={260} option={{
      grid: { left: 4, right: 8, top: 8, bottom: 26, containLabel: true },
      xAxis: { type: 'category', data: RATINGS, name: 'stressed grade', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: INK.muted, fontSize: 11 }, splitLine: { show: false } },
      yAxis: { type: 'category', data: RATINGS, inverse: true, splitLine: { show: false } },
      visualMap: { show: false, min: 0, max, inRange: { color: SEQUENTIAL_STOPS } },
      tooltip: { formatter: (p: unknown) => { const v = (p as { value: number[] }).value; return `${RATINGS[v[1]]} → ${RATINGS[v[0]]}<br/>${money(v[2])}` } },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: '#11151c', borderWidth: 2, borderRadius: 3 },
        label: { show: true, fontSize: 10, color: INK.primary, formatter: (p: unknown) => money((p as { value: number[] }).value[2], 0) } }],
    }} />
  )
}

function Library() {
  const [sev, setSev] = useState(1)
  const { data: lib } = useQuery({ queryKey: ['library'], queryFn: api.stressLibrary })
  const { data: cmp } = useQuery({ queryKey: ['compare', sev], queryFn: () => api.stressCompare(sev) })
  const curves = useQuery({ queryKey: ['allCurves'], queryFn: async () => Object.fromEntries(await Promise.all(
    (lib ?? []).map(async (l) => [l.key, await api.stressCurve(l.key)] as const))), enabled: !!lib })
  if (!lib) return null
  const fmtBp = (v?: number) => (v === undefined ? '—' : `${v > 0 ? '+' : ''}${Math.round(v)}`)
  const fmtPc = (v?: number) => (v === undefined ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(0)}%`)
  return (
    <section className="mt-10">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-base font-semibold">Scenario library</h2>
        <div className="flex items-center gap-3 text-xs text-ink-3">
          <span>Each template is calibrated to market moves measured over its anchor episode. Compare at</span>
          <Segmented size="sm" value={sev} onChange={setSev} options={[{ value: 0.5, label: '50%' }, { value: 1, label: '100%' }, { value: 1.5, label: '150%' }]} />
        </div>
      </div>
      <Panel flush>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead><tr className="border-b border-line">
              {['Scenario', 'Anchor episode', '10Y UST bp', 'BBB spread bp', 'S&P 500', 'Brent', 'VIX pts', 'CET1 after', 'Breaks requirement at'].map((h, i) =>
                <th key={h} className={clsx(th, i > 1 && 'text-right')}>{h}</th>)}
            </tr></thead>
            <tbody className="tnum">
              {lib.map((l) => {
                const c = cmp?.find((x) => x.key === l.key)
                const br = curves.data?.[l.key]?.breach_buffer
                return (
                  <tr key={l.key} className="border-b border-line last:border-0">
                    <td className={clsx(td, 'text-ink')}>{l.name.replace(/ \(.*\)$/, '')}</td>
                    <td className={clsx(td, 'text-ink-3')}>{l.name.match(/\((.*) analogue\)/)?.[1]} · {l.episode.start} → {l.episode.end}</td>
                    <td className={clsx(td, 'text-right')}>{fmtBp(l.measured.rates_bp?.USD)}</td>
                    <td className={clsx(td, 'text-right')}>{fmtBp(l.measured.credit_spread_bp?.BBB)}</td>
                    <td className={clsx(td, 'text-right')}>{fmtPc(l.measured.equity_pct?._all)}</td>
                    <td className={clsx(td, 'text-right')}>{fmtPc(l.measured.commodity_pct?.crude_oil)}</td>
                    <td className={clsx(td, 'text-right text-ink-2')}>{l.measured.vix_change === undefined ? '—' : `+${l.measured.vix_change.toFixed(0)}`}</td>
                    <td className={clsx(td, 'text-right text-ink')}>{c ? pct(c.summary.cet1_ratio_stressed, 2) : '…'}</td>
                    <td className={clsx(td, 'text-right text-ink-2')}>{curves.data ? (br ? `${Math.round(br.severity * 100)}%` : 'holds to 150%') : '…'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </section>
  )
}

function Book() {
  const { data: pf } = useQuery({ queryKey: ['portfolio'], queryFn: api.portfolio })
  if (!pf) return null
  const inst = Object.entries(pf.by_instrument).sort((a, b) => b[1] - a[1])
  const ratings = RATINGS.map((r) => pf.by_rating[r] ?? 0)
  return (
    <section className="mt-10">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-base font-semibold">The book</h2>
        <span className="text-xs text-ink-3">Synthetic wholesale bank · {pf.n_positions} positions · {money(pf.total_notional)} gross notional · rate risk hedged with pay-fixed swaps</span>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Exposure by instrument">
          <Chart height={inst.length * 28 + 20} option={{
            grid: { left: 4, right: 48, top: 0, bottom: 0, containLabel: true },
            xAxis: { type: 'value', show: false },
            yAxis: { type: 'category', inverse: true, data: inst.map(([k]) => instrumentLabel(k)), axisLine: { show: false }, axisLabel: { color: INK.secondary } },
            tooltip: { trigger: 'axis', valueFormatter: (v) => money(Number(v)) },
            series: [{ type: 'bar', ...barStyle(true), barMaxWidth: 12, data: inst.map(([, v]) => Math.round(v)), itemStyle: { color: SERIES[0], borderRadius: [0, 4, 4, 0] },
              label: { show: true, position: 'right', color: INK.muted, fontSize: 11, formatter: (p: unknown) => money((p as { value: number }).value, 0) } }],
          }} />
        </Panel>
        <Panel title="Exposure by internal grade" meta="Investment grade on the left of BB">
          <Chart height={inst.length * 28 + 20} option={{
            grid: { left: 4, right: 8, top: 8, bottom: 4, containLabel: true },
            xAxis: { type: 'category', data: RATINGS, splitLine: { show: false } },
            yAxis: { type: 'value', axisLabel: { formatter: (v: number) => money(v, 0) } },
            tooltip: { trigger: 'axis', valueFormatter: (v) => money(Number(v)) },
            series: [{ type: 'bar', ...barStyle(false), barMaxWidth: 22,
              data: ratings.map((v, i) => ({ value: Math.round(v), itemStyle: { color: i < 4 ? SERIES[0] : DEEMPH } })) }],
          }} />
        </Panel>
      </div>
    </section>
  )
}

export default function StressPage() {
  const { stressRuns: live, status } = useStream()
  const { data: runs } = useQuery({ queryKey: ['stressRuns'], queryFn: api.stressRuns, refetchInterval: status?.running ? 3000 : 20000 })
  const { data: pf } = useQuery({ queryKey: ['portfolio'], queryFn: api.portfolio })
  const [selected, setSelected] = useState<string | null>(null)
  const all = useMemo(() => runs ?? [], [runs])
  useEffect(() => { if (!selected && all.length) setSelected(all[0].run_id) }, [all, selected])
  useEffect(() => { if (live.length) setSelected(live[0].run_id) }, [live])
  const { data: run } = useQuery({ queryKey: ['stressRun', selected], queryFn: () => api.stressRun(selected!), enabled: !!selected })

  const s = run?.summary
  const trig = (run?.scenario.trigger ?? {}) as Record<string, string | number | boolean>
  const c = s ? compliance(s) : null

  return (
    <>
      <PageTitle title="Stress">Module B · event-driven stress testing of a wholesale banking book</PageTitle>
      <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Builder onRun={setSelected} />
          <Runs runs={all} selected={selected} onSelect={setSelected} />
        </div>
        <div className="min-w-0 space-y-6">
          <div className="rounded-2xl border border-line bg-panel px-6 py-6">
            {s && run && c ? (
              <>
                <div className="flex flex-wrap items-center gap-10">
                  <CapitalGauge id={run.run_id} base={s.cet1_ratio_base} stressed={s.cet1_ratio_stressed} requirement={s.cet1_requirement} minimum={s.cet1_minimum} />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium text-ink-3">{run.scenario.name}</div>
                    <div className="mt-1 text-sm text-ink-2">CET1 from {pct(s.cet1_ratio_base, 2)} to <b className="text-ink">{pct(s.cet1_ratio_stressed, 2)}</b>;
                      requirement {pct(s.cet1_requirement, 1)} (4.5% minimum plus buffers).</div>
                    <div className="mt-5 flex flex-wrap items-end gap-8">
                      <Fact label="Total loss" value={<Counter key={run.run_id} value={s.total_loss} format={(v) => money(v)} duration={1.2} />} />
                      <Fact label="Credit losses (IFRS 9)" value={<Counter key={run.run_id} value={s.ecl_increase} format={(v) => money(v)} duration={1.2} />} />
                      <Fact label="Risk-weighted assets" value={`${money(s.rwa_base, 0)} → ${money(s.rwa_stressed, 0)}`} />
                      <StatusPill level={c.level}>{c.text}</StatusPill>
                    </div>
                  </div>
                </div>
                <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-line pt-4 text-[13px]">
                  {trig.manual ? <span className="text-ink-3">Run by hand at {Math.round(run.scenario.severity * 100)}% severity.</span> : (
                    <>
                      <EventTag event={String(trig.event)} />
                      <ImpactMeter score={Number(trig.impact)} />
                      <span className="text-ink-3">{when(String(trig.published_at))} · {String(trig.n_sources)} outlets</span>
                      <a href={trig.url ? String(trig.url) : undefined} target="_blank" rel="noreferrer"
                        className="flex min-w-0 flex-1 items-center gap-1 truncate text-ink hover:underline">“{String(trig.headline)}” {trig.url && <ExternalLink size={12} className="shrink-0" />}</a>
                    </>
                  )}
                </div>
              </>
            ) : <Empty>Run a scenario, or replay a crisis stream and let the engine trigger one.</Empty>}
          </div>

          {run && (
            <>
              <div className="grid gap-6 2xl:grid-cols-2">
                <Panel title="Value bridge" meta="Each risk factor applied alone; interaction is the remainder"><Waterfall run={run} /></Panel>
                <Panel title="Breaking point" meta="CET1 as this scenario intensifies; the dot is this run"><BreakingPoint run={run} /></Panel>
              </div>
              <div className="grid gap-6 lg:grid-cols-3">
                <Panel title="Loss by asset class"><LossBars rows={run.by_asset_class ?? []} keyName="instrument" /></Panel>
                <Panel title="Loss by sector"><LossBars rows={run.by_sector ?? []} keyName="sector" /></Panel>
                <Panel title="Rating migration" meta="Loans and bonds, original grade (rows) to stressed grade"><Migration run={run} /></Panel>
              </div>
              <Panel title="Largest position losses" flush>
                <div className="overflow-x-auto">
                  <table className="w-full text-[13px]">
                    <thead><tr className="border-b border-line">
                      {['Position', 'Instrument', 'Counterparty', 'Sector · country', 'Grade', 'Notional', 'PD', 'Loss'].map((h, i) =>
                        <th key={h} className={clsx(th, i > 4 && 'text-right')}>{h}</th>)}
                    </tr></thead>
                    <tbody className="tnum">
                      {(run.top_losses ?? []).map((p) => (
                        <tr key={String(p.position_id)} className="border-b border-line last:border-0">
                          <td className={clsx(td, 'text-ink-3')}>{p.position_id}</td>
                          <td className={td}>{instrumentLabel(String(p.instrument))}</td>
                          <td className={clsx(td, 'text-ink')}>{p.counterparty}</td>
                          <td className={clsx(td, 'text-ink-3')}>{p.sector} · {p.country}</td>
                          <td className={td}>{p.rating_orig}{p.rating_orig !== p.rating_stressed && <span className="text-ink-3"> → <span className="text-ink">{p.rating_stressed}</span></span>}</td>
                          <td className={clsx(td, 'text-right text-ink-2')}>{money(Number(p.notional))}</td>
                          <td className={clsx(td, 'text-right text-ink-2')}>{pct(Number(p.pd_base), 2)} → {pct(Number(p.pd_stressed), 2)}</td>
                          <td className={clsx(td, 'text-right font-semibold text-ink')}>{money(Number(p.loss), 1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          )}
          {!run && pf && <Empty tall>{pf.n_positions} positions loaded. Run a scenario to see its effect.</Empty>}
        </div>
      </div>
      <Library />
      <Book />
    </>
  )
}
