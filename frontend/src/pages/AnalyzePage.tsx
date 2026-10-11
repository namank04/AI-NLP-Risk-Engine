import { useMutation } from '@tanstack/react-query'
import { Braces, Sparkles } from 'lucide-react'
import { motion } from 'motion/react'
import { useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { Signal } from '../api/types'
import { PageTitle } from '../components/Layout'
import { Counter, EASE, Grow } from '../components/motion'
import { Button, Empty, EventTag, ImpactMeter, Panel, Segmented, ToneBar } from '../components/ui'
import { eventLabel, signed, titleCase } from '../lib/format'
import { divergingColor, SERIES } from '../lib/theme'

const EXAMPLES: { label: string; text: string }[] = [
  { label: 'Two companies, one sentence', text: 'Apple beats quarterly estimates on record iPhone demand, while Tesla misses badly as margins collapse.' },
  { label: 'Credit downgrade', text: "Boeing downgraded to junk by Moody's as 737 MAX crisis deepens and cash burn accelerates." },
  { label: 'Sanctions and oil', text: 'U.S. imposes sweeping sanctions on Russian banks after the invasion of Ukraine; Brent crude surges above $100.' },
  { label: 'Bank run', text: 'Silicon Valley Bank shares plunge as depositors pull funds; regulators weigh options amid a liquidity crisis.' },
  { label: 'Central bank', text: 'Federal Reserve raises interest rates by 75 basis points and signals more hikes to fight inflation.' },
  { label: 'Restructuring', text: 'Exxon Mobil cuts 2,000 jobs as it restructures amid weak oil demand.' },
  { label: 'Ambiguous name', text: 'The Dow jumps 400 points while Dow Inc shares slide after the chemical maker cut its outlook.' },
]

/** The input with each entity's scored clause underlined in its sentiment colour. */
function Highlighted({ text, signals }: { text: string; signals: Signal[] }) {
  const spans = signals.flatMap((s) => {
    const span = s.evidence.span?.trim()
    const at = span ? text.indexOf(span) : -1
    return at >= 0 ? [{ at, end: at + span!.length, s }] : []
  }).sort((a, b) => a.at - b.at).filter((x, i, arr) => i === 0 || x.at >= arr[i - 1].end)
  if (!spans.length) return null
  const parts: ReactNode[] = []
  let pos = 0
  spans.forEach(({ at, end, s }, i) => {
    if (at > pos) parts.push(text.slice(pos, at))
    const c = divergingColor(s.sentiment.score)
    parts.push(
      <motion.mark key={i} title={`${s.entity}: ${signed(s.sentiment.score)}`} initial={{ backgroundSize: '0% 100%' }} animate={{ backgroundSize: '100% 100%' }}
        transition={{ duration: 0.8, delay: 0.2 + i * 0.25, ease: EASE }}
        style={{ color: 'inherit', backgroundImage: `linear-gradient(${c}33, ${c}33)`, backgroundRepeat: 'no-repeat', borderBottom: `2px solid ${c}`, padding: '1px 0' }}>
        {text.slice(at, end)}<sup className="ml-0.5 text-[10px] font-semibold" style={{ color: c }}>{s.entity === 'MARKET' ? 'MKT' : s.entity}</sup>
      </motion.mark>)
    pos = end
  })
  if (pos < text.length) parts.push(text.slice(pos))
  return <p className="mb-5 rounded-xl border border-line bg-page/50 px-4 py-3 text-[14px] leading-loose text-ink-2">{parts}</p>
}

function Result({ s }: { s: Signal }) {
  const [json, setJson] = useState(false)
  const probs = Object.entries(s.event.probs).filter(([k]) => k !== 'none').sort((a, b) => b[1] - a[1]).slice(0, 6)
  const scope = [...s.scope.countries, ...s.scope.sectors, ...s.scope.commodities, ...s.scope.currencies]
  return (
    <article className="rounded-xl border border-line bg-page/50">
      <header className="flex flex-wrap items-center gap-3 border-b border-line px-5 py-3">
        <span className="text-base font-semibold">{s.entity === 'MARKET' ? 'Market-wide' : s.entity}</span>
        <EventTag event={s.event.primary} />
        {s.event.subtype && <span className="text-xs text-ink-3">{titleCase(s.event.subtype)}</span>}
        <button onClick={() => setJson(!json)} className="ml-auto flex items-center gap-1 text-xs text-ink-3 hover:text-ink-2" aria-pressed={json}>
          <Braces size={13} /> {json ? 'Hide' : 'JSON'}
        </button>
      </header>
      {json ? (
        <pre className="max-h-80 overflow-auto px-5 py-4 text-[11px] leading-relaxed text-ink-2">{JSON.stringify(s, null, 2)}</pre>
      ) : (
        <div className="grid gap-6 px-5 py-4 md:grid-cols-[220px_1fr]">
          <div className="space-y-4">
            <div>
              <div className="text-xs text-ink-3">Sentiment</div>
              <div className="mt-1 flex items-center gap-3">
                <span className="text-3xl font-semibold" style={{ color: divergingColor(s.sentiment.score) }}>
                  <Counter value={s.sentiment.score} format={(v) => signed(v)} /></span>
              </div>
              <div className="mt-2"><ToneBar value={s.sentiment.score} width={150} /></div>
            </div>
            <div>
              <div className="text-xs text-ink-3">Impact</div>
              <div className="mt-1"><ImpactMeter score={s.impact.score} /></div>
            </div>
          </div>
          <div className="space-y-4">
            <blockquote className="border-l-2 border-accent/60 pl-3 text-[13px] leading-relaxed text-ink-2">
              {s.evidence.span ?? s.evidence.headline}
            </blockquote>
            <div className="space-y-1.5">
              {probs.map(([k, v], i) => (
                <div key={k} className="grid grid-cols-[130px_1fr_36px] items-center gap-2 text-xs">
                  <span className={i === 0 ? 'text-ink' : 'text-ink-3'}>{eventLabel(k)}</span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-raised">
                    <Grow to={v} delay={0.1 + i * 0.05} color={i === 0 ? SERIES[0] : '#3a4250'} className="h-1.5 w-full rounded-full" />
                  </span>
                  <span className="tnum text-right text-ink-3">{v.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <dl className="grid grid-cols-2 gap-3 text-xs">
              <div><dt className="text-ink-3">Transmission channels</dt><dd className="mt-0.5 text-ink-2">{s.channels.map(titleCase).join(', ') || '—'}</dd></div>
              <div><dt className="text-ink-3">Scope</dt><dd className="mt-0.5 text-ink-2">{scope.map(titleCase).join(', ') || '—'}</dd></div>
            </dl>
          </div>
        </div>
      )}
    </article>
  )
}

export default function AnalyzePage() {
  const [text, setText] = useState(EXAMPLES[0].text)
  const [sourceType, setSourceType] = useState<'news' | 'social' | 'filing'>('news')
  const m = useMutation({ mutationFn: () => api.analyze({ text, source_type: sourceType }) })
  const [submitted, setSubmitted] = useState('')
  return (
    <>
      <PageTitle title="Analyze">Score any text with the same engine the stream uses.</PageTitle>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel title="Text">
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={9} aria-label="Text to analyze"
            className="w-full resize-y rounded-xl border border-line bg-page p-4 text-[14px] leading-relaxed text-ink outline-none focus:border-accent/60" />
          <div className="mt-3 flex items-center justify-between gap-3">
            <Segmented size="sm" value={sourceType} onChange={setSourceType}
              options={[{ value: 'news', label: 'News' }, { value: 'social', label: 'Social post' }, { value: 'filing', label: 'Filing' }]} />
            <Button onClick={() => { setSubmitted(text); m.mutate() }} disabled={m.isPending || text.trim().length < 3}>
              <Sparkles size={14} /> {m.isPending ? 'Scoring…' : 'Analyze'}
            </Button>
          </div>
          <div className="mt-6">
            <div className="mb-2 text-xs text-ink-3">Try a case</div>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((e) => (
                <button key={e.label} onClick={() => setText(e.text)}
                  className="rounded-full border border-line px-3 py-1 text-xs text-ink-2 transition-colors hover:border-line-strong hover:text-ink">
                  {e.label}
                </button>
              ))}
            </div>
          </div>
        </Panel>
        <Panel title="Structured output" meta={m.data ? `${m.data.length} signal${m.data.length === 1 ? '' : 's'}, one per linked entity` : undefined}>
          {m.isError && <div className="rounded-lg border border-critical/40 px-3 py-2 text-sm text-critical">{String(m.error)}</div>}
          {!m.data && !m.isError && <Empty tall>The engine links entities, scores sentiment per entity, classifies the event and estimates impact.</Empty>}
          {m.data?.length === 0 && <Empty tall>No universe company and no market-wide economic topic in this text.</Empty>}
          {m.data && m.data.length > 0 && <Highlighted text={submitted} signals={m.data} />}
          <div className="space-y-4">{m.data?.map((s, i) => (
            <motion.div key={s.signal_id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: i * 0.12, ease: EASE }}>
              <Result s={s} />
            </motion.div>))}</div>
        </Panel>
      </div>
    </>
  )
}
