import { useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Siren, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Signal, StressRun } from '../api/types'
import { useStream } from '../hooks/useStream'
import { pct, signed } from '../lib/format'
import { divergingColor } from '../lib/theme'
import { Counter, EASE, STATIC } from './motion'

function TapeItem({ s }: { s: Signal }) {
  return (
    <span className="flex shrink-0 items-center gap-2 px-5 text-xs">
      <span className="font-semibold text-ink-2">{s.entity === 'MARKET' ? 'MKT' : s.entity}</span>
      <span className="tnum" style={{ color: divergingColor(s.sentiment.score) }}>{signed(s.sentiment.score)}</span>
      <span className="max-w-[440px] truncate text-ink-3">{s.evidence.headline}</span>
      {s.impact.score >= 7 && <span className="rounded bg-accent-soft px-1.5 text-[10px] font-semibold text-[#8fbdf2]">IMPACT {s.impact.score}</span>}
    </span>
  )
}

/** Scrolling tape of the latest signals, one per story. */
export function Tape() {
  const { signals: live } = useStream()
  const { data: stored } = useQuery({ queryKey: ['signals'], queryFn: () => api.signals({ limit: 400 }), refetchInterval: 20000 })
  const items = useMemo(() => {
    const seen = new Set<string>()
    return (live.length ? live : stored ?? []).filter((s) => !seen.has(s.cluster_id) && seen.add(s.cluster_id)).slice(0, 28)
  }, [live, stored])
  if (items.length < 4) return null
  return (
    <div className="tape overflow-hidden border-b border-line bg-panel/60 py-1.5" aria-label="Latest signals">
      <div className="flex w-max" style={{ ['--tape-s' as string]: `${items.length * 5}s` }}>
        {[0, 1].map((k) => <div key={k} className="flex" aria-hidden={k === 1}>{items.map((s) => <TapeItem key={s.signal_id} s={s} />)}</div>)}
      </div>
    </div>
  )
}

function Toast({ run, dismiss }: { run: StressRun; dismiss: (id: string) => void }) {
  const onClose = () => dismiss(run.run_id)
  const t = run.scenario.trigger as Record<string, string | number>
  const s = run.summary
  const level = s.breaches_minimum ? 'text-critical' : s.breaches_buffer ? 'text-warning' : 'text-good'
  useEffect(() => { const id = setTimeout(() => dismiss(run.run_id), 12000); return () => clearTimeout(id) }, [dismiss, run.run_id])
  return (
    <motion.div layout initial={{ opacity: 0, x: 60, scale: 0.96 }} animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 60, transition: { duration: 0.25 } }} transition={{ duration: 0.5, ease: EASE }}
      className="pointer-events-auto w-[360px] overflow-hidden rounded-2xl border border-critical/40 bg-raised/95 shadow-2xl shadow-black/50 backdrop-blur">
      <div className="flex items-start gap-3 px-4 pt-3.5">
        <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-critical/15 text-critical"><Siren size={15} /></span>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-semibold tracking-wide text-critical uppercase">Stress test triggered</div>
          <div className="mt-0.5 text-[13px] font-medium text-ink">{run.scenario.name.replace(/ \(.*\)$/, '')}</div>
          {t.headline && <div className="mt-1 line-clamp-2 text-xs text-ink-3">“{t.headline}”</div>}
        </div>
        <button onClick={onClose} aria-label="Dismiss" className="text-ink-3 hover:text-ink"><X size={14} /></button>
      </div>
      <div className="mt-3 flex items-end justify-between border-t border-line px-4 py-3">
        <div>
          <div className="text-[11px] text-ink-3">CET1 ratio</div>
          <div className="text-lg font-semibold">
            <span className="text-ink-3">{pct(s.cet1_ratio_base, 1)} → </span>
            <span className={level}><Counter value={100 * s.cet1_ratio_stressed} start={100 * s.cet1_ratio_base} format={(v) => `${v.toFixed(1)}%`} duration={1.6} /></span>
          </div>
        </div>
        <Link to="/stress" onClick={onClose} className="flex items-center gap-1 text-xs font-medium text-accent hover:underline">Open <ArrowRight size={12} /></Link>
      </div>
      <motion.div className="h-0.5 origin-left bg-critical/60" initial={{ scaleX: 1 }} animate={{ scaleX: 0 }} transition={{ duration: 12, ease: 'linear' }} />
    </motion.div>
  )
}

/** Toasts and a screen-edge flash whenever the engine fires a stress test. */
export function StressAlerts() {
  const { stressRuns } = useStream()
  const [shown, setShown] = useState<StressRun[]>([])
  const [flash, setFlash] = useState(0)
  const last = useRef<string | null>(null)
  const dismiss = useCallback((id: string) => setShown((s) => s.filter((x) => x.run_id !== id)), [])
  useEffect(() => {
    const top = stressRuns[0]
    if (!top || top.run_id === last.current || STATIC) return
    last.current = top.run_id
    setShown((s) => [top, ...s].slice(0, 3))
    setFlash((f) => f + 1)
  }, [stressRuns])
  return (
    <>
      {flash > 0 && <div key={flash} className="alarm" />}
      <div className="pointer-events-none fixed right-5 bottom-5 z-50 flex flex-col gap-3">
        <AnimatePresence initial={false}>
          {shown.map((r) => <Toast key={r.run_id} run={r} dismiss={dismiss} />)}
        </AnimatePresence>
      </div>
    </>
  )
}
