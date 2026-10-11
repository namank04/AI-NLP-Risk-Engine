import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, OctagonX } from 'lucide-react'
import { motion } from 'motion/react'
import type { MouseEvent, ReactNode } from 'react'
import { eventLabel, signed } from '../lib/format'
import { divergingColor } from '../lib/theme'
import { EASE, STATIC } from './motion'

const track = (e: MouseEvent<HTMLElement>) => {
  const r = e.currentTarget.getBoundingClientRect()
  e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
  e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
}

export function Panel({ title, meta, actions, children, className, flush }: {
  title?: ReactNode
  meta?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  flush?: boolean
}) {
  return (
    <motion.section onMouseMove={track} className={clsx('spot rounded-2xl border border-line bg-panel transition-colors hover:border-line-strong', className)}
      initial={STATIC ? false : { opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: '-30px' }}
      transition={{ duration: 0.55, ease: EASE }}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-4 px-5 pt-4 pb-3">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold text-ink">{title}</h2>}
            {meta && <p className="mt-0.5 text-xs text-ink-3">{meta}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={flush ? '' : 'px-5 pb-5'}>{children}</div>
    </motion.section>
  )
}

/** The single headline figure of a view. */
export function Hero({ label, value, detail, aside }: { label: string; value: ReactNode; detail?: ReactNode; aside?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-6">
      <div>
        <div className="text-xs font-medium text-ink-3">{label}</div>
        <div className="mt-1 text-5xl font-semibold tracking-tight text-ink">{value}</div>
        {detail && <div className="mt-2 max-w-xl text-sm text-ink-2">{detail}</div>}
      </div>
      {aside}
    </div>
  )
}

export function Fact({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <div className="text-xs text-ink-3">{label}</div>
      <div className="mt-0.5 text-lg font-semibold text-ink">{value}</div>
    </div>
  )
}

const STATUS_STYLE = {
  good: { icon: CheckCircle2, cls: 'text-good bg-good/12 border-good/30' },
  warning: { icon: AlertTriangle, cls: 'text-warning bg-warning/10 border-warning/30' },
  critical: { icon: OctagonX, cls: 'text-critical bg-critical/12 border-critical/35' },
} as const

export function StatusPill({ level, children }: { level: keyof typeof STATUS_STYLE; children: ReactNode }) {
  const { icon: Icon, cls } = STATUS_STYLE[level]
  return (
    <span className={clsx('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium', cls)}>
      <Icon size={14} strokeWidth={2.2} aria-hidden />
      {children}
    </span>
  )
}

export function Segmented<T extends string | number>({ options, value, onChange, size = 'md' }: {
  options: { value: T; label: ReactNode }[]
  value: T
  onChange: (v: T) => void
  size?: 'sm' | 'md'
}) {
  return (
    <div role="radiogroup" className="inline-flex rounded-lg border border-line bg-page p-0.5">
      {options.map((o) => (
        <button key={String(o.value)} role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}
          className={clsx('rounded-md font-medium transition-colors', size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-[13px]',
            o.value === value ? 'bg-raised text-ink shadow-sm' : 'text-ink-3 hover:text-ink-2')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Select({ value, onChange, children, label, className }: {
  value: string | number
  onChange: (v: string) => void
  children: ReactNode
  label?: string
  className?: string
}) {
  return (
    <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}
      className={clsx('rounded-lg border border-line bg-page px-2.5 py-1.5 text-[13px] text-ink outline-none hover:border-line-strong', className)}>
      {children}
    </select>
  )
}

export function Button({ children, onClick, tone = 'primary', disabled, className, type = 'button' }: {
  children: ReactNode
  onClick?: () => void
  tone?: 'primary' | 'quiet'
  disabled?: boolean
  className?: string
  type?: 'button' | 'submit'
}) {
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={clsx('inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        tone === 'primary' ? 'bg-accent text-white hover:bg-[#4a94ea]' : 'border border-line text-ink-2 hover:border-line-strong hover:text-ink',
        className)}>
      {children}
    </button>
  )
}

export function EventTag({ event }: { event: string }) {
  return (
    <span className={clsx('inline-flex rounded-md border px-1.5 py-px text-[11px] whitespace-nowrap',
      event === 'none' ? 'border-line text-ink-3' : 'border-line-strong text-ink-2')}>
      {eventLabel(event)}
    </span>
  )
}

/** Signed sentiment: number in ink + a diverging bar that carries the colour. */
export function ToneBar({ value, width = 56 }: { value: number; width?: number }) {
  const half = width / 2
  const len = Math.min(1, Math.abs(value)) * half
  return (
    <span className="inline-flex items-center gap-2">
      <svg width={width} height={8} aria-hidden>
        <rect x={0} y={3} width={width} height={2} rx={1} fill="#262c37" />
        <rect x={value >= 0 ? half : half - len} y={1} width={Math.max(len, 1.5)} height={6} rx={2}
          fill={divergingColor(value)} />
        <rect x={half - 0.5} y={0} width={1} height={8} fill="#4a5262" />
      </svg>
      <span className="tnum w-11 text-right text-xs text-ink-2">{signed(value)}</span>
    </span>
  )
}

/** Impact 1-10 on a single-hue scale (status colours are reserved for compliance). */
export function ImpactMeter({ score }: { score: number }) {
  return (
    <span className="inline-flex items-center gap-2" title={`Impact ${score} of 10`}>
      <span className="flex gap-[2px]" aria-hidden>
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className="h-2.5 w-[4px] rounded-[1px] transition-colors duration-500"
            style={{ background: i < score ? (score >= 7 ? '#6da7ec' : '#3d6aa8') : '#232934', transitionDelay: `${i * 40}ms` }} />
        ))}
      </span>
      <span className="tnum w-4 text-xs text-ink-2">{score}</span>
    </span>
  )
}

export function SourceTag({ type }: { type: string }) {
  const label = type === 'filing' ? 'SEC filing' : type === 'social' ? 'Social' : 'News'
  return <span className="text-[11px] text-ink-3">{label}</span>
}

export function Empty({ children, tall }: { children: ReactNode; tall?: boolean }) {
  return <div className={clsx('flex items-center justify-center px-6 text-center text-sm text-ink-3', tall ? 'h-72' : 'h-36')}>{children}</div>
}

export const th = 'px-3 py-2 text-left text-[11px] font-medium text-ink-3 first:pl-5 last:pr-5'
export const td = 'px-3 py-2 first:pl-5 last:pr-5'
