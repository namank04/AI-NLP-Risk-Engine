import { motion } from 'motion/react'
import { STATUS } from '../lib/theme'
import { Counter, EASE, STATIC } from './motion'

const MAX = 0.2
const CX = 120, CY = 118, R = 96

const at = (v: number, r = R) => {
  const a = Math.PI * Math.min(1, Math.max(0, v / MAX))
  return [CX - r * Math.cos(a), CY - r * Math.sin(a)] as const
}
const arc = (a: number, b: number) => {
  const [x0, y0] = at(a), [x1, y1] = at(b)
  return `M ${x0} ${y0} A ${R} ${R} 0 0 1 ${x1} ${y1}`
}

/** Semicircular CET1 gauge: zones for minimum and buffers, ghost arc for the starting ratio. */
export function CapitalGauge({ base, stressed, requirement, minimum, id }: {
  base: number; stressed: number; requirement: number; minimum: number; id: string
}) {
  const color = stressed < minimum ? STATUS.critical : stressed < requirement ? STATUS.warning : '#3987e5'
  const tick = (v: number, label: string) => {
    const [x0, y0] = at(v, R - 13), [x1, y1] = at(v, R + 9), [lx, ly] = at(v, R + 22)
    return (
      <g key={label}>
        <line x1={x0} y1={y0} x2={x1} y2={y1} stroke="#8a93a1" strokeWidth={1.2} />
        <text x={lx} y={ly} textAnchor="middle" fontSize={10} fill="#8a93a1">{label}</text>
      </g>
    )
  }
  return (
    <div className="relative w-[240px] shrink-0">
      <svg viewBox="0 0 240 132" className="w-full overflow-visible" aria-hidden>
        <path d={arc(0, MAX)} stroke="#1c222c" strokeWidth={14} fill="none" strokeLinecap="round" />
        <path d={arc(0, minimum)} stroke={STATUS.critical} strokeOpacity={0.22} strokeWidth={14} fill="none" />
        <path d={arc(minimum, requirement)} stroke={STATUS.warning} strokeOpacity={0.18} strokeWidth={14} fill="none" />
        <path d={arc(0, base)} stroke="#4a5262" strokeWidth={4} fill="none" strokeLinecap="round" />
        <motion.path key={id} d={arc(0, MAX)} stroke={color} strokeWidth={8} fill="none" strokeLinecap="round"
          initial={STATIC ? false : { pathLength: Math.min(1, base / MAX) }}
          animate={{ pathLength: Math.max(0.002, Math.min(1, stressed / MAX)) }} transition={{ duration: 1.4, ease: EASE }}
          style={{ filter: `drop-shadow(0 0 6px ${color}66)` }} />
        {tick(minimum, `${(100 * minimum).toFixed(1)}%`)}
        {tick(requirement, `${(100 * requirement).toFixed(1)}%`)}
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className="text-4xl font-semibold tracking-tight" style={{ color }}>
          <Counter key={id} value={100 * stressed} start={100 * base} duration={1.4} format={(v) => `${v.toFixed(2)}%`} />
        </div>
        <div className="text-[11px] text-ink-3">CET1 after stress</div>
      </div>
    </div>
  )
}
