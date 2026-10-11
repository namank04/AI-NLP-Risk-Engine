import { animate, motion, useInView } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'

export const STATIC = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('snapshot')
export const EASE = [0.16, 1, 0.3, 1] as const

/** Number that glides to each new value instead of jumping. */
export function Counter({ value, format, duration = 0.9, start = 0 }: {
  value: number
  format: (n: number) => string
  duration?: number
  start?: number
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true })
  const cur = useRef(STATIC ? value : start)
  const [shown, setShown] = useState(cur.current)
  useEffect(() => {
    if (STATIC || !inView) return
    const c = animate(cur.current, value, { duration, ease: EASE, onUpdate: (v) => { cur.current = v; setShown(v) } })
    return () => c.stop()
  }, [value, duration, inView])
  return <span ref={ref} className="tnum">{format(STATIC ? value : shown)}</span>
}

export function Reveal({ children, delay = 0, className, y = 14 }: { children: ReactNode; delay?: number; className?: string; y?: number }) {
  return (
    <motion.div className={className} initial={STATIC ? false : { opacity: 0, y }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }} transition={{ duration: 0.6, delay, ease: EASE }}>
      {children}
    </motion.div>
  )
}

/** Horizontal bar that grows from its origin to a fraction (0..1). */
export function Grow({ to, color, className, delay = 0 }: { to: number; color: string; className?: string; delay?: number }) {
  return (
    <motion.span className={className} style={{ background: color, transformOrigin: 'left', display: 'block' }}
      initial={STATIC ? false : { scaleX: 0 }} animate={{ scaleX: Math.max(0, Math.min(1, to)) }}
      transition={{ duration: 0.8, delay, ease: EASE }} />
  )
}
