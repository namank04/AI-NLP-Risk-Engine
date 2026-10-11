import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { Pause, Play } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { api } from '../api/client'
import { useStream } from '../hooks/useStream'
import { when } from '../lib/format'
import { Tape, StressAlerts } from './Live'
import { EASE, STATIC } from './motion'
import { Segmented, Select } from './ui'

const NAV = [
  { to: '/', label: 'Overview' },
  { to: '/desk', label: 'Desk' },
  { to: '/analyze', label: 'Analyze' },
  { to: '/index', label: 'Index' },
  { to: '/stress', label: 'Stress' },
  { to: '/evaluation', label: 'Evaluation' },
]

export const REPLAY_NAMES: Record<string, string> = {
  recent: 'Last 48 hours',
  svb_2023: 'SVB collapse · Mar 2023',
  ukraine_2022: 'Russia–Ukraine · Feb 2022',
  covid_2020: 'COVID crash · Mar 2020',
  smoke_covid_headlines: 'COVID headlines · 2020',
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
      <rect x="2" y="13" width="4" height="9" rx="1.2" fill="#3987e5" opacity=".55" />
      <rect x="10" y="8" width="4" height="14" rx="1.2" fill="#3987e5" opacity=".8" />
      <rect x="18" y="3" width="4" height="19" rx="1.2" fill="#3987e5" />
    </svg>
  )
}

function Transport() {
  const { status, clear } = useStream()
  const { data: replays } = useQuery({ queryKey: ['replays'], queryFn: api.replays })
  const [name, setName] = useState('')
  const [speed, setSpeed] = useState(21600)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!name && replays?.length) setName(replays.includes('svb_2023') ? 'svb_2023' : replays[0])
  }, [replays, name])
  const running = !!status?.running

  const toggle = async () => {
    setBusy(true)
    try {
      if (running) await api.stopReplay()
      else { clear(); await api.startReplay({ name, speed, reset: true }) }
    } finally { setBusy(false) }
  }

  return (
    <div className="flex items-center gap-2">
      <Select label="Replay stream" value={name} onChange={setName} className="max-w-[210px]">
        {(replays ?? []).map((r) => <option key={r} value={r}>{REPLAY_NAMES[r] ?? r}</option>)}
      </Select>
      <Segmented size="sm" value={speed} onChange={setSpeed}
        options={[{ value: 3600, label: '1h/s' }, { value: 21600, label: '6h/s' }, { value: 86400, label: '1d/s' }]} />
      <button onClick={toggle} disabled={busy || !name} aria-label={running ? 'Stop replay' : 'Start replay'}
        className={clsx('grid h-8 w-8 place-items-center rounded-full transition-colors disabled:opacity-40',
          running ? 'bg-raised text-ink hover:bg-hover' : 'bg-accent text-white hover:bg-[#4a94ea]')}>
        {running ? <Pause size={15} /> : <Play size={15} className="translate-x-px" />}
      </button>
      <div className="hidden w-[132px] text-right lg:block">
        <div className="tnum text-xs text-ink-2">{status?.sim_time ? when(status.sim_time) : busy ? 'loading models…' : 'idle'}</div>
        <div className="tnum text-[11px] text-ink-3">{status?.total ? `${status.processed.toLocaleString()} / ${status.total.toLocaleString()}` : ''}</div>
      </div>
    </div>
  )
}

export function Layout({ children }: { children: ReactNode }) {
  const { status, connected } = useStream()
  const { pathname } = useLocation()
  const progress = status?.total ? (100 * status.processed) / status.total : 0
  return (
    <div className="min-h-full">
      <div className="backdrop" aria-hidden><i /><i /></div>
      <header className="sticky top-0 z-30 border-b border-line bg-page/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-8 px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <Logo />
            <span className="text-[15px] font-semibold tracking-tight">Signal&nbsp;Risk</span>
          </Link>
          <nav className="flex h-full items-center gap-1">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.to === '/'}
                className={({ isActive }) => clsx('relative flex h-full items-center px-3 text-[13px] font-medium transition-colors',
                  isActive ? 'text-ink' : 'text-ink-3 hover:text-ink-2')}>
                {({ isActive }) => (
                  <>
                    {n.label}
                    {isActive && <motion.span layoutId="nav-underline" className="absolute inset-x-3 bottom-0 h-0.5 rounded-full bg-accent"
                      transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-xs text-ink-3" title={connected ? 'Live stream connected' : 'Stream disconnected'}>
              <span className={clsx('h-1.5 w-1.5 rounded-full', connected ? 'bg-good' : 'bg-critical', connected && status?.running && 'live-dot')} />
              {connected ? 'Live' : 'Offline'}
            </span>
            <Transport />
          </div>
        </div>
        <div className="h-px bg-line">
          <div className="h-px bg-accent transition-[width] duration-500" style={{ width: `${progress}%`, opacity: status?.running ? 1 : 0.4 }} />
        </div>
      </header>
      <Tape />
      <motion.main key={pathname} initial={STATIC ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: EASE }} className="mx-auto max-w-[1440px] px-6 pt-8 pb-16">{children}</motion.main>
      <StressAlerts />
    </div>
  )
}

export function PageTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-baseline justify-between gap-3">
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      {children && <div className="text-sm text-ink-3">{children}</div>}
    </div>
  )
}
