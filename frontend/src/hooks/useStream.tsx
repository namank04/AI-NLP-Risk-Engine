import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import type { IndexSnapshot, ReplayStatus, Signal, StressRun } from '../api/types'

interface StreamState {
  connected: boolean
  signals: Signal[] // newest first, capped
  index: IndexSnapshot | null
  stressRuns: StressRun[] // newest first
  status: ReplayStatus | null
  counter: number // total signals received this session
  clear: () => void
}

const Ctx = createContext<StreamState | null>(null)
const MAX_SIGNALS = 400

export function StreamProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false)
  const [signals, setSignals] = useState<Signal[]>([])
  const [index, setIndex] = useState<IndexSnapshot | null>(null)
  const [stressRuns, setStressRuns] = useState<StressRun[]>([])
  const [status, setStatus] = useState<ReplayStatus | null>(null)
  const [counter, setCounter] = useState(0)
  const buffer = useRef<Signal[]>([])

  useEffect(() => {
    let es: EventSource | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    // batch signal updates so a fast replay does not re-render per message
    const flush = setInterval(() => {
      if (buffer.current.length === 0) return
      const batch = buffer.current.reverse()
      buffer.current = []
      setCounter((c) => c + batch.length)
      setSignals((prev) => [...batch, ...prev].slice(0, MAX_SIGNALS))
    }, 400)

    // ?snapshot renders without the live connection (static captures for docs/slides)
    if (new URLSearchParams(window.location.search).has('snapshot')) return () => clearInterval(flush)
    const connect = () => {
      es = new EventSource('/api/stream')
      es.onopen = () => setConnected(true)
      es.onerror = () => {
        setConnected(false)
        es?.close()
        retry = setTimeout(connect, 2000)
      }
      es.addEventListener('signal', (e) => buffer.current.push(JSON.parse((e as MessageEvent).data)))
      es.addEventListener('index', (e) => setIndex(JSON.parse((e as MessageEvent).data)))
      es.addEventListener('stress', (e) => {
        const run = JSON.parse((e as MessageEvent).data) as StressRun
        setStressRuns((prev) => [run, ...prev].slice(0, 50))
      })
      es.addEventListener('status', (e) => setStatus(JSON.parse((e as MessageEvent).data)))
    }
    connect()
    return () => {
      clearInterval(flush)
      if (retry) clearTimeout(retry)
      es?.close()
    }
  }, [])

  const clear = () => {
    setSignals([])
    setStressRuns([])
    setIndex(null)
    setCounter(0)
  }

  return (
    <Ctx.Provider value={{ connected, signals, index, stressRuns, status, counter, clear }}>{children}</Ctx.Provider>
  )
}

export function useStream() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useStream outside StreamProvider')
  return v
}
