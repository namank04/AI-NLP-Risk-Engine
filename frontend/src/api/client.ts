import type {
  CompareRow,
  IndexSnapshot,
  ScenarioLibraryEntry,
  SeverityCurve,
  PortfolioOverview,
  ReplayStatus,
  Signal,
  StressRun,
  TaxonomyClass,
  UniverseMember,
} from './types'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  return res.json() as Promise<T>
}

const qs = (p: Record<string, string | number | undefined | null>) => {
  const s = new URLSearchParams()
  Object.entries(p).forEach(([k, v]) => v !== undefined && v !== null && v !== '' && s.set(k, String(v)))
  const out = s.toString()
  return out ? `?${out}` : ''
}

export const api = {
  health: () => req<{ status: string; models_loaded: boolean; signals: number; raw_items: number; replay: ReplayStatus }>('/api/health'),
  universe: () => req<UniverseMember[]>('/api/universe'),
  taxonomy: () => req<Record<string, TaxonomyClass>>('/api/taxonomy'),
  signals: (p: { entity?: string; event?: string; min_impact?: number; limit?: number } = {}) =>
    req<Signal[]>(`/api/signals${qs(p)}`),
  analyze: (body: { text: string; title?: string; source_type?: string; store?: boolean }) =>
    req<Signal[]>('/api/analyze', { method: 'POST', body: JSON.stringify(body) }),
  entitySummary: (days = 7) =>
    req<{ entity: string; n: number; avg_sentiment: number; weighted_sentiment: number; max_impact: number; avg_impact: number; last_signal: string }[]>(
      `/api/entities/summary${qs({ days })}`),
  entityTimeseries: (entity: string, freq = 'day') =>
    req<{ t: string; n: number; sentiment: number; max_impact: number; top_event: string }[]>(
      `/api/entities/${encodeURIComponent(entity)}/timeseries${qs({ freq })}`),
  eventDistribution: () =>
    req<{ event: string; n: number; avg_impact: number; avg_sentiment: number }[]>('/api/events/distribution'),
  indexCurrent: () => req<IndexSnapshot>('/api/index/current'),
  indexHistory: (limit = 2000) => req<IndexSnapshot[]>(`/api/index/history${qs({ limit })}`),
  portfolio: () => req<PortfolioOverview>('/api/stress/portfolio'),
  scenarios: () =>
    req<{ trigger: Record<string, unknown>; systemic: Record<string, { name: string; episode: { start: string; end: string } }>; counterparties: string[] }>(
      '/api/stress/scenarios'),
  stressRuns: () => req<StressRun[]>('/api/stress/runs'),
  stressRun: (id: string) => req<StressRun>(`/api/stress/runs/${id}`),
  runStress: (body: { kind: 'systemic' | 'idiosyncratic'; event_class?: string; severity: number; scope_countries?: string[]; counterparty?: string; subtype?: string }) =>
    req<StressRun>('/api/stress/run', { method: 'POST', body: JSON.stringify(body) }),
  stressCurve: (eventClass: string, scope: string[] = []) =>
    req<SeverityCurve>(`/api/stress/curve${qs({ event_class: eventClass, scope: scope.join(',') })}`),
  stressCompare: (severity: number) => req<CompareRow[]>(`/api/stress/compare${qs({ severity })}`),
  stressLibrary: () => req<ScenarioLibraryEntry[]>('/api/stress/library'),
  goldStats: () => req<Record<string, unknown>>('/api/gold/stats'),
  replays: () => req<string[]>('/api/replay/list'),
  startReplay: (body: { name: string; speed: number; reset?: boolean }) =>
    req<ReplayStatus>('/api/replay/start', { method: 'POST', body: JSON.stringify(body) }),
  stopReplay: () => req<ReplayStatus>('/api/replay/stop', { method: 'POST' }),
}
