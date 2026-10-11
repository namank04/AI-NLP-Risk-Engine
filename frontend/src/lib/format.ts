export const EVENT_LABEL: Record<string, string> = {
  geopolitical: 'Geopolitical',
  macroeconomic: 'Macroeconomic',
  credit_event: 'Credit event',
  merger_acquisition: 'M&A',
  product_launch: 'Product',
  earnings: 'Earnings',
  regulatory_legal: 'Regulatory & legal',
  management_change: 'Management',
  operational: 'Operational',
  none: 'No event',
}

export const eventLabel = (k: string) => EVENT_LABEL[k] ?? k
export const titleCase = (s: string) => s.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())

export const signed = (x: number, d = 2) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(d)}`
export const pct = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${(100 * x).toFixed(d)}%`
export const signedPct = (x: number, d = 1) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(100 * x).toFixed(d)}%`
export const pp = (x: number, d = 1) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(100 * x).toFixed(d)} pp`

/** USD millions -> compact money ("$1.2bn", "$640m"). */
export const money = (m: number, d = 1) => {
  const a = Math.abs(m)
  const s = m < 0 ? '−' : ''
  return a >= 1000 ? `${s}$${(a / 1000).toFixed(d)}bn` : `${s}$${a.toFixed(a >= 100 ? 0 : d)}m`
}

export const compact = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(0)}K` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : `${n}`

export const when = (iso: string | null | undefined, withTime = true) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString(undefined, withTime
    ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
    : { year: 'numeric', month: 'short', day: 'numeric' })
}

export const day = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })

const INSTRUMENT: Record<string, string> = {
  irs: 'Interest-rate swap', fx_forward: 'FX forward', cds: 'CDS', corp_bond: 'Corporate bond',
  sovereign_bond: 'Sovereign bond', term_loan: 'Term loan', revolver: 'Revolving credit', equity: 'Equity',
}
export const instrumentLabel = (k: string) => INSTRUMENT[k] ?? titleCase(k)
