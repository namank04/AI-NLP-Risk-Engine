// Mirrors src/riskengine/schema.py and the API responses.

export type SourceType = 'news' | 'social' | 'filing'

export interface Signal {
  signal_id: string
  item_id: string
  cluster_id: string
  entity: string
  entity_type: 'company' | 'market'
  source: string
  source_type: SourceType
  published_at: string
  processed_at: string
  sentiment: { score: number; confidence: number; label: 'positive' | 'neutral' | 'negative' }
  event: { primary: string; labels: string[]; probs: Record<string, number>; subtype: string | null }
  impact: { score: number; raw: number; confidence: number; method: string }
  scope: { countries: string[]; sectors: string[]; commodities: string[]; currencies: string[] }
  channels: string[]
  novelty: number
  n_sources: number
  source_weight: number
  evidence: { headline: string; url: string | null; span: string | null }
  model_version: string
}

export interface UniverseMember {
  ticker: string
  name: string
  sector: string
  benchmark_weight: number
}

export interface TaxonomyClass {
  label: string
  description: string
  subtypes: string[]
  base_impact: number
  channels: string[]
}

export interface TickerSignalState {
  tilt: number
  sentiment: number
  evidence: number
  n: number
}

export interface IndexSnapshot {
  ts: string
  weights: Record<string, number>
  benchmark: Record<string, number>
  signals: Record<string, TickerSignalState>
  turnover: number
  reason: string
  method: 'black_litterman' | 'exp_tilt'
  ex_ante_te: number | null
  alpha_bps: Record<string, number>
  trigger_signal_ids: string[]
  degraded?: string | null
}

export interface CurvePoint { severity: number; cet1_ratio: number; loss: number }
export interface Breach { severity: number; impact_equivalent: number }
export interface SeverityCurve {
  points: CurvePoint[]
  requirement: number
  minimum: number
  breach_buffer: Breach | null
  breach_minimum: Breach | null
}

export interface ScenarioLibraryEntry {
  key: string
  name: string
  episode: { start: string; end: string }
  measured: {
    rates_bp?: Record<string, number>
    credit_spread_bp?: Record<string, number>
    equity_pct?: Record<string, number>
    fx_pct?: Record<string, number>
    commodity_pct?: Record<string, number>
    vix_change?: number
  }
  credit_z: Record<string, number>
  revolver_drawdown: number | null
  template: Record<string, Record<string, number>>
}

export interface CompareRow {
  key: string
  name: string
  episode: { start: string; end: string }
  summary: StressSummary
  attribution: Record<string, number>
}

export interface StressSummary {
  total_loss: number
  mtm_loss: number
  ecl_increase: number
  portfolio_value: number
  portfolio_value_stressed: number
  ecl_base: number
  ecl_stressed: number
  rwa_base: number
  rwa_stressed: number
  cet1_base: number
  cet1_stressed: number
  cet1_ratio_base: number
  cet1_ratio_stressed: number
  cet1_requirement: number
  cet1_minimum: number
  breaches_buffer: boolean
  breaches_minimum: boolean
  stage2_exposure: number
  units: string
}

export interface ScenarioInfo {
  key: string
  name: string
  severity: number
  kind: 'systemic' | 'idiosyncratic' | 'custom'
  scope_countries: string[]
  target_counterparty: string | null
  force_default: boolean
  trigger: Record<string, unknown>
}

export interface GroupRow {
  exposure: number
  mtm_pnl: number
  ecl_increase: number
  loss: number
  [key: string]: string | number
}

export interface StressRun {
  run_id: string
  at: string
  scenario: ScenarioInfo
  summary: StressSummary
  attribution: Record<string, number>
  by_asset_class?: GroupRow[]
  by_sector?: GroupRow[]
  by_country?: GroupRow[]
  by_rating_migration?: { rating_orig: string; rating_stressed: string; exposure: number; n: number }[]
  top_losses?: Record<string, string | number | null>[]
}

export interface PortfolioOverview {
  n_positions: number
  total_notional: number
  market_value: number
  baseline: StressSummary
  by_instrument: Record<string, number>
  by_sector: Record<string, number>
  by_rating: Record<string, number>
  by_country: Record<string, number>
}

export interface ReplayStatus {
  running: boolean
  name: string | null
  speed: number
  sim_time: string | null
  processed: number
  total: number
  signals: number
}
