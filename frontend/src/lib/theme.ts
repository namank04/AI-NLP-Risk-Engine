// Data colours. Categorical slots are used in this fixed order (validated for
// colour-vision deficiency on the panel surface #11151c); never cycled.
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'] as const

export const INK = { primary: '#eef1f5', secondary: '#aab2bf', muted: '#6f7886' }
export const SURFACE = { page: '#0b0e13', panel: '#11151c', raised: '#171c25' }
export const GRID = '#20262f'
export const AXIS = '#343c4a'
export const DEEMPH = '#3a4250' // the "rest" in emphasis charts

// Diverging (sentiment, active weight, P&L): blue = positive, red = negative,
// neutral grey midpoint. One scale everywhere so one reading rule applies.
export const POS = '#3987e5'
export const NEG = '#e66767'
export const NEUTRAL = '#353a44'
export const DIVERGING_STOPS = [NEG, NEUTRAL, POS]

// Sequential (magnitude): one hue, from near-surface to full strength.
export const SEQUENTIAL_STOPS = ['#18243a', '#2a5fa6', '#6da7ec']

export const STATUS = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' }

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
const mix = (a: string, b: string, t: number) => {
  const [x, y] = [hex(a), hex(b)]
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(',')})`
}

/** Colour for a signed value on the diverging scale (|v| / max saturates at the pole). */
export function divergingColor(v: number, max = 1): string {
  const t = Math.max(-1, Math.min(1, v / (max || 1)))
  return t >= 0 ? mix(NEUTRAL, POS, t) : mix(NEUTRAL, NEG, -t)
}
