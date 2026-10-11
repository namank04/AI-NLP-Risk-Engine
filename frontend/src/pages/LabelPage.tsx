import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ExternalLink } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { TaxonomyClass } from '../api/types'
import { Button, Empty, Panel, SourceTag } from '../components/ui'
import { eventLabel, titleCase, when } from '../lib/format'
import { NEG, POS } from '../lib/theme'

interface PoolItem {
  item_id: string
  source: string
  source_type: string
  title: string
  body: string
  url: string | null
  publisher: string | null
  published_at: string
  suggestions: { entities: { entity: string; sentiment: number }[]; events: string[]; subtype: string | null }
}
export interface GoldStats {
  pool: number; labeled: number; skipped: number; remaining: number
  baseline_event_changed: number | null; baseline_sentiment_changed: number | null
}
interface EntityState { entity: string; relevant: boolean; sentiment: number }

const KEYS: Record<string, string> = { g: 'geopolitical', m: 'macroeconomic', c: 'credit_event', a: 'merger_acquisition', e: 'earnings',
  p: 'product_launch', l: 'regulatory_legal', x: 'management_change', o: 'operational', n: 'none' }
const KEY_OF = Object.fromEntries(Object.entries(KEYS).map(([k, v]) => [v, k]))
const SENT = [-2, -1, 0, 1, 2]

async function jget<T>(u: string): Promise<T> { const r = await fetch(u); if (!r.ok) throw new Error(await r.text()); return r.json() }
async function jpost<T>(u: string, body: unknown): Promise<T> {
  const r = await fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(await r.text()); return r.json()
}

function Kbd({ children }: { children: string }) {
  return <kbd className="rounded border border-line-strong px-1 font-mono text-[10px] text-ink-3 uppercase">{children}</kbd>
}

export default function LabelPage({ onStats }: { onStats?: (s: GoldStats) => void }) {
  const { data: tax } = useQuery({ queryKey: ['taxonomy'], queryFn: () => jget<Record<string, TaxonomyClass>>('/api/taxonomy') })
  const [item, setItem] = useState<PoolItem | null>(null)
  const [ents, setEnts] = useState<EntityState[]>([])
  const [active, setActive] = useState(0)
  const [events, setEvents] = useState<string[]>([])
  const [subtype, setSubtype] = useState('')
  const [newEnt, setNewEnt] = useState('')
  const [error, setError] = useState<string | null>(null)
  const started = useRef(Date.now())

  const load = useCallback((it: PoolItem | null) => {
    setItem(it); setError(null)
    if (!it) return
    setEnts(it.suggestions.entities.map((e) => ({ entity: e.entity, relevant: true, sentiment: e.sentiment })))
    setEvents(it.suggestions.events); setSubtype(it.suggestions.subtype ?? ''); setActive(0)
    started.current = Date.now()
  }, [])

  useEffect(() => {
    jget<{ item: PoolItem | null; stats: GoldStats }>('/api/gold/next').then((r) => { load(r.item); onStats?.(r.stats) }).catch((e) => setError(String(e)))
  }, [load, onStats])

  const submit = useMutation({
    mutationFn: (skipped: boolean) => jpost<{ next: PoolItem | null; stats: GoldStats }>('/api/gold/label', {
      item_id: item!.item_id, entities: ents, events, subtype: subtype || null, skipped, seconds: (Date.now() - started.current) / 1000 }),
    onSuccess: (r) => { onStats?.(r.stats); load(r.next) },
    onError: (e) => setError(String(e)),
  })

  const toggleEvent = useCallback((k: string) => setEvents((ev) => {
    if (k === 'none') return ev.includes('none') ? [] : ['none']
    const rest = ev.filter((x) => x !== 'none')
    return rest.includes(k) ? rest.filter((x) => x !== k) : [...rest, k]
  }), [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (!item || tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      const k = e.key.toLowerCase()
      if (k >= '1' && k <= '5' && ents.length) setEnts((s) => s.map((x, i) => (i === active ? { ...x, sentiment: SENT[Number(k) - 1] } : x)))
      else if (k === 'r' && ents.length) setEnts((s) => s.map((x, i) => (i === active ? { ...x, relevant: !x.relevant } : x)))
      else if (e.key === 'Tab' && ents.length) { e.preventDefault(); setActive((a) => (a + 1) % ents.length) }
      else if (KEYS[k]) toggleEvent(KEYS[k])
      else if (e.key === 'Enter' && !submit.isPending && events.length) submit.mutate(false)
      else if (k === 's' && !submit.isPending) submit.mutate(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [item, ents.length, active, submit, toggleEvent, events.length])

  if (error) return <div className="rounded-xl border border-critical/40 px-4 py-3 text-sm text-critical">{error}</div>
  if (!item) return <Panel><Empty tall>Nothing left to label, or no pool yet (python scripts/build_gold_pool.py).</Empty></Panel>
  const primary = events[0]

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Panel title={<span className="flex items-center gap-2 font-normal"><SourceTag type={item.source_type} />
        <span className="text-ink-3">· {item.publisher ?? item.source} · {when(item.published_at)}</span></span>}>
        <h3 className="text-lg leading-snug font-semibold">{item.title || 'Social post'}</h3>
        {item.body && <p className="mt-3 max-h-[460px] overflow-y-auto text-[14px] leading-relaxed whitespace-pre-wrap text-ink-2">{item.body}</p>}
        {item.url && <a href={item.url} target="_blank" rel="noreferrer" className="mt-4 inline-flex items-center gap-1 text-xs text-accent hover:underline">Open source <ExternalLink size={12} /></a>}
      </Panel>
      <div className="space-y-6">
        <Panel title="Entities" actions={<span className="flex items-center gap-1.5 text-[11px] text-ink-3"><Kbd>tab</Kbd> next <Kbd>1</Kbd>–<Kbd>5</Kbd> tone <Kbd>r</Kbd> relevant</span>}>
          <div className="space-y-2">
            {ents.map((e, i) => (
              <div key={e.entity} onClick={() => setActive(i)} className={clsx('rounded-xl border p-3', i === active ? 'border-accent/70' : 'border-line', !e.relevant && 'opacity-50')}>
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{e.entity === 'MARKET' ? 'Market' : e.entity}</span>
                  <button onClick={() => setEnts((s) => s.map((x, j) => (j === i ? { ...x, relevant: !x.relevant } : x)))}
                    className="text-xs text-ink-3 hover:text-ink-2">{e.relevant ? 'Relevant' : 'Not relevant'}</button>
                </div>
                <div className="mt-2 grid grid-cols-5 gap-1" role="radiogroup" aria-label={`Sentiment toward ${e.entity}`}>
                  {SENT.map((v) => (
                    <button key={v} role="radio" aria-checked={e.sentiment === v}
                      onClick={() => setEnts((s) => s.map((x, j) => (j === i ? { ...x, sentiment: v } : x)))}
                      className={clsx('rounded-md py-1 text-xs font-medium', e.sentiment === v ? 'text-white' : 'bg-raised text-ink-3 hover:text-ink-2')}
                      style={e.sentiment === v ? { background: v > 0 ? POS : v < 0 ? NEG : '#4a5262' } : undefined}>
                      {v > 0 ? `+${v}` : v}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <form className="flex gap-2" onSubmit={(ev) => { ev.preventDefault(); const t = newEnt.trim().toUpperCase()
              if (t && !ents.some((x) => x.entity === t)) setEnts((s) => [...s, { entity: t, relevant: true, sentiment: 0 }]); setNewEnt('') }}>
              <input value={newEnt} onChange={(e) => setNewEnt(e.target.value)} placeholder="Add a missed entity (ticker or MARKET)" aria-label="Add entity"
                className="flex-1 rounded-lg border border-line bg-page px-2.5 py-1.5 text-xs outline-none focus:border-accent/60" />
              <Button tone="quiet" type="submit">Add</Button>
            </form>
          </div>
        </Panel>
        <Panel title="Event classes" meta="First pick is the primary class">
          <div className="grid grid-cols-2 gap-1.5">
            {Object.keys(KEY_OF).map((k) => {
              const idx = events.indexOf(k)
              return (
                <button key={k} onClick={() => toggleEvent(k)} aria-pressed={idx >= 0}
                  className={clsx('flex items-center justify-between rounded-lg border px-2.5 py-1.5 text-left text-xs',
                    idx >= 0 ? 'border-accent bg-accent-soft text-ink' : 'border-line text-ink-3 hover:text-ink-2')}>
                  <span>{idx === 0 && <span className="mr-1 text-accent">●</span>}{eventLabel(k)}</span>
                  <Kbd>{KEY_OF[k]}</Kbd>
                </button>
              )
            })}
          </div>
          {primary && primary !== 'none' && tax?.[primary]?.subtypes.length ? (
            <select value={subtype} onChange={(e) => setSubtype(e.target.value)} aria-label="Subtype"
              className="mt-3 w-full rounded-lg border border-line bg-page px-2.5 py-1.5 text-xs">
              <option value="">Subtype (optional)</option>
              {tax[primary].subtypes.map((s) => <option key={s} value={s}>{titleCase(s)}</option>)}
            </select>
          ) : null}
        </Panel>
        <div className="flex gap-2">
          <Button onClick={() => submit.mutate(false)} disabled={submit.isPending || !events.length} className="flex-1 py-2">Save and next <Kbd>enter</Kbd></Button>
          <Button tone="quiet" onClick={() => submit.mutate(true)} disabled={submit.isPending}>Skip <Kbd>s</Kbd></Button>
        </div>
      </div>
    </div>
  )
}
