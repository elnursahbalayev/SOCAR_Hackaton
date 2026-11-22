// Reusable building blocks: Card, KpiCard, Sparkline, Pill, Skeleton.
import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { ArrowDownRight, ArrowUpRight, Info, Maximize2, X } from 'lucide-react'
import { pct } from '../lib'

// A card knows whether it contains a chart (Plot registers itself). Chart cards get a maximise button that
// opens the whole card as a near-fullscreen overlay; Plot then redraws to fill it. Esc / click outside closes.
const CardContext = createContext(null)
export const useCard = () => useContext(CardContext)

export function Card({ title, subtitle, actions, children, className = '', bodyClass = 'p-5', expandable }) {
  const [hasPlot, setHasPlot] = useState(false)
  const [full, setFull] = useState(false)
  const canExpand = expandable ?? hasPlot
  const ctx = useMemo(() => ({ full, registerPlot: () => setHasPlot(true) }), [full])

  useEffect(() => {
    if (!full) return undefined
    const onKey = (e) => e.key === 'Escape' && setFull(false)
    const scroll = document.body.style.overflow
    document.body.style.overflow = 'hidden'                // no page scrolling behind the overlay
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = scroll }
  }, [full])

  const frame = full
    ? 'fixed inset-3 z-50 flex flex-col rounded-2xl border border-line-strong bg-panel shadow-2xl md:inset-6'
    : `rounded-2xl border border-line bg-panel/80 backdrop-blur-sm shadow-[var(--card-shadow)] ${className}`
  return (
    <CardContext.Provider value={ctx}>
      {full && <div className="fixed inset-0 z-40 bg-black/55 backdrop-blur-[2px]" onClick={() => setFull(false)} />}
      <section className={`rise ${frame}`}>
        {(title || actions || canExpand) && (
          <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
            <div>
              <h3 className={`font-semibold tracking-wide text-ink ${full ? 'text-[15px]' : 'text-[13px]'}`}>{title}</h3>
              {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
            </div>
            <div className="flex items-center gap-2">
              {actions}
              {canExpand && (
                <button onClick={() => setFull(!full)} title={full ? 'Close (Esc)' : 'Enlarge'}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-line text-faint transition hover:border-line-strong hover:text-ink">
                  {full ? <X size={15} /> : <Maximize2 size={13} />}
                </button>
              )}
            </div>
          </header>
        )}
        <div className={full ? `${bodyClass} min-h-0 flex-1 overflow-auto` : bodyClass}>{children}</div>
      </section>
    </CardContext.Provider>
  )
}

// Tiny SVG trend line (no axes) used in KPI tiles and tables.
export function Sparkline({ values, color, width = 120, height = 32, fill = true }) {
  const v = values.filter((x) => x !== null && x !== undefined)
  if (v.length < 2) return null
  const min = Math.min(...v), max = Math.max(...v), span = max - min || 1
  const pts = v.map((y, i) => [(i / (v.length - 1)) * width, height - 2 - ((y - min) / span) * (height - 4)])
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const id = `g${color.replace('#', '')}`
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {fill && <path d={`${line} L${width},${height} L0,${height} Z`} fill={`url(#${id})`} />}
      <path d={line} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  )
}

// Change badge: green when the change is "good" (goodWhenUp says which direction is good).
export function Delta({ value, goodWhenUp = true, suffix = 'vs prev 30 d' }) {
  if (value === null || value === undefined) return null
  const up = value >= 0
  const good = up === goodWhenUp
  const Icon = up ? ArrowUpRight : ArrowDownRight
  return (
    <span className="inline-flex items-center gap-1 text-[11px]">
      <span className={`inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-medium num
        ${good ? 'bg-good/10 text-good' : 'bg-bad/10 text-bad'}`}>
        <Icon size={12} strokeWidth={2.5} />{pct(value)}
      </span>
      <span className="text-faint">{suffix}</span>
    </span>
  )
}

export function KpiCard({ label, value, unit, color, delta, goodWhenUp, spark, footer, hint }) {
  return (
    <div className="rise group relative overflow-hidden rounded-2xl border border-line bg-panel/80 p-5
      transition hover:border-line-strong">
      <div className="absolute inset-x-0 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.12em] text-muted">
          <span className="h-2 w-2 rounded-full" style={{ background: color, boxShadow: `0 0 12px ${color}` }} />
          {label}
        </span>
        {hint && <span title={hint} className="text-faint hover:text-muted"><Info size={14} /></span>}
      </div>
      <div className="mt-3 flex items-end justify-between gap-3">
        <div>
          <span className="num text-[28px] font-medium leading-none text-ink">{value}</span>
          {unit && <span className="ml-1.5 text-xs text-muted">{unit}</span>}
        </div>
        {spark && <Sparkline values={spark} color={color} width={96} height={34} />}
      </div>
      <div className="mt-3 min-h-5">{delta !== undefined ? <Delta value={delta} goodWhenUp={goodWhenUp} /> : footer}</div>
    </div>
  )
}

const PILL_STYLES = {
  good: 'bg-good/10 text-good ring-good/20',
  warn: 'bg-warn/10 text-warn ring-warn/20',
  bad: 'bg-bad/10 text-bad ring-bad/20',
  info: 'bg-water/10 text-water ring-water/20',
  neutral: 'bg-white/5 text-muted ring-white/10',
}

export function Pill({ tone = 'neutral', children, dot = false }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${PILL_STYLES[tone]}`}>
      {dot && <span className="live-dot h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}

export function Skeleton({ className }) {
  return <div className={`skeleton ${className}`} />
}

export function ErrorBox({ error }) {
  return (
    <div className="rounded-xl border border-bad/30 bg-bad/5 p-4 text-sm text-bad">
      Could not load data from the backend: {String(error?.message || error)}.
      <span className="block text-muted">Is the API running? <code className="num">python -m uvicorn backend.api:app --port 8000</code></span>
    </div>
  )
}
