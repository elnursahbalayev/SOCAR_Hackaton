// Reusable building blocks: Card, KpiCard, Sparkline, Pill, Skeleton.
import { ArrowDownRight, ArrowUpRight, Info } from 'lucide-react'
import { pct } from '../lib'

export function Card({ title, subtitle, actions, children, className = '', bodyClass = 'p-5' }) {
  return (
    <section className={`rise rounded-2xl border border-line bg-panel/80 backdrop-blur-sm
      shadow-[var(--card-shadow)] ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div>
            <h3 className="text-[13px] font-semibold tracking-wide text-ink">{title}</h3>
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
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
