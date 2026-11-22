// Pressure transient analysis: automatic build-up interpretation for every detected shut-in.
import { useEffect, useMemo, useRef, useState } from 'react'
import { CircleSlash, ClipboardCheck, Download, RotateCcw, SlidersHorizontal, Sparkles, Timer } from 'lucide-react'
import LogRangeSlider from '../components/LogRangeSlider'
import Plot from '../components/Plot'
import { Card, ErrorBox, Pill, Skeleton } from '../components/ui'
import { COLORS, fmt, useApi } from '../lib'

const BOUNDARY_LABEL = { infinite: 'Infinite-acting', constant_pressure: 'Pressure support', closed: 'Closed boundary' }
const MODEL_STYLE = { infinite: 'dot', constant_pressure: 'dash', closed: 'dashdot' }

const isTest = (e) => e.event_type === 'planned build-up test'
function skinTone(s) { return s > 3 ? 'bad' : s > 1 ? 'warn' : 'good' }
function hours(h) { return h < 1 ? `${fmt(h * 60, 0)} min` : `${fmt(h, h < 10 ? 2 : 1)} h` }

// ---------- event selector ----------
function EventStrip({ events, selected, onSelect }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {events.map((e) => {
        const active = e.event_id === selected
        return (
          <button key={e.event_id} onClick={() => onSelect(e.event_id)}
            className={`rise rounded-xl border p-3.5 text-left transition
              ${!isTest(e) && !active ? 'opacity-70 ' : ''}${active ? 'border-accent/50 bg-accent/[0.07] shadow-[0_0_0_1px_rgba(245,181,68,0.25),0_10px_30px_-12px_rgba(245,181,68,0.35)]'
                : 'border-line bg-panel/70 hover:border-line-strong hover:bg-panel'}`}>
            <div className="flex items-center justify-between">
              <span className="font-semibold text-ink">{e.well_name}</span>
              <Pill tone={skinTone(e.skin)}>S {e.skin.toFixed(1)}</Pill>
            </div>
            <div className="num mt-1.5 text-[11px] text-muted">{e.t_shut.slice(0, 16).replace('T', ' ')}</div>
            <div className="mt-2 flex items-center justify-between text-[11px]">
              <span className="flex items-center gap-1 text-faint"><Timer size={11} />{hours(e.duration_hr)}</span>
              <span className="num text-muted">{fmt(e.k_md, 0)} md</span>
            </div>
            <div className="mt-2.5 border-t border-line pt-2">
              {isTest(e) ? <Pill tone="good"><ClipboardCheck size={11} />Build-up test</Pill>
                : <Pill tone="neutral"><CircleSlash size={11} />Unplanned shut-in</Pill>}
            </div>
          </button>
        )
      })}
    </div>
  )
}

// ---------- plots ----------
function HistoryPlot({ h }) {
  const data = [
    { x: h.time, y: h.pressure, name: 'Bottomhole pressure', type: 'scatter', mode: 'lines',
      line: { color: COLORS.accent, width: 1.6 } },
    { x: h.rate_date, y: h.oil_rate, name: 'VFM oil rate', type: 'scatter', mode: 'lines', yaxis: 'y2',
      line: { color: COLORS.oil, width: 1.6, shape: 'hv' }, fill: 'tozeroy', fillcolor: 'rgba(52,199,123,0.08)' },
  ]
  const layout = {
    yaxis: { title: { text: 'Pressure (psia)' } },
    yaxis2: { title: { text: 'Oil (STB/d)' }, overlaying: 'y', side: 'right', showgrid: false, rangemode: 'tozero' },
    shapes: [{ type: 'rect', x0: h.t_shut, x1: h.t_open, yref: 'paper', y0: 0, y1: 1,
      fillcolor: 'rgba(255,255,255,0.04)', line: { width: 0 } }],
    annotations: [{ x: h.t_shut, yref: 'paper', y: 1, text: 'shut-in', showarrow: false, xanchor: 'left',
      font: { size: 10, color: COLORS.muted } }],
    margin: { l: 60, r: 60, t: 44, b: 44 },
  }
  return <Plot data={data} layout={layout} height={330} />
}

function LogLogPlot({ d, s }) {
  const der = d.derivative
  const ys = [...der.dp, ...der.der].filter((v) => v > 0)
  const lo = Math.min(...ys) * 0.6, hi = Math.max(...ys) * 1.6
  const data = [
    { x: [s.radial_start_hr, s.radial_end_hr, s.radial_end_hr, s.radial_start_hr], y: [lo, lo, hi, hi],
      fill: 'toself', fillcolor: 'rgba(52,199,123,0.10)', line: { width: 0 }, name: 'Radial flow window',
      hoverinfo: 'skip', type: 'scatter', mode: 'lines' },
    { x: d.buildup.dt, y: d.buildup.dp, name: 'ΔP', type: 'scatter', mode: 'markers',
      marker: { color: COLORS.water, size: 3.5 } },
    { x: der.dt, y: der.der, name: "Derivative ΔP'", type: 'scatter', mode: 'markers+lines',
      marker: { color: COLORS.gas, size: 5 }, line: { color: 'rgba(240,96,93,0.45)', width: 1 } },
    { x: [d.limits.dt_min, d.limits.dt_max], y: [s.m_psi_cycle / 2.303, s.m_psi_cycle / 2.303], name: 'm / 2.303 (radial)',
      type: 'scatter', mode: 'lines', line: { color: COLORS.oil, dash: 'dash', width: 1.2 } },
  ]
  const layout = {
    xaxis: { type: 'log', title: { text: 'Δt since shut-in (h)' } },
    yaxis: { type: 'log', title: { text: "ΔP, ΔP' (psi)" }, range: [Math.log10(lo), Math.log10(hi)] },
    hovermode: 'closest',
  }
  return <Plot data={data} layout={layout} height={330} />
}

function HornerPlot({ d, s }) {
  const b = d.buildup
  const inWindow = b.dt.map((t) => t >= s.radial_start_hr && t <= s.radial_end_hr)
  const data = [
    { x: b.horner_time.filter((_, i) => !inWindow[i]), y: b.pws.filter((_, i) => !inWindow[i]), name: 'Build-up',
      type: 'scatter', mode: 'markers', marker: { color: 'rgba(141,154,171,0.45)', size: 3.5 } },
    { x: b.horner_time.filter((_, i) => inWindow[i]), y: b.pws.filter((_, i) => inWindow[i]), name: 'Radial flow',
      type: 'scatter', mode: 'markers', marker: { color: COLORS.water, size: 5 } },
    { x: d.horner_line.horner_time, y: d.horner_line.pws, name: 'Horner line',
      type: 'scatter', mode: 'lines', line: { color: COLORS.accent, width: 2 },
      hovertemplate: `m = ${s.m_psi_cycle.toFixed(2)} psi/cycle<extra></extra>` },
    { x: [1], y: [s.p_star_psia], name: 'P*', type: 'scatter', mode: 'markers',
      marker: { symbol: 'star', size: 14, color: COLORS.accent },
      hovertemplate: `P* = ${fmt(s.p_star_psia)} psia<extra></extra>` },
  ]
  const layout = {
    xaxis: { type: 'log', autorange: 'reversed', title: { text: 'Horner time (tp + Δt) / Δt' } },
    yaxis: { title: { text: 'Shut-in pressure Pws (psia)' } },
    hovermode: 'closest',
  }
  return <Plot data={data} layout={layout} height={330} />
}

function ModelMatchPlot({ d }) {
  const m = d.model
  const data = [
    { x: d.derivative.dt, y: d.derivative.dp, name: 'ΔP measured', type: 'scatter', mode: 'markers',
      marker: { color: COLORS.water, size: 5 } },
    { x: d.derivative.dt, y: d.derivative.der, name: "ΔP' measured", type: 'scatter', mode: 'markers',
      marker: { color: COLORS.gas, size: 5, symbol: 'square' } },
  ]
  for (const c of m.curves) {
    const style = { color: c.chosen ? COLORS.ink : 'rgba(231,237,244,0.35)', width: c.chosen ? 2.2 : 1, dash: MODEL_STYLE[c.boundary] }
    const name = `${BOUNDARY_LABEL[c.boundary]}${c.chosen ? ' ✓' : ''}`
    data.push({ x: m.dt, y: c.dp, name, type: 'scatter', mode: 'lines', line: style, legendgroup: c.boundary })
    data.push({ x: m.dt, y: c.der.map((v) => (v > 0 ? v : null)), name, type: 'scatter', mode: 'lines', line: style,
      legendgroup: c.boundary, showlegend: false })
  }
  const layout = {
    xaxis: { type: 'log', title: { text: 'Δt since shut-in (h)' } },
    yaxis: { type: 'log', title: { text: "ΔP, ΔP' (psi)" } },
    hovermode: 'closest', margin: { l: 56, r: 24, t: 56, b: 44 },
  }
  return <Plot data={data} layout={layout} height={330} />
}

// ---------- results panel ----------
function Big({ label, value, unit, tone, auto }) {
  return (
    <div className="rounded-xl border border-line bg-white/[0.02] p-3.5">
      <div className="text-[11px] uppercase tracking-[0.1em] text-faint">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className={`num text-2xl ${tone || 'text-ink'}`}>{value}</span>
        <span className="text-xs text-muted">{unit}</span>
      </div>
      {auto !== undefined && <div className="num mt-1 text-[10px] text-faint">auto: {auto}</div>}
    </div>
  )
}

function Row({ label, value, unit }) {
  return (
    <div className="flex items-baseline justify-between py-1.5 text-[13px]">
      <span className="text-muted">{label}</span>
      <span className="num text-ink">{value}<span className="ml-1 text-[10px] text-faint">{unit}</span></span>
    </div>
  )
}

function ResultsPanel({ d, win, setWin, busy }) {
  const s = d.summary, a = d.auto
  const manual = s.manual_window
  const toneClass = { bad: 'text-bad', warn: 'text-warn', good: 'text-good' }[skinTone(s.skin)]
  const best = d.model.curves.find((c) => c.chosen)
  return (
    <div className="space-y-5">
      <Card title="Reservoir properties" subtitle="Horner analysis · rate from the VFM"
        actions={manual ? <Pill tone="warn">Manual window</Pill> : <Pill tone="good"><Sparkles size={11} />Auto</Pill>}>
        <div className={`grid grid-cols-2 gap-3 transition-opacity ${busy ? 'opacity-50' : ''}`}>
          <Big label="Permeability k" value={fmt(s.k_md, 1)} unit="md" auto={manual ? fmt(a.k_md, 1) : undefined} />
          <Big label="Skin" value={s.skin.toFixed(2)} tone={toneClass} auto={manual ? a.skin.toFixed(2) : undefined} />
          <Big label="Horner slope m" value={fmt(s.m_psi_cycle, 2)} unit="psi/cycle" auto={manual ? fmt(a.m_psi_cycle, 2) : undefined} />
          <Big label="ΔP skin" value={fmt(s.dP_skin_psi, 1)} unit="psi" auto={manual ? fmt(a.dP_skin_psi, 1) : undefined} />
        </div>
        <div className="mt-4 divide-y divide-line border-t border-line pt-2">
          <Row label="Extrapolated pressure P*" value={fmt(s.p_star_psia)} unit="psia" />
          <Row label="Flowing pressure Pwf (Δt = 0)" value={fmt(s.pwf_psia)} unit="psia" />
          <Row label="Flow efficiency" value={s.flow_efficiency.toFixed(2)} />
          <Row label="Rate before shut-in (VFM)" value={fmt(s.q_oil_stbd)} unit="STB/d" />
          <Row label="Radius of investigation" value={fmt(s.radius_investigation_ft)} unit="ft" />
        </div>
      </Card>

      <Card title="Radial-flow window" subtitle="Where the Horner line is fitted (flat derivative)"
        actions={<SlidersHorizontal size={15} className="text-faint" />}>
        <LogRangeSlider min={d.limits.dt_min} max={d.limits.dt_max} start={win.start} end={win.end}
          onChange={(start, end) => setWin({ start, end, manual: true })} />
        <div className="num mt-3 flex justify-between text-xs text-muted">
          <span>{hours(win.start)}</span><span className="text-faint">log scale</span><span>{hours(win.end)}</span>
        </div>
        <button onClick={() => setWin({ start: a.radial_start_hr, end: a.radial_end_hr, manual: false })}
          disabled={!manual}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-line py-2 text-xs text-muted
            transition hover:border-line-strong hover:text-ink disabled:opacity-40">
          <RotateCcw size={13} />Reset to automatic ({hours(a.radial_start_hr)} – {hours(a.radial_end_hr)})
        </button>
      </Card>

      <Card title="Model match" subtitle="Full analytical model fitted to the whole build-up">
        <Row label="Boundary" value={BOUNDARY_LABEL[s.model_boundary]} />
        {s.model_boundary_ft && <Row label="Boundary distance (approx.)" value={`~${fmt(Math.round(s.model_boundary_ft / 10) * 10)}`} unit="ft" />}
        <Row label="Wellbore storage C" value={s.model_C_bbl_psi.toFixed(3)} unit="bbl/psi" />
        <Row label="k / skin (model)" value={`${fmt(best.k_md, 1)} / ${best.skin.toFixed(2)}`} />
        <Row label="Fit error (RMS)" value={best.rms_pct.toFixed(2)} unit="%" />
      </Card>
    </div>
  )
}

function Interpretation({ text }) {
  const parts = text.split('; ')
  return (
    <Card title="Interpretation" subtitle="Automatic engineering reading of this build-up">
      <ul className="space-y-2.5">
        {parts.map((p, i) => (
          <li key={i} className="flex gap-3 text-[13px] text-ink">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
            <span>{p.charAt(0).toUpperCase() + p.slice(1)}</span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

function ResultsTable({ events: all, selected, onSelect }) {
  const head = ['Well', 'Shut-in', 'Duration', 'q (VFM)', 'Horner m', 'k', 'Skin', 'ΔP skin', 'Flow eff.', 'Boundary']
  const events = all.filter(isTest)
  return (
    <Card title="Build-up tests · deliverable" subtitle="Horner slope, skin, permeability and ΔP skin for every planned build-up test"
      bodyClass="overflow-x-auto"
      actions={<a href="/api/pta/export.xlsx" className="flex items-center gap-2 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent
        shadow-[0_8px_24px_-10px_rgba(245,181,68,0.8)] transition hover:brightness-110"><Download size={14} />Export Excel</a>}>
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
          {head.map((h) => <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {events.map((e) => (
            <tr key={e.event_id} onClick={() => onSelect(e.event_id)}
              className={`cursor-pointer transition ${e.event_id === selected ? 'bg-accent/[0.05]' : 'hover:bg-white/[0.025]'}`}>
              <td className="px-4 py-3 font-semibold text-ink">{e.well_name}</td>
              <td className="num px-4 py-3 text-muted">{e.t_shut.slice(0, 16).replace('T', ' ')}</td>
              <td className="num px-4 py-3 text-muted">{hours(e.duration_hr)}</td>
              <td className="num px-4 py-3 text-muted">{fmt(e.q_oil_stbd)}</td>
              <td className="num px-4 py-3 text-ink">{fmt(e.m_psi_cycle, 2)}</td>
              <td className="num px-4 py-3 text-ink">{fmt(e.k_md, 1)}<span className="ml-1 text-[10px] text-faint">md</span></td>
              <td className="px-4 py-3"><Pill tone={skinTone(e.skin)}>{e.skin.toFixed(2)}</Pill></td>
              <td className="num px-4 py-3 text-ink">{fmt(e.dP_skin_psi, 1)}<span className="ml-1 text-[10px] text-faint">psi</span></td>
              <td className="num px-4 py-3 text-muted">{e.flow_efficiency.toFixed(2)}</td>
              <td className="px-4 py-3 text-muted">{BOUNDARY_LABEL[e.model_boundary]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {events.length === 0 && <div className="px-4 py-6 text-sm text-muted">No planned build-up test in this dataset.</div>}
    </Card>
  )
}

// Short unplanned shut-ins: analysed and shown, but not reported as well tests (Earlougher / Horne: a build-up
// test is a designed operation -- stabilised rate, long enough shut-in, staged restart).
function UnplannedTable({ events, selected, onSelect }) {
  const rows = events.filter((e) => !isTest(e))
  if (!rows.length) return null
  return (
    <Card title="Unplanned shut-ins · not reported" bodyClass="overflow-x-auto"
      subtitle="Analysed for information only — they fail the build-up test criterion (≥ 24 h AND staged choke restart)">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
          {['Well', 'Shut-in', 'Duration', 'Why not a build-up test', 'k (info)', 'Skin (info)'].map((h) =>
            <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((e) => (
            <tr key={e.event_id} onClick={() => onSelect(e.event_id)}
              className={`cursor-pointer transition ${e.event_id === selected ? 'bg-accent/[0.05]' : 'hover:bg-white/[0.025]'}`}>
              <td className="px-4 py-3 font-semibold text-ink">{e.well_name}</td>
              <td className="num px-4 py-3 text-muted">{e.t_shut.slice(0, 16).replace('T', ' ')}</td>
              <td className="num px-4 py-3 text-muted">{hours(e.duration_hr)}</td>
              <td className="px-4 py-3 text-muted">{e.classification_reason}</td>
              <td className="num px-4 py-3 text-faint">{fmt(e.k_md, 1)} md</td>
              <td className="num px-4 py-3 text-faint">{e.skin.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function TypeBanner({ e }) {
  if (!e) return null
  return isTest(e) ? (
    <div className="flex items-center gap-3 rounded-xl border border-good/25 bg-good/[0.05] px-4 py-2.5 text-[13px]">
      <ClipboardCheck size={16} className="shrink-0 text-good" />
      <span className="text-ink">Planned build-up test</span><span className="text-muted">· {e.classification_reason} · reported in the deliverable</span>
    </div>
  ) : (
    <div className="flex items-center gap-3 rounded-xl border border-line-strong bg-white/[0.03] px-4 py-2.5 text-[13px]">
      <CircleSlash size={16} className="shrink-0 text-muted" />
      <span className="text-ink">Unplanned shut-in — not a build-up test</span>
      <span className="text-muted">· {e.classification_reason} · analysed for information only, not reported</span>
    </div>
  )
}

// ---------- page ----------
export default function Pta() {
  const events = useApi('/api/pta/events')
  const [selected, setSelected] = useState(null)
  const [win, setWin] = useState(null)              // radial window {start, end, manual}; null = automatic
  const [detail, setDetail] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const timer = useRef(null)

  // choosing another event resets everything to that event's automatic analysis
  function selectEvent(id) {
    setSelected(id)
    setWin(null)
    setDetail(null)
  }

  // fetch the event detail (debounced while dragging the slider); old plots stay visible while loading
  // start on the first planned build-up test
  useEffect(() => {
    if (events.data && events.data.length && selected === null) {
      setSelected((events.data.find(isTest) || events.data[0]).event_id)
    }
  }, [events.data]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selected === null) return
    const showingAuto = detail && detail.summary.event_id === selected && !detail.summary.manual_window
    if (win && !win.manual && showingAuto) return            // automatic result already on screen
    clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      const query = win?.manual ? `?start=${win.start}&end=${win.end}` : ''
      setBusy(true)
      fetch(`/api/pta/event/${selected}${query}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
        .then((d) => {
          setDetail(d)
          if (!win) setWin({ start: d.summary.radial_start_hr, end: d.summary.radial_end_hr, manual: false })
        })
        .catch(setError)
        .finally(() => setBusy(false))
    }, win?.manual ? 220 : 0)
    return () => clearTimeout(timer.current)
  }, [selected, win]) // eslint-disable-line react-hooks/exhaustive-deps

  const list = events.data
  const current = useMemo(() => list?.find((e) => e.event_id === selected), [list, selected])
  if (events.error || error) return <ErrorBox error={events.error || error} />
  if (list && list.length === 0) {
    return <Card title="No shut-ins in this dataset" subtitle="Build-up analysis needs a shut-in (choke closed / gas lift stopped)">
      <div className="text-sm text-muted">The uploaded period contains no shut-in, so there is nothing to interpret. Rates are still on the VFM pages.</div></Card>
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between text-xs text-muted">
        <span>{list ? <><span className="num text-ink">{list.length}</span> shut-ins detected automatically (choke = 0 %) · <span className="num text-ink">{list.filter(isTest).length}</span> planned build-up tests · rate q from the virtual flow meter</> : 'Detecting shut-ins…'}</span>
        {current && <span>Selected: <span className="text-ink">{current.well_name}</span> · {current.t_shut.slice(0, 10)}</span>}
      </div>
      {list ? <EventStrip events={list} selected={selected} onSelect={selectEvent} /> : <Skeleton className="h-24" />}

      <TypeBanner e={current} />
      {!detail || !win ? (
        <div className="grid grid-cols-12 gap-5"><Skeleton className="col-span-8 h-[640px]" /><Skeleton className="col-span-4 h-[640px]" /></div>
      ) : (
        <div className="grid grid-cols-12 gap-5">
          <div className="col-span-12 space-y-5 xl:col-span-8">
            <div className="grid grid-cols-1 gap-5 2xl:grid-cols-2">
              <Card title="Pressure & rate history" subtitle="Downhole gauge (5-min) and VFM daily oil rate"><HistoryPlot h={detail.history} /></Card>
              <Card title="Log-log diagnostic" subtitle="Flat derivative = radial flow → Horner window (green)"><LogLogPlot d={detail} s={detail.summary} /></Card>
              <Card title="Horner plot" subtitle="Straight line through radial flow · slope m, extrapolated P*"><HornerPlot d={detail} s={detail.summary} /></Card>
              <Card title="Analytical model match" subtitle="Storage + skin + boundary models · ✓ = selected (AIC)"><ModelMatchPlot d={detail} /></Card>
            </div>
            <Interpretation text={detail.summary.interpretation} />
          </div>
          <div className="col-span-12 xl:col-span-4">
            <ResultsPanel d={detail} win={win} setWin={setWin} busy={busy} />
          </div>
        </div>
      )}

      {list && <ResultsTable events={list} selected={selected} onSelect={selectEvent} />}
      {list && <UnplannedTable events={list} selected={selected} onSelect={selectEvent} />}
    </div>
  )
}
