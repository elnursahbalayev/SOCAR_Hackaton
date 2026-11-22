// PTA validation: synthetic benchmark with known answers, cross-checks per event, and sensitivity of the selected event.
import Plot from '../components/Plot'
import { Card, ErrorBox, Pill, Skeleton } from '../components/ui'
import { COLORS, fmt, useApi } from '../lib'

const METHOD_LABEL = {
  'Horner auto (v5)': 'Horner, automatic (ours)', 'Horner hourly data (F)': 'Horner on hourly data',
  'Horner P1hr measured (E)': 'Horner, measured P1hr', 'Horner tp since 2025 (C)': 'Horner, tp since 2025',
  'Horner window 1h-end (D)': 'Horner, fixed window 1 h → end', 'Superposition Horner': 'Superposition Horner',
  'Model match (best AIC)': 'Model match (best AIC)', 'Model match (infinite)': 'Model match (infinite)',
}
const ours = (m) => m === 'Horner auto (v5)'
const color = (m) => (ours(m) ? COLORS.accent : m.startsWith('Model') ? COLORS.oil : '#4a5566')

function Synthetic({ rows }) {
  const name = (r) => METHOD_LABEL[r.method] || r.method
  const bar = (key, p90, title, unit) => [{
    type: 'bar', orientation: 'h', y: rows.map((r) => `${name(r)} · ${r[key].toFixed(2)}${unit}`), x: rows.map((r) => r[key]),
    marker: { color: rows.map((r) => color(r.method)) },
    error_x: { type: 'data', symmetric: false, array: rows.map((r) => r[p90] - r[key]), arrayminus: rows.map(() => 0), color: COLORS.muted, thickness: 1 },
    name: title, hovertemplate: '%{y} (median; whisker = 90th percentile)<extra></extra>',
  }]
  const layout = (t) => ({ hovermode: 'closest', showlegend: false, xaxis: { title: { text: t } }, yaxis: { autorange: 'reversed' },
    margin: { l: 230, r: 20, t: 20, b: 44 } })
  return (
    <Card title="Accuracy on 180 synthetic build-ups with known answers"
      subtitle="Our simulator + real PVT + real VFM rate history + 0.3 psi gauge noise · bar = median error, whisker = 90th percentile · amber = our method">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Plot data={bar('k_err_median', 'k_err_p90', 'k', ' %')} layout={layout('|k error| %')} height={320} />
        <Plot data={bar('skin_err_median', 'skin_err_p90', 'skin', '')} layout={layout('|skin error|')} height={320} />
      </div>
    </Card>
  )
}

function CrossCheck({ events, selected, onSelect }) {
  const head = ['Well', 'Shut-in', 'Type', 'k Horner / model', 'Skin Horner / model', 'Boundary', 'Window', 'Derivative flatness', 'Build-up end vs VFM P_res']
  return (
    <Card title="Cross-checks per shut-in" bodyClass="overflow-x-auto"
      subtitle="Two independent interpretations (Horner vs full model match) · quality of the radial-flow window · agreement with the VFM">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
          {head.map((h) => <th key={h} className="whitespace-nowrap px-3 py-2.5 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {events.map((e) => {
            const dk = (100 * (e.k_model - e.k_horner)) / e.k_horner
            const dp = e.p_end_buildup != null ? e.p_end_buildup - e.p_res_vfm : null
            return (
              <tr key={e.event_id} onClick={() => onSelect(e.event_id)}
                className={`cursor-pointer ${e.event_id === selected ? 'bg-accent/[0.06]' : 'hover:bg-white/[0.025]'}`}>
                <td className="whitespace-nowrap px-3 py-2 font-semibold text-ink">{e.well}</td>
                <td className="num whitespace-nowrap px-3 py-2 text-muted">{e.t_shut}</td>
                <td className="whitespace-nowrap px-3 py-2"><Pill tone={e.event_type === 'planned build-up test' ? 'good' : 'neutral'}>{e.event_type === 'planned build-up test' ? 'build-up test' : 'unplanned'}</Pill></td>
                <td className="num whitespace-nowrap px-3 py-2 text-ink">{fmt(e.k_horner, 1)} / {fmt(e.k_model, 1)} <span className="text-faint">({dk > 0 ? '+' : ''}{dk.toFixed(1)} %)</span></td>
                <td className="num whitespace-nowrap px-3 py-2 text-ink">{e.skin_horner.toFixed(2)} / {e.skin_model.toFixed(2)}</td>
                <td className="whitespace-nowrap px-3 py-2 text-muted">{e.boundary.replace('_', ' ')}</td>
                <td className="num whitespace-nowrap px-3 py-2 text-muted">{e.log_cycles.toFixed(2)} log cycle</td>
                <td className="whitespace-nowrap px-3 py-2"><Pill tone={e.flatness_cv_pct < 5 ? 'good' : 'warn'}>{e.flatness_cv_pct != null ? `±${e.flatness_cv_pct.toFixed(1)} %` : '—'}</Pill></td>
                <td className="num whitespace-nowrap px-3 py-2 text-muted">{dp != null ? `${dp > 0 ? '+' : ''}${fmt(dp, 1)} psi` : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="px-3 pb-3 pt-2 text-[11px] leading-relaxed text-faint">
        Derivative flatness = scatter of the Bourdet derivative inside the radial-flow window (std / mean): small = a clean radial-flow
        straight line. Build-up end vs VFM: close where the build-up reaches the reservoir pressure (pressure support); otherwise the
        pressure is still rising when the well reopens (lower bound). Click a row to see its sensitivity below.
      </p>
    </Card>
  )
}

function Sensitivity({ eventId, label }) {
  const { data, error } = useApi(eventId == null ? null : `/api/pta/event/${eventId}/sensitivity`)
  if (error) return <ErrorBox error={error} />
  const plot = (key, title, unit) => [{
    type: 'bar', orientation: 'h', y: data.map((r) => r.case), x: data.map((r) => r[key]),
    marker: { color: data.map((r) => (r.case.startsWith('q') ? COLORS.water : r[key] >= 0 ? COLORS.accent : '#4a5566')) },
    text: data.map((r) => `${r[key] > 0 ? '+' : ''}${r[key].toFixed(2)}${unit}`), textposition: 'outside', cliponaxis: false, name: title,
  }]
  const range = (key) => { const m = Math.max(...data.map((r) => Math.abs(r[key]))) || 1; return [-m * 1.6, m * 1.6] }
  const layout = (t, key) => ({ hovermode: 'closest', showlegend: false, xaxis: { title: { text: t }, zeroline: true, range: range(key) },
    yaxis: { autorange: 'reversed' }, margin: { l: 150, r: 20, t: 20, b: 44 } })
  return (
    <Card title={`Sensitivity · ${label}`}
      subtitle="Each judgement call of the automatic analysis changed on its own: derivative smoothing L, minimum window width, window start / end, VFM rate q ±5 %">
      {!data ? <Skeleton className="h-72" /> : (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <Plot data={plot('k_change_pct', 'k', ' %')} layout={layout('change of k (%)', 'k_change_pct')} height={330} />
          <Plot data={plot('skin_change', 'skin', '')} layout={layout('change of skin', 'skin_change')} height={330} />
        </div>
      )}
    </Card>
  )
}

export default function PtaValidation({ selected, onSelect }) {
  const { data, error } = useApi('/api/pta/validation')
  if (error) return <ErrorBox error={error} />
  if (!data) return <Skeleton className="h-[600px]" />
  const ev = data.events.find((e) => e.event_id === selected) || data.events[0]
  return (
    <div className="space-y-5">
      <Synthetic rows={data.synthetic} />
      {data.events.length > 0 && <CrossCheck events={data.events} selected={ev?.event_id} onSelect={onSelect} />}
      {ev && <Sensitivity eventId={ev.event_id} label={`${ev.well} ${ev.t_shut}`} />}
    </div>
  )
}
