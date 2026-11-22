// Production forecast: next months of oil, water and gas if the wells keep today's operating conditions.
import { useState } from 'react'
import { TrendingDown } from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, KpiCard, Pill, Skeleton } from '../components/ui'
import { COLORS, compact, fmt, useApi } from '../lib'

const FLUIDS = [
  { key: 'oil', label: 'Oil', unit: 'STB/d', color: COLORS.oil, band: 'rgba(52,199,123,0.16)' },
  { key: 'water', label: 'Water', unit: 'STB/d', color: COLORS.water, band: 'rgba(76,141,255,0.16)' },
  { key: 'gas', label: 'Gas', unit: 'Mscf/d', color: COLORS.gas, band: 'rgba(240,96,93,0.16)' },
]

function ForecastPlot({ d }) {
  const traces = []
  FLUIDS.forEach((f, i) => {
    const ax = i ? `${i + 1}` : ''
    traces.push({ x: d.history.date, y: d.history[f.key], name: 'History (VFM)', showlegend: !i, legendgroup: 'h', type: 'scatter',
      mode: 'lines', line: { color: f.color, width: 1.6 }, xaxis: 'x', yaxis: `y${ax}` })
    traces.push({ x: d.future.date, y: d.future[f.key + '_p10'], showlegend: false, legendgroup: 'b', type: 'scatter', mode: 'lines',
      line: { width: 0 }, hoverinfo: 'skip', xaxis: 'x', yaxis: `y${ax}` })
    traces.push({ x: d.future.date, y: d.future[f.key + '_p90'], name: 'P10–P90 range', showlegend: !i, legendgroup: 'b', type: 'scatter',
      mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: f.band, hoverinfo: 'skip', xaxis: 'x', yaxis: `y${ax}` })
    traces.push({ x: d.future.date, y: d.future[f.key], name: 'Forecast', showlegend: !i, legendgroup: 'f', type: 'scatter', mode: 'lines',
      line: { color: f.color, width: 2, dash: 'dash' }, xaxis: 'x', yaxis: `y${ax}` })
  })
  const layout = {
    grid: { rows: 3, columns: 1, pattern: 'coupled', roworder: 'top to bottom' },
    yaxis: { title: { text: 'Oil (STB/d)' } }, yaxis2: { title: { text: 'Water (STB/d)' } }, yaxis3: { title: { text: 'Gas (Mscf/d)' } },
    shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: d.start, x1: d.start, y0: 0, y1: 1, line: { color: COLORS.muted, width: 1, dash: 'dot' } }],
    margin: { l: 64, r: 20, t: 40, b: 36 },
  }
  return <Plot data={traces} layout={layout} height={540} />
}

export default function VfmForecast() {
  const [months, setMonths] = useState(6)
  const { data: d, error } = useApi(`/api/vfm/forecast?months=${months}`)
  if (error) return <ErrorBox error={error} />
  if (!d) return <Skeleton className="h-[600px]" />
  const t = d.totals
  const change = (100 * (t.oil_end - t.oil_now)) / t.oil_now
  const head = ['Well', 'Oil now', 'In 3 months', `In ${months} months`, 'Change', `Cum. oil ${months} mo`, 'Water cut', 'P_res trend', 'Pwf − Pb']
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>Forecast from {d.start} to {d.end} · <b className="text-ink">if every well keeps today's choke and lift gas</b> (flowing pressure held constant)</span>
        <div className="flex gap-1">{[3, 6, 12].map((m) => (
          <button key={m} onClick={() => setMonths(m)} className={`rounded-lg border px-3 py-1 ${m === months ? 'border-accent/50 bg-accent/10 text-ink' : 'border-line hover:text-ink'}`}>{m} months</button>
        ))}</div>
      </div>
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <KpiCard label="Field oil today" value={fmt(t.oil_now)} unit="STB/d" color={COLORS.oil}
          footer={<span className="text-[11px] text-faint">mean of the last 30 flowing days</span>} />
        <KpiCard label={`Field oil in ${months} months`} value={fmt(t.oil_end)} unit="STB/d" color={COLORS.accent}
          footer={<span className="text-[11px] text-faint">{change.toFixed(1)} % without intervention</span>} />
        <KpiCard label={`Cumulative oil, next ${months} months`} value={compact(t.cum_oil)} unit="STB" color={COLORS.water}
          footer={<span className="text-[11px] text-faint">P10–P90 {compact(t.cum_oil_p10)} – {compact(t.cum_oil_p90)}</span>} />
        <KpiCard label="Uncertainty band" value={`±${fmt(((d.band.oil.p90 - d.band.oil.p10) / 2) * 100, 0)}`} unit="% oil (0–3 mo)" color={COLORS.gas}
          footer={<span className="text-[11px] text-faint">from the forecast validation; widened beyond 3 months</span>} />
      </div>
      <Card title="Field forecast" subtitle="Solid = VFM history · dashed = forecast · shaded = P10–P90 range from our forecast validation">
        <ForecastPlot d={d} />
      </Card>
      <Card title="Per well" subtitle="Ranked by oil loss — the wells that need an action first (see the what-if on the Daily rates tab)"
        bodyClass="overflow-x-auto" actions={<TrendingDown size={15} className="text-faint" />}>
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
            {head.map((h) => <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}</tr></thead>
          <tbody className="divide-y divide-line">
            {[...d.wells].sort((a, b) => (a.oil_end - a.oil_now) - (b.oil_end - b.oil_now)).map((w) => {
              const ch = (100 * (w.oil_end - w.oil_now)) / w.oil_now
              return (
                <tr key={w.well}>
                  <td className="px-4 py-2.5 font-semibold text-ink">{w.well}</td>
                  <td className="num px-4 py-2.5 text-ink">{fmt(w.oil_now)}</td>
                  <td className="num px-4 py-2.5 text-muted">{fmt(w.oil_3m)}</td>
                  <td className="num px-4 py-2.5 text-ink">{fmt(w.oil_end)}</td>
                  <td className="px-4 py-2.5"><Pill tone={ch < -30 ? 'bad' : ch < -10 ? 'warn' : 'good'}>{ch.toFixed(0)} %</Pill></td>
                  <td className="num px-4 py-2.5 text-muted">{compact(w.cum_oil)}</td>
                  <td className="num px-4 py-2.5 text-muted">{fmt(w.wc_now, 1)} → {fmt(w.wc_end, 1)} %</td>
                  <td className="num px-4 py-2.5 text-muted">{fmt(w.p_res_slope_month, 1)} psi/mo</td>
                  <td className="num px-4 py-2.5 text-muted">{fmt(w.pwf_margin_to_pb)} psi</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="px-4 pb-4 pt-3 text-[11px] leading-relaxed text-faint">
          Method: reservoir pressure and water cut follow their straight-line trend of the last 90 days; oil = J (P<sub>res</sub> − Pwf)
          with Pwf held at its 30-day mean; water from the water cut, gas = oil × R<sub>s</sub>. A falling rate with a constant Pwf means the
          drawdown shrinks as the reservoir depletes — lowering Pwf (more lift gas, bigger choke) or stimulating recovers it.
        </p>
      </Card>
    </div>
  )
}
