// Insights & actions: where oil is lost or can be gained, with its value at a chosen oil price.
import { useEffect, useState } from 'react'
import { AlertTriangle, CalendarClock, DollarSign, Flame, Gauge, Wrench } from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, KpiCard, Pill, Skeleton } from '../components/ui'
import { COLORS, compact, fmt, useApi } from '../lib'

const usd = (v) => `$${compact(v)}`

function OilPrice({ value, onChange }) {
  const [text, setText] = useState(String(value))
  useEffect(() => {                                   // apply after the user stops typing
    const t = setTimeout(() => { const v = Number(text); if (v > 0 && v < 1000 && v !== value) onChange(v) }, 500)
    return () => clearTimeout(t)
  }, [text]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <label className="flex items-center gap-2 rounded-xl border border-line bg-panel/70 px-3 py-2 text-xs text-muted">
      <DollarSign size={14} className="text-accent" />Oil price
      <input value={text} onChange={(e) => setText(e.target.value)} inputMode="decimal"
        className="num w-16 rounded-md border border-line bg-panel-2 px-2 py-1 text-right text-ink outline-none focus:border-accent/50" />
      $/bbl
    </label>
  )
}

function Stimulation({ items, price }) {
  return (
    <Card title="Stimulation candidates" subtitle="Removing the skin lifts the rate to q / FE at the same drawdown (FE = flow efficiency from the build-up)"
      actions={<Wrench size={15} className="text-faint" />}>
      {items.length === 0 ? <div className="text-sm text-muted">No damaged well found (all skins ≤ 0).</div> : (
        <div className="space-y-3">
          {items.map((s) => (
            <div key={s.well + s.t_shut} className={`rounded-xl border p-4 ${s.planned ? 'border-good/25 bg-good/[0.04]' : 'border-line bg-white/[0.015]'}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="text-base font-semibold text-ink">{s.well}</span>
                  <Pill tone={s.skin > 3 ? 'bad' : 'warn'}>skin {s.skin.toFixed(2)}</Pill>
                  <Pill tone={s.planned ? 'good' : 'neutral'}>{s.confidence}</Pill>
                </div>
                <div className="text-right">
                  <span className="num text-xl text-good">+{fmt(s.upside_stbd)}</span><span className="ml-1 text-xs text-muted">STB/d</span>
                  <div className="num text-xs text-muted">≈ {usd(s.value_usd_year)} / year at ${price}/bbl</div>
                </div>
              </div>
              <div className="mt-3">
                <div className="flex justify-between text-[11px] text-faint"><span>today {fmt(s.q_oil_stbd)} STB/d</span><span>without skin {fmt(s.q_oil_stbd + s.upside_stbd)} STB/d</span></div>
                <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/5">
                  <div className="h-full rounded-full bg-good" style={{ width: `${100 * s.flow_efficiency}%` }} />
                </div>
                <div className="mt-1.5 text-xs text-muted">{fmt(s.dP_skin_psi)} psi of drawdown lost in the damaged zone · flow efficiency {s.flow_efficiency.toFixed(2)} · build-up {s.t_shut}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

function LiftGas({ items }) {
  if (!items.length) return null
  const rev = [...items].reverse()
  const worst = items[items.length - 1]
  const data = [{
    type: 'bar', orientation: 'h', y: rev.map((x) => x.well), x: rev.map((x) => x.oil_per_mscf),
    marker: { color: rev.map((x) => (x === worst ? COLORS.gas : x.vs_median_pct > 0 ? COLORS.oil : '#4a5566')) },
    text: rev.map((x) => x.oil_per_mscf.toFixed(2)), textposition: 'outside', cliponaxis: false,
    customdata: rev.map((x) => [x.oil_stbd, x.lift_gas_mscfd]),
    hovertemplate: '%{y}: %{x:.2f} STB oil per Mscf lift gas<br>oil %{customdata[0]:.0f} STB/d · lift gas %{customdata[1]:.0f} Mscf/d<extra></extra>',
  }]
  return (
    <Card title="Gas-lift efficiency" subtitle="Oil per Mscf of injected lift gas, last 30 days" actions={<Flame size={15} className="text-faint" />}>
      <Plot data={data} layout={{ hovermode: 'closest', margin: { l: 56, r: 40, t: 10, b: 36 }, xaxis: { title: { text: 'STB oil per Mscf lift gas' } }, showlegend: false }} height={300} />
      <div className="mt-2 rounded-lg border border-gas/25 bg-gas/[0.05] p-3 text-xs text-ink">
        <b>{worst.well}</b> takes {fmt(worst.lift_gas_mscfd)} Mscf/d of lift gas for {fmt(worst.oil_stbd)} STB/d
        ({worst.oil_per_mscf.toFixed(2)} STB/Mscf, {fmt(worst.vs_median_pct)} % vs field median) → run a lift-gas step test;
        gas moved to a more responsive well can earn more.
      </div>
    </Card>
  )
}

function Deferred({ items, price }) {
  const data = [{
    type: 'bar', x: items.map((d) => `${d.well} ${d.t_shut.slice(5, 10)}`), y: items.map((d) => d.deferred_oil_stb),
    marker: { color: items.map((d) => (d.event_type === 'planned build-up test' ? COLORS.accent : '#4a5566')) },
    text: items.map((d) => fmt(d.deferred_oil_stb)), textposition: 'outside', cliponaxis: false,
    customdata: items.map((d) => [d.duration_hr, d.value_usd, d.event_type]),
    hovertemplate: '%{x}: %{y:,.0f} STB · %{customdata[0]:.1f} h · $%{customdata[1]:,.0f}<br>%{customdata[2]}<extra></extra>',
  }]
  const total = items.reduce((a, d) => a + d.deferred_oil_stb, 0)
  return (
    <Card title="Deferred production" subtitle={`Oil not produced while shut in · amber = planned build-up test · total ${fmt(total)} STB ≈ ${usd(total * price)}`}
      actions={<CalendarClock size={15} className="text-faint" />}>
      <Plot data={data} layout={{ hovermode: 'closest', margin: { l: 56, r: 20, t: 20, b: 50 }, yaxis: { title: { text: 'STB' } }, showlegend: false }} height={300} />
    </Card>
  )
}

function Health({ rows }) {
  const head = ['Well', 'P_res', 'Trend', 'P_res − Pb', 'Years to Pb', 'Pwf − Pb', 'Water cut', 'WC trend', 'Flags']
  return (
    <Card title="Reservoir health" bodyClass="overflow-x-auto" actions={<Gauge size={15} className="text-faint" />}
      subtitle="Depletion speed and distance to the bubble point (below Pb gas leaves the oil and the inflow model changes)">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
          {head.map((h) => <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((h) => (
            <tr key={h.well}>
              <td className="px-4 py-2.5 font-semibold text-ink">{h.well}</td>
              <td className="num px-4 py-2.5 text-ink">{fmt(h.p_res)}<span className="ml-1 text-[10px] text-faint">psia</span></td>
              <td className={`num px-4 py-2.5 ${h.p_res_slope_psi_month < -15 ? 'text-bad' : 'text-muted'}`}>{fmt(h.p_res_slope_psi_month, 1)}<span className="ml-1 text-[10px] text-faint">psi/mo</span></td>
              <td className="num px-4 py-2.5 text-muted">{fmt(h.p_res_minus_pb)}</td>
              <td className="num px-4 py-2.5 text-muted">{h.months_to_pb != null ? fmt(h.months_to_pb / 12, 1) : '—'}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(h.pwf_minus_pb)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(h.water_cut, 1)}%</td>
              <td className={`num px-4 py-2.5 ${h.wc_slope_pp_month > 1 ? 'text-warn' : 'text-muted'}`}>{h.wc_slope_pp_month != null ? `${h.wc_slope_pp_month > 0 ? '+' : ''}${h.wc_slope_pp_month.toFixed(2)}` : '—'}<span className="ml-1 text-[10px] text-faint">pp/mo</span></td>
              <td className="px-4 py-2.5"><div className="flex flex-wrap gap-1">{h.flags.length ? h.flags.map((f) => <Pill key={f} tone="warn">{f}</Pill>) : <Pill tone="good">ok</Pill>}</div></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function TestPlan({ gaps }) {
  if (!gaps.length) return null
  return (
    <Card title="Build-up test plan" subtitle="Wells without a planned build-up test: k and skin unknown, J fitted from well tests only — highest rate first"
      actions={<AlertTriangle size={15} className="text-faint" />}>
      <ol className="space-y-2">
        {gaps.map((g, i) => (
          <li key={g.well} className="flex items-center justify-between rounded-lg border border-line px-3 py-2 text-[13px]">
            <span className="flex items-center gap-3"><span className="num flex h-6 w-6 items-center justify-center rounded-full bg-accent/15 text-[11px] text-accent">{i + 1}</span>
              <span className="font-semibold text-ink">{g.well}</span><span className="text-xs text-muted">{g.note}</span></span>
            <span className="num text-muted">{fmt(g.oil_stbd)} STB/d</span>
          </li>
        ))}
      </ol>
      <p className="mt-3 text-xs text-muted">A 48–72 h build-up with a staged restart gives k, skin and boundaries — and anchors the VFM's reservoir-pressure trend.</p>
    </Card>
  )
}

export default function Insights() {
  const [price, setPrice] = useState(70)
  const { data, error } = useApi(`/api/insights?oil_price=${price}`)
  if (error) return <ErrorBox error={error} />
  if (!data) return <div className="space-y-5"><Skeleton className="h-28" /><Skeleton className="h-96" /></div>
  const k = data.kpis
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>Every number comes from the virtual flow meter or the build-up analysis of the active dataset.</span>
        <OilPrice value={price} onChange={setPrice} />
      </div>
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <KpiCard label="Stimulation upside" value={`+${fmt(k.stimulation_upside_stbd)}`} unit="STB/d" color={COLORS.oil}
          footer={<span className="text-[11px] text-faint">≈ {usd(k.stimulation_value_usd_year)} per year · planned tests only</span>} />
        <KpiCard label="Deferred oil" value={fmt(k.deferred_oil_stb)} unit="STB" color={COLORS.accent}
          footer={<span className="text-[11px] text-faint">≈ {usd(k.deferred_value_usd)} at ${price}/bbl</span>} />
        <KpiCard label="Wells flagged" value={k.wells_flagged} color={COLORS.gas}
          footer={<span className="text-[11px] text-faint">fast depletion · rising water cut · near Pb</span>} />
        <KpiCard label="Build-up tests to run" value={k.wells_to_test} color={COLORS.water}
          footer={<span className="text-[11px] text-faint">wells without k and skin</span>} />
      </div>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 xl:col-span-7"><Stimulation items={data.stimulation} price={price} /></div>
        <div className="col-span-12 xl:col-span-5"><LiftGas items={data.lift} /></div>
      </div>
      <Health rows={data.health} />
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 xl:col-span-7"><Deferred items={data.deferred} price={price} /></div>
        <div className="col-span-12 xl:col-span-5"><TestPlan gaps={data.gaps} /></div>
      </div>
    </div>
  )
}
