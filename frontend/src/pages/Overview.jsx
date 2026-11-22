// Field overview: KPIs, field production, model health, attention list, well table.
import { AlertTriangle, Gauge, Info, ShieldCheck, TrendingDown } from 'lucide-react'
import { Link } from 'react-router-dom'
import Plot from '../components/Plot'
import { Card, Delta, ErrorBox, KpiCard, Pill, Skeleton, Sparkline } from '../components/ui'
import { COLORS, compact, fmt, useApi } from '../lib'

function FieldChart({ daily, shutIns }) {
  const x = daily.map((d) => d.date)
  const data = [
    { x, y: daily.map((d) => d.oil), name: 'Oil (STB/d)', type: 'scatter', mode: 'lines', stackgroup: 'liq',
      line: { color: COLORS.oil, width: 1.5 }, fillcolor: 'rgba(52,199,123,0.22)' },
    { x, y: daily.map((d) => d.water), name: 'Water (STB/d)', type: 'scatter', mode: 'lines', stackgroup: 'liq',
      line: { color: COLORS.water, width: 1.5 }, fillcolor: 'rgba(76,141,255,0.16)' },
    { x, y: daily.map((d) => d.gas), name: 'Gas (Mscf/d)', type: 'scatter', mode: 'lines', yaxis: 'y2',
      line: { color: COLORS.gas, width: 1.5, dash: 'dot' } },
  ]
  // shut-ins: faint vertical lines + one hoverable diamond marker each (labels would overlap)
  const shapes = shutIns.map((s) => ({
    type: 'line', x0: s.t_shut, x1: s.t_shut, yref: 'paper', y0: 0, y1: 1,
    line: { color: 'rgba(245,181,68,0.3)', width: 1, dash: 'dot' },
  }))
  const top = Math.max(...daily.map((d) => d.oil + d.water)) * 1.06
  data.push({
    x: shutIns.map((s) => s.t_shut), y: shutIns.map(() => top), name: 'Shut-in', type: 'scatter',
    mode: 'markers', marker: { symbol: 'diamond', size: 9, color: COLORS.accent, line: { width: 0 } },
    text: shutIns.map((s) => `${s.well} · ${s.duration_hr.toFixed(1)} h · ${fmt(s.deferred_oil_stb)} STB deferred`),
    hovertemplate: '%{text}<extra></extra>',
  })
  const layout = {
    yaxis: { title: { text: 'Liquid (STB/d)' }, rangemode: 'tozero' },
    yaxis2: { title: { text: 'Gas (Mscf/d)' }, overlaying: 'y', side: 'right', showgrid: false,
      rangemode: 'tozero', tickmode: 'auto', nticks: 6 },
    legend: { orientation: 'h', x: 0, y: -0.1, yanchor: 'top' },
    margin: { l: 60, r: 60, t: 12, b: 64 }, shapes,
  }
  return <Plot data={data} layout={layout} height="100%" className="min-h-[360px]" />
}

function ModelHealth({ model }) {
  const rows = [
    { label: 'Oil', key: 'oil', color: COLORS.oil },
    { label: 'Water', key: 'water', color: COLORS.water },
    { label: 'Gas', key: 'gas', color: COLORS.gas },
  ]
  return (
    <Card title="VFM model accuracy" subtitle="Volume-weighted error (WAPE) vs measured well tests"
      actions={<Pill tone="good"><ShieldCheck size={12} />Validated</Pill>}>
      <div className="mb-2 flex justify-end gap-4 text-[10px] uppercase tracking-[0.1em] text-faint">
        <span>between tests</span><span>forecast 0–3 mo</span>
      </div>
      <div className="space-y-3.5">
        {rows.map((r) => {
          const v = model[`${r.key}_wape`], f = model[`forward_${r.key}_wape`]
          return (
            <div key={r.label}>
              <div className="mb-1.5 flex justify-between text-xs">
                <span className="text-muted">{r.label}</span>
                <span className="num flex gap-6"><span className="text-ink">{v.toFixed(1)}%</span><span className="w-12 text-right text-muted">{f.toFixed(1)}%</span></span>
              </div>
              <div className="relative h-1.5 overflow-hidden rounded-full bg-white/5">
                <div className="absolute h-full rounded-full opacity-30" style={{ width: `${Math.min(f / 10, 1) * 100}%`, background: r.color }} />
                <div className="absolute h-full rounded-full" style={{ width: `${Math.min(v / 10, 1) * 100}%`, background: r.color }} />
              </div>
            </div>
          )
        })}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-4 text-xs">
        <div><div className="num text-lg text-ink">{model.n_tests}</div><div className="text-faint">well tests</div></div>
        <div><div className="num text-lg text-ink">{model.n_experiments}</div><div className="text-faint">model runs compared</div></div>
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-faint">
        Physics-informed: q<sub>oil</sub> = J·(P<sub>res</sub> − P<sub>wf</sub>), gas = oil × R<sub>s</sub>, water blended with a Gaussian process.
        Forecast = trained only on earlier tests, scored on the next 3 months.
      </p>
    </Card>
  )
}

const LEVEL = {
  high: { icon: AlertTriangle, tone: 'bad', color: 'text-bad' },
  medium: { icon: TrendingDown, tone: 'warn', color: 'text-warn' },
  info: { icon: Info, tone: 'info', color: 'text-water' },
}

function Attention({ items }) {
  return (
    <Card title="Needs attention" subtitle={`${items.length} findings from VFM and well-test analysis`} bodyClass="p-2">
      <ul className="divide-y divide-line">
        {items.map((a, i) => {
          const { icon: Icon, tone, color } = LEVEL[a.level]
          return (
            <li key={i} className="flex gap-3 rounded-lg px-3 py-3 hover:bg-white/[0.02]">
              <Icon size={16} className={`mt-0.5 shrink-0 ${color}`} />
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium text-ink">{a.title}</span>
                  <Pill tone={tone}>{a.well}</Pill>
                </div>
                <p className="mt-0.5 text-xs text-muted">{a.text}</p>
              </div>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}

function skinTone(skin) {
  if (skin === null) return 'neutral'
  if (skin > 3) return 'bad'
  if (skin > 1) return 'warn'
  return 'good'
}

function WellTable({ wells }) {
  const head = ['Well', 'Status', 'Oil', 'Water', 'Gas', 'Water cut', 'GOR', 'Oil 30 d', 'P res', 'Depletion', 'k', 'Skin', 'Oil trend']
  return (
    <Card title="Well performance" subtitle="Latest day (VFM) · reservoir pressure from the inflow model · k and skin from build-up tests"
      bodyClass="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
            {head.map((h) => <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {wells.map((w) => (
            <tr key={w.well} className="transition hover:bg-white/[0.025]">
              <td className="px-4 py-3 font-semibold text-ink">{w.well}</td>
              <td className="px-4 py-3"><Pill tone={w.status === 'Flowing' ? 'good' : 'warn'} dot>{w.status}</Pill></td>
              <td className="num px-4 py-3 text-ink">{fmt(w.oil)}<span className="ml-1 text-[10px] text-faint">STB/d</span></td>
              <td className="num px-4 py-3 text-muted">{fmt(w.water)}</td>
              <td className="num px-4 py-3 text-muted">{fmt(w.gas)}</td>
              <td className="num px-4 py-3 text-muted">{fmt(w.water_cut, 1)}%</td>
              <td className="num px-4 py-3 text-muted">{fmt(w.gor)}</td>
              <td className="px-4 py-3"><Delta value={w.oil_change_pct} suffix="" /></td>
              <td className="num px-4 py-3 text-muted">{fmt(w.p_res)}<span className="ml-1 text-[10px] text-faint">psia</span></td>
              <td className="num px-4 py-3 text-muted">−{fmt(w.p_res_decline_psi_month, 1)}<span className="ml-1 text-[10px] text-faint">psi/mo</span></td>
              <td className="num px-4 py-3 text-muted">{w.k_md ? fmt(w.k_md, 0) : '—'}<span className="ml-1 text-[10px] text-faint">{w.k_md ? 'md' : ''}</span></td>
              <td className="px-4 py-3">{w.skin !== null ? <Pill tone={skinTone(w.skin)}>{w.skin.toFixed(2)}</Pill> : <span className="text-faint">—</span>}</td>
              <td className="px-4 py-2"><Sparkline values={w.spark} color={COLORS.oil} width={130} height={28} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function LoadingState() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-5">{[...Array(5)].map((_, i) => <Skeleton key={i} className="h-32" />)}</div>
      <div className="grid grid-cols-12 gap-5"><Skeleton className="col-span-8 h-96" /><Skeleton className="col-span-4 h-96" /></div>
      <Skeleton className="h-80" />
    </div>
  )
}

export default function Overview() {
  const { data, error, loading } = useApi('/api/overview')
  if (loading) return <LoadingState />
  if (error) return <ErrorBox error={error} />
  const { kpis, field_daily: daily, wells, attention, model, shut_ins: shutIns } = data
  const last90 = daily.slice(-90)
  const wc = last90.map((d) => (100 * d.water) / (d.oil + d.water))

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between text-xs text-muted">
        <span>Snapshot for <span className="num text-ink">{kpis.date}</span> · rates are VFM estimates (STB/d, Mscf/d)</span>
        <span className="flex items-center gap-2"><Gauge size={14} className="text-accent" />
          Cumulative oil in period <span className="num text-ink">{compact(kpis.cum_oil_period)}</span> STB</span>
      </div>

      <div className="grid grid-cols-2 gap-4 xl:grid-cols-5">
        <KpiCard label="Oil" value={fmt(kpis.oil)} unit="STB/d" color={COLORS.oil}
          delta={kpis.oil_change_pct} goodWhenUp spark={last90.map((d) => d.oil)} />
        <KpiCard label="Water" value={fmt(kpis.water)} unit="STB/d" color={COLORS.water}
          delta={kpis.water_change_pct} goodWhenUp={false} spark={last90.map((d) => d.water)} />
        <KpiCard label="Gas" value={fmt(kpis.gas)} unit="Mscf/d" color={COLORS.gas}
          delta={kpis.gas_change_pct} goodWhenUp spark={last90.map((d) => d.gas)}
          hint="Formation gas (excludes injected lift gas)" />
        <KpiCard label="Water cut" value={fmt(kpis.water_cut, 1)} unit="%" color="#7dd3fc" spark={wc}
          footer={<span className="text-[11px] text-faint">Field average, latest day</span>} />
        <KpiCard label="Wells online" value={`${kpis.wells_flowing}/${kpis.wells_total}`} color={COLORS.accent}
          footer={<span className="text-[11px] text-faint">
            <span className="num text-muted">{kpis.shut_in_events}</span> shut-ins ({kpis.buildup_tests} build-up tests) ·{' '}
            <span className="num text-muted">{compact(kpis.deferred_oil_stb)}</span> STB deferred</span>} />
      </div>

      <div className="grid grid-cols-12 gap-5">
        <Card className="col-span-12 flex flex-col xl:col-span-8" bodyClass="flex-1 p-5" title="Field production"
          subtitle="Daily VFM estimate, all wells · ◆ = shut-ins analysed in the well-test module (hover for details)"
          actions={<Link to="/vfm" className="text-xs text-accent hover:underline">Open VFM →</Link>}>
          <FieldChart daily={daily} shutIns={shutIns} />
        </Card>
        <div className="col-span-12 space-y-5 xl:col-span-4">
          <ModelHealth model={model} />
          <Attention items={attention} />
        </div>
      </div>

      <WellTable wells={wells} />
    </div>
  )
}
