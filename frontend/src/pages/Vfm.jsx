// Virtual flow meter: model card, daily rates per well vs measured tests, reservoir pressure, all-well table.
import { useMemo, useState } from 'react'
import { Activity, Download, FlaskConical, ShieldCheck, TrendingDown } from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, KpiCard, Pill, Skeleton, Tabs } from '../components/ui'
import { COLORS, fmt, useApi } from '../lib'
import VfmForecast from './VfmForecast'
import VfmValidation from './VfmValidation'
import WhatIf from './WhatIf'

const FLUIDS = [
  { key: 'oil', label: 'Oil', unit: 'STB/d', color: COLORS.oil },
  { key: 'water', label: 'Water', unit: 'STB/d', color: COLORS.water },
  { key: 'gas', label: 'Gas', unit: 'Mscf/d', color: COLORS.gas },
]
const sub = (a, b) => (a == null || b == null ? null : a - b)

// ---------- model card ----------
function ModelCard({ ov }) {
  const v = ov.validation
  const eqs = [
    { f: 'Oil', c: COLORS.oil, e: <>q<sub>o</sub> = J·(P<sub>res</sub>(t) − P<sub>wf</sub>)</>,
      d: 'J from Darcy with our own build-up k & skin (or fitted from tests); P_res local linear trend per well' },
    { f: 'Gas', c: COLORS.gas, e: <>q<sub>g</sub> = q<sub>o</sub>·R<sub>s</sub></>,
      d: 'reservoir above the bubble point → produced GOR = solution GOR (lift gas excluded)' },
    { f: 'Water', c: COLORS.water, e: <>0.5 × physics WC(t) + 0.5 × GP</>,
      d: 'Gaussian process on separator-safe inputs only (no flowline / choke pressures)' },
  ]
  return (
    <Card title="Model" subtitle={`${ov.model.name} · trained on well tests up to ${ov.model.trained_until.slice(0, 10)}`}
      actions={<Pill tone="good"><ShieldCheck size={12} />Stored model (.pkl)</Pill>}>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="space-y-3 lg:col-span-3">
          {eqs.map((x) => (
            <div key={x.f} className="flex gap-4 rounded-xl border border-line bg-white/[0.015] px-4 py-3">
              <span className="w-12 shrink-0 text-sm font-semibold" style={{ color: x.c }}>{x.f}</span>
              <div>
                <div className="num text-[15px] text-ink">{x.e}</div>
                <div className="mt-0.5 text-xs text-muted">{x.d}</div>
              </div>
            </div>
          ))}
          <div className="text-xs text-faint">Daily rate = flowing rate × fraction of the day on stream; fully shut-in day = 0.</div>
        </div>
        <div className="lg:col-span-2">
          <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Validation (official metrics, WAPE %)</div>
          <table className="w-full text-[13px]">
            <thead><tr className="text-left text-[11px] text-faint"><th className="py-1.5 font-medium">Fluid</th>
              <th className="font-medium">Between tests</th><th className="font-medium">Forecast 0–3 mo</th></tr></thead>
            <tbody>{FLUIDS.map((f) => (
              <tr key={f.key} className="border-t border-line">
                <td className="py-2 font-medium" style={{ color: f.color }}>{f.label}</td>
                <td className="num text-ink">{v[`${f.key}_wape`].toFixed(2)}</td>
                <td className="num text-ink">{v[`forward_${f.key}_wape`].toFixed(2)}</td>
              </tr>))}
            </tbody>
          </table>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Between tests = leave-one-test-out. Forecast = trained only on tests before a cutoff, scored on the next
            3 months. {v.n_experiments} model runs compared on the same folds.
          </p>
          {ov.accuracy_on_dataset_tests && (
            <div className="mt-3 rounded-lg border border-line p-3 text-xs">
              <div className="mb-1 text-faint">On this dataset's {ov.accuracy_on_dataset_tests.n_tests} well tests
                {ov.accuracy_on_dataset_tests.n_in_sample > 0 ? ' (in-sample: used for training)' : ' (blind)'}</div>
              <div className="num flex gap-4 text-ink">{FLUIDS.map((f) => (
                <span key={f.key}><span style={{ color: f.color }}>{f.label}</span> {ov.accuracy_on_dataset_tests[f.key].WAPE.toFixed(1)}%</span>))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

// ---------- well charts ----------
function RatesPlot({ d }) {
  const traces = []
  const BAND = { oil: 'rgba(52,199,123,0.16)', water: 'rgba(76,141,255,0.16)', gas: 'rgba(240,96,93,0.16)' }
  FLUIDS.forEach((f, i) => {
    const ax = i ? `${i + 1}` : ''
    if (d.band) {    // P10-P90 range: rate x (measured / predicted) quantiles of the forecast validation
      traces.push({ x: d.daily.date, y: d.daily[f.key].map((v) => v * d.band[f.key].p10), showlegend: false, legendgroup: 'band',
        type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', xaxis: 'x', yaxis: `y${ax}` })
      traces.push({ x: d.daily.date, y: d.daily[f.key].map((v) => v * d.band[f.key].p90), name: 'P10–P90 range', showlegend: !i,
        legendgroup: 'band', type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: BAND[f.key], hoverinfo: 'skip',
        xaxis: 'x', yaxis: `y${ax}` })
    }
    traces.push({ x: d.daily.date, y: d.daily[f.key], name: `${f.label} (VFM)`, type: 'scatter', mode: 'lines',
      line: { color: f.color, width: 1.6 }, xaxis: 'x', yaxis: `y${ax}` })
    if (d.tests) {
      traces.push({ x: d.tests.time, y: d.tests[f.key], name: i ? undefined : 'Well test (measured)', showlegend: !i,
        type: 'scatter', mode: 'markers', marker: { color: '#ffffff', size: 8, line: { color: f.color, width: 2 } },
        xaxis: 'x', yaxis: `y${ax}`, legendgroup: 'tests' })
    }
  })
  const layout = {
    grid: { rows: 3, columns: 1, pattern: 'coupled', roworder: 'top to bottom' },
    yaxis: { title: { text: 'Oil (STB/d)' } }, yaxis2: { title: { text: 'Water (STB/d)' } }, yaxis3: { title: { text: 'Gas (Mscf/d)' } },
    margin: { l: 64, r: 20, t: 40, b: 36 },
  }
  return <Plot data={traces} layout={layout} height={560} />
}

function PressurePlot({ d }) {
  const traces = [
    { x: d.daily.date, y: d.daily.p_res, name: 'Reservoir pressure P_res (model)', type: 'scatter', mode: 'lines',
      line: { color: COLORS.accent, width: 2 } },
    { x: d.daily.date, y: d.daily.pwf, name: 'Flowing pressure Pwf (gauge)', type: 'scatter', mode: 'lines',
      line: { color: '#c084fc', width: 1.2 } },
  ]
  if (d.pres_obs) {
    traces.push({ x: d.pres_obs.date, y: d.pres_obs.p_res, name: 'P_res implied by tests = Pwf + q/J', type: 'scatter',
      mode: 'markers', marker: { color: COLORS.accent, size: 8, symbol: 'diamond-open', line: { width: 2 } } })
  }
  return <Plot data={traces} layout={{ yaxis: { title: { text: 'Pressure (psia)' } }, margin: { l: 64, r: 20, t: 40, b: 36 } }} height={300} />
}

function TestsTable({ t }) {
  if (!t) return <div className="text-sm text-muted">No well tests in this dataset — rates are predictions only.</div>
  const err = (p, m) => (m ? (100 * (p - m)) / m : null)
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12.5px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
          <th className="px-2 py-2 font-medium">Test</th>
          {FLUIDS.map((f) => <th key={f.key} className="px-2 py-2 font-medium" colSpan={2}>{f.label} meas. / error</th>)}
        </tr></thead>
        <tbody className="divide-y divide-line">
          {t.time.map((time, i) => (
            <tr key={time}>
              <td className="num whitespace-nowrap px-2 py-1.5 text-muted">{time.slice(0, 10)}</td>
              {FLUIDS.map((f) => {
                const e = err(t[`pred_${f.key}`][i], t[f.key][i])
                return [
                  <td key={f.key} className="num px-2 py-1.5 text-ink">{fmt(t[f.key][i])}</td>,
                  <td key={f.key + 'e'} className={`num px-2 py-1.5 ${Math.abs(e) > 5 ? 'text-warn' : 'text-muted'}`}>{e > 0 ? '+' : ''}{e.toFixed(1)}%</td>,
                ]
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Row({ label, value, unit, hint }) {
  return (
    <div className="flex items-baseline justify-between py-2 text-[13px]" title={hint}>
      <span className="text-muted">{label}</span>
      <span className="num text-ink">{value}{unit && <span className="ml-1 text-[10px] text-faint">{unit}</span>}</span>
    </div>
  )
}

function WellPanel({ s }) {
  return (
    <Card title="Well calibration" subtitle="What the model knows about this well">
      <div className="divide-y divide-line">
        <Row label="Productivity index J" value={fmt(s.J, 2)} unit="STB/d/psi" hint="q_oil = J · (P_res − Pwf)" />
        <Row label="J source" value={s.J_source} />
        <Row label="Permeability k (build-up, Jan–Sep)" hint="Value the stored model was calibrated with (VFM-PTA iteration, q = oil rate of the day before the shut-in). The Well Test Analysis page reports the deliverable values (q = 7-day mean rate), so they differ slightly." value={s.k_md != null ? fmt(s.k_md, 1) : '—'} unit={s.k_md != null ? 'md' : ''} />
        <Row label="Skin (build-up, Jan–Sep)" hint="Value the stored model was calibrated with (VFM-PTA iteration, q = oil rate of the day before the shut-in). The Well Test Analysis page reports the deliverable values (q = 7-day mean rate), so they differ slightly." value={s.skin != null ? s.skin.toFixed(2) : '—'} />
        <Row label="Solution GOR Rs" value={fmt(s.rs_scf_stb)} unit="scf/STB" />
        <Row label="Bubble point Pb" value={fmt(s.pb_psia)} unit="psia" />
        <Row label="P_res − Pb margin" value={fmt(sub(s.p_res, s.pb_psia))} unit="psi" />
        <Row label="P_res trend (90 d)" value={fmt(s.p_res_slope_psi_month, 1)} unit="psi/month" />
        <Row label="Water-cut trend (90 d)" value={s.wc_slope_pp_month != null ? `${s.wc_slope_pp_month > 0 ? '+' : ''}${s.wc_slope_pp_month.toFixed(2)}` : '—'} unit="pp/month" />
        <Row label="Cumulative oil (period)" value={fmt(s.cum_oil)} unit="STB" />
        <Row label="Oil error on tests (WAPE)" value={s.oil_wape_tests != null ? s.oil_wape_tests.toFixed(2) : '—'} unit={s.oil_wape_tests != null ? `% · ${s.n_tests} tests` : ''} />
      </div>
    </Card>
  )
}

function AllWells({ wells, selected, onSelect }) {
  const head = ['Well', 'Oil', 'Water', 'Gas', 'WC %', 'P_res', 'ΔP_res / mo', 'J', 'J source', 'Cum oil', 'Oil WAPE']
  return (
    <Card title="All wells · latest day" subtitle="Click a row to open the well" bodyClass="overflow-x-auto"
      actions={<div className="flex gap-2"><a href="/api/vfm/export.csv" className="flex items-center gap-2 rounded-lg border border-line px-3 py-1.5 text-xs font-semibold text-ink transition hover:border-line-strong"><Download size={14} />CSV</a><a href="/api/vfm/export.xlsx" className="flex items-center gap-2 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent
        transition hover:brightness-110"><Download size={14} />Export VFM Excel</a></div>}>
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
          {head.map((h) => <th key={h} className="whitespace-nowrap px-4 py-3 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {wells.map((w) => (
            <tr key={w.well} onClick={() => onSelect(w.well)}
              className={`cursor-pointer transition ${w.well === selected ? 'bg-accent/[0.05]' : 'hover:bg-white/[0.025]'}`}>
              <td className="px-4 py-2.5 font-semibold text-ink">{w.well}</td>
              <td className="num px-4 py-2.5 text-ink">{fmt(w.oil)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.water)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.gas)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.water_cut, 1)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.p_res)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.p_res_slope_psi_month, 1)}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.J, 2)}</td>
              <td className="px-4 py-2.5 text-xs text-muted">{w.J_source}</td>
              <td className="num px-4 py-2.5 text-muted">{fmt(w.cum_oil)}</td>
              <td className="num px-4 py-2.5 text-muted">{w.oil_wape_tests != null ? `${w.oil_wape_tests.toFixed(1)}%` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

// ---------- page ----------
export default function Vfm() {
  const ov = useApi('/api/vfm/overview')
  const [well, setWell] = useState(null)
  const current = well || ov.data?.wells[0]?.well
  const d = useApi(current ? `/api/vfm/well/${current}` : null)
  const spark = useMemo(() => d.data?.daily.oil.slice(-90), [d.data])
  const [tab, setTab] = useState(() => new URLSearchParams(window.location.search).get('tab') || 'rates')   // ?tab=forecast / validation
  const tabs = [{ id: 'rates', label: 'Daily rates', icon: Activity }, { id: 'forecast', label: 'Forecast', icon: TrendingDown },
    { id: 'validation', label: 'Validation', icon: ShieldCheck }]

  if (ov.error || d.error) return <ErrorBox error={ov.error || d.error} />
  if (!ov.data) return <div className="space-y-5"><Skeleton className="h-72" /><Skeleton className="h-96" /></div>
  const s = d.data?.summary
  if (tab !== 'rates') {
    return (
      <div className="space-y-5">
        <Tabs tabs={tabs} value={tab} onChange={setTab} />
        {tab === 'forecast' ? <VfmForecast /> : <VfmValidation />}
      </div>
    )
  }
  return (
    <div className="space-y-5">
      <Tabs tabs={tabs} value={tab} onChange={setTab} />
      <ModelCard ov={ov.data} />

      <div className="flex flex-wrap gap-2">
        {ov.data.wells.map((w) => (
          <button key={w.well} onClick={() => setWell(w.well)}
            className={`rounded-xl border px-3.5 py-2 text-left transition ${w.well === current
              ? 'border-accent/50 bg-accent/[0.07] shadow-[0_0_0_1px_rgba(245,181,68,0.25)]' : 'border-line bg-panel/70 hover:border-line-strong'}`}>
            <div className="text-[13px] font-semibold text-ink">{w.well}</div>
            <div className="num text-[11px] text-muted">{fmt(w.oil)} STB/d</div>
          </button>
        ))}
      </div>

      {!d.data || d.loading ? <Skeleton className="h-[620px]" /> : (<>
        <div className="grid grid-cols-2 gap-4 xl:grid-cols-6">
          <KpiCard label="Oil" value={fmt(s.oil)} unit="STB/d" color={COLORS.oil} spark={spark}
            footer={<span className="text-[11px] text-faint">30-day mean {fmt(s.oil_30d)}</span>} />
          <KpiCard label="Water" value={fmt(s.water)} unit="STB/d" color={COLORS.water} footer={<span className="text-[11px] text-faint">latest flowing day</span>} />
          <KpiCard label="Gas" value={fmt(s.gas)} unit="Mscf/d" color={COLORS.gas} footer={<span className="text-[11px] text-faint">formation gas</span>} />
          <KpiCard label="Water cut" value={fmt(s.water_cut, 1)} unit="%" color="#7dd3fc"
            footer={<span className="text-[11px] text-faint">{s.wc_slope_pp_month != null ? `${s.wc_slope_pp_month > 0 ? '+' : ''}${s.wc_slope_pp_month.toFixed(2)} pp/month` : ''}</span>} />
          <KpiCard label="Reservoir P" value={fmt(s.p_res)} unit="psia" color={COLORS.accent}
            footer={<span className="text-[11px] text-faint">{fmt(s.p_res_slope_psi_month, 1)} psi/month</span>} />
          <KpiCard label="Drawdown" value={fmt(sub(s.p_res, s.pwf))} unit="psi" color="#c084fc"
            footer={<span className="text-[11px] text-faint">P_res − Pwf</span>} />
        </div>

        <div className="grid grid-cols-12 gap-5">
          <div className="col-span-12 space-y-5 xl:col-span-8">
            <Card title={`${current} · daily rates`} subtitle="Line = VFM every day · shaded = P10–P90 range from the forecast validation · circles = measured well tests">
              <RatesPlot d={d.data} />
            </Card>
            <Card title="Reservoir and flowing pressure" subtitle="Drawdown P_res − Pwf drives the oil rate; diamonds = reservoir pressure implied by each training test">
              <PressurePlot d={d.data} />
            </Card>
          </div>
          <div className="col-span-12 space-y-5 xl:col-span-4">
            <WellPanel s={s} />
            <Card title="Well tests vs model" subtitle="Measured rate and model error at each test"
              actions={<FlaskConical size={15} className="text-faint" />}>
              <TestsTable t={d.data.tests} />
            </Card>
          </div>
        </div>
        <WhatIf well={current} />
      </>)}

      <AllWells wells={ov.data.wells} selected={current} onSelect={setWell} />
    </div>
  )
}
