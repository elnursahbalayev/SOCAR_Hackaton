// VFM validation: every way the stored model was tested, with the evidence (not only one error number).
import { CheckCircle2, FlaskConical, GitCompare, Scale, ShieldCheck, Timer } from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, Pill, Skeleton } from '../components/ui'
import { COLORS, fmt, useApi } from '../lib'

const FLUIDS = [
  { key: 'oil', label: 'Oil', color: COLORS.oil },
  { key: 'water', label: 'Water', color: COLORS.water },
  { key: 'gas', label: 'Gas', color: COLORS.gas },
]
const FAMILY = { '0': ['Naive', '#5d6877'], A: ['Pure ML', '#94a3b8'], B: ['Physics', '#34c77b'], C1: ['Hybrid', '#c084fc'],
  C2: ['Hybrid', '#c084fc'], C3: ['Engineered targets', '#c084fc'], D: ['Sim-to-real', '#22d3ee'], E2: ['Ensemble', '#4c8dff'],
  E3: ['Ensemble', '#4c8dff'], F1: ['Ensemble', '#4c8dff'], F2: ['Ensemble', '#4c8dff'], rec120: ['Ensemble', '#4c8dff'] }

function Methods({ v }) {
  const items = [
    { icon: FlaskConical, t: 'Leave-one-test-out', d: `${v.n_tests} folds: each test predicted by a model that never saw it` },
    { icon: Timer, t: 'Time-forward', d: `${v.forecast.length} forecasts: trained before 1 Jun / 1 Jul, predicting 0–92 days ahead` },
    { icon: Scale, t: 'Paired significance tests', d: 'Wilcoxon, test by test: is an improvement real or noise?' },
    { icon: ShieldCheck, t: 'Normal-day check', d: 'labels are tests at the test separator; normal days checked against physics' },
    { icon: CheckCircle2, t: 'Physics consistency', d: 'build-up pressure vs VFM reservoir pressure; test GOR vs solution GOR' },
    { icon: GitCompare, t: '47 alternatives', d: 'same folds and metrics for ML, physics, hybrids, sim-to-real, ensembles' },
  ]
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {items.map(({ icon: Icon, t, d }) => (
        <div key={t} className="rise rounded-2xl border border-line bg-panel/80 p-4">
          <Icon size={17} className="text-accent" />
          <div className="mt-2 text-[13px] font-semibold text-ink">{t}</div>
          <div className="mt-1 text-[11px] leading-relaxed text-muted">{d}</div>
        </div>
      ))}
    </div>
  )
}

function MetricsTable({ m }) {
  const rows = [['between_tests', 'Between tests (leave-one-test-out)'], ['forecast', 'Forecast 0–92 days (time-forward)']]
  return (
    <Card title="Official metrics" subtitle="R², MAPE and WAPE (volume-weighted) per fluid — the organisers' scoring metrics" bodyClass="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
          <th className="px-4 py-2.5 font-medium">Scheme</th>
          {FLUIDS.map((f) => <th key={f.key} colSpan={3} className="px-4 py-2.5 font-medium" style={{ color: f.color }}>{f.label}: R² · MAPE · WAPE</th>)}
        </tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map(([k, label]) => (
            <tr key={k}>
              <td className="px-4 py-2.5 text-ink">{label}</td>
              {FLUIDS.map((f) => [
                <td key={f.key + 'r'} className="num px-2 py-2.5 text-muted">{m[k][f.key].R2.toFixed(3)}</td>,
                <td key={f.key + 'm'} className="num px-2 py-2.5 text-muted">{m[k][f.key].MAPE.toFixed(2)}%</td>,
                <td key={f.key + 'w'} className="num px-2 py-2.5 font-semibold text-ink">{m[k][f.key].WAPE.toFixed(2)}%</td>,
              ])}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function Parity({ v }) {
  const traces = []
  FLUIDS.forEach((f, i) => {
    const ax = i ? `${i + 1}` : ''
    const lo = v.loto.map((r) => r[f.key]), all = lo.concat(v.forecast.map((r) => r[f.key]))
    const mn = Math.min(...all) * 0.95, mx = Math.max(...all) * 1.05
    traces.push({ x: [mn, mx], y: [mn, mx], mode: 'lines', line: { color: COLORS.muted, width: 1, dash: 'dot' }, showlegend: false,
      hoverinfo: 'skip', xaxis: `x${ax}`, yaxis: `y${ax}`, type: 'scatter' })
    traces.push({ x: lo, y: v.loto.map((r) => r[f.key + '_pred']), name: 'Between tests (left out)', showlegend: !i, legendgroup: 'l',
      mode: 'markers', type: 'scatter', marker: { color: f.color, size: 7, opacity: 0.8 }, text: v.loto.map((r) => `${r.well} ${r.time.slice(0, 10)}`),
      hovertemplate: '%{text}<br>measured %{x:.0f} · predicted %{y:.0f}<extra></extra>', xaxis: `x${ax}`, yaxis: `y${ax}` })
    traces.push({ x: v.forecast.map((r) => r[f.key]), y: v.forecast.map((r) => r[f.key + '_pred']), name: 'Forecast (time-forward)',
      showlegend: !i, legendgroup: 'f', mode: 'markers', type: 'scatter', marker: { color: COLORS.ink, size: 7, symbol: 'x-thin', line: { width: 2, color: COLORS.ink } },
      text: v.forecast.map((r) => `${r.well} ${r.time.slice(0, 10)} · ${r.horizon_days} d ahead`),
      hovertemplate: '%{text}<br>measured %{x:.0f} · predicted %{y:.0f}<extra></extra>', xaxis: `x${ax}`, yaxis: `y${ax}` })
  })
  const layout = {
    grid: { rows: 1, columns: 3, pattern: 'independent' }, hovermode: 'closest', margin: { l: 56, r: 16, t: 44, b: 48 },
    xaxis: { title: { text: 'Measured oil (STB/d)' } }, yaxis: { title: { text: 'Predicted' } },
    xaxis2: { title: { text: 'Measured water (STB/d)' } }, xaxis3: { title: { text: 'Measured gas (Mscf/d)' } },
  }
  return (
    <Card title="Every test: measured vs predicted" subtitle="Each point predicted by a model that did not see it · dotted line = perfect prediction">
      <Plot data={traces} layout={layout} height={360} />
    </Card>
  )
}

function Horizon({ v }) {
  const data = FLUIDS.map((f) => ({ type: 'bar', name: f.label, x: v.horizon.map((h) => h.bucket), y: v.horizon.map((h) => h[f.key].WAPE),
    marker: { color: f.color }, text: v.horizon.map((h) => h[f.key].WAPE.toFixed(1)), textposition: 'outside', cliponaxis: false }))
  return (
    <Card title="Forecast error vs horizon" subtitle={`WAPE % by how far ahead the forecast is (${v.horizon.map((h) => `${h.bucket}: ${h.n}`).join(' · ')} tests)`}>
      <Plot data={data} layout={{ barmode: 'group', hovermode: 'closest', yaxis: { title: { text: 'WAPE %' } }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
    </Card>
  )
}

function PerWell({ v }) {
  const data = FLUIDS.map((f) => ({ type: 'bar', name: f.label, x: v.per_well.map((w) => w.well), y: v.per_well.map((w) => w[f.key].WAPE),
    marker: { color: f.color } }))
  return (
    <Card title="Error per well" subtitle="WAPE % between tests · gas is higher where the measured test GOR scatters most (see the GOR check below)">
      <Plot data={data} layout={{ barmode: 'group', hovermode: 'closest', yaxis: { title: { text: 'WAPE %' } }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
    </Card>
  )
}

function OverTime({ v }) {
  const data = FLUIDS.map((f) => ({ type: 'scatter', mode: 'markers', name: f.label, x: v.loto.map((r) => r.time),
    y: v.loto.map((r) => (100 * (r[f.key + '_pred'] - r[f.key])) / r[f.key]), marker: { color: f.color, size: 6, opacity: 0.8 },
    text: v.loto.map((r) => r.well), hovertemplate: '%{text} %{x|%d %b}: %{y:.1f} %<extra></extra>' }))
  return (
    <Card title="Error over time" subtitle="Prediction error % of each test (left out) by date — no drift, errors centred on zero">
      <Plot data={data} layout={{ hovermode: 'closest', yaxis: { title: { text: 'error %' }, zeroline: true }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
    </Card>
  )
}

function Significance({ rows }) {
  return (
    <Card title="Is the improvement real?" subtitle="Paired Wilcoxon test on the absolute % error, test by test · p < 0.05 = significant" bodyClass="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
          {['Comparison', 'Scheme', 'Fluid', 'Final better on', 'Mean error final / other', 'p-value'].map((h) => <th key={h} className="px-4 py-2.5 font-medium">{h}</th>)}
        </tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.comparison + r.scheme + r.fluid}>
              <td className="px-4 py-2 text-ink">{r.comparison}</td>
              <td className="px-4 py-2 text-muted">{r.scheme}</td>
              <td className="px-4 py-2 capitalize text-muted">{r.fluid}</td>
              <td className="num px-4 py-2 text-muted">{r.final_better_on} / {r.n}</td>
              <td className="num px-4 py-2 text-muted">{r.mean_error_final_pct.toFixed(2)} / {r.mean_error_other_pct.toFixed(2)} %</td>
              <td className="px-4 py-2"><Pill tone={r.p_value < 0.05 ? 'good' : 'neutral'}>{r.p_value < 0.001 ? '< 0.001' : r.p_value.toFixed(3)}</Pill></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function Steps({ rows }) {
  return (
    <Card title="What each component adds" subtitle="Final model built up step by step · WAPE % between tests / forecast" bodyClass="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
          <th className="px-4 py-2.5 font-medium">Step</th>{FLUIDS.map((f) => <th key={f.key} className="px-4 py-2.5 font-medium" style={{ color: f.color }}>{f.label}</th>)}
        </tr></thead>
        <tbody className="divide-y divide-line">
          {rows.map((r, i) => (
            <tr key={r.step} className={i === rows.length - 1 ? 'bg-accent/[0.06] font-semibold' : ''}>
              <td className="px-4 py-2 text-ink">{r.step}</td>
              {FLUIDS.map((f) => <td key={f.key} className="num px-4 py-2 text-muted">{r[f.key + '_between'].toFixed(2)} / <span className="text-ink">{r[f.key + '_forecast'].toFixed(2)}</span></td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function Comparison({ models }) {
  const groups = {}
  models.forEach((m) => {
    const [name, color] = m.experiment.includes('(selected)') ? ['Selected', COLORS.accent] : (FAMILY[m.family] || ['Other', '#94a3b8'])
    groups[name] = groups[name] || { color, rows: [] }
    groups[name].rows.push(m)
  })
  const data = Object.entries(groups).map(([name, g]) => ({
    type: 'scatter', mode: 'markers', name, x: g.rows.map((r) => r.between_WAPE), y: g.rows.map((r) => r.forecast_WAPE),
    text: g.rows.map((r) => r.experiment), hovertemplate: '%{text}<br>between %{x:.2f} % · forecast %{y:.2f} %<extra></extra>',
    marker: { color: g.color, size: name === 'Selected' ? 18 : 9, symbol: name === 'Selected' ? 'star' : 'circle', line: { width: 1, color: COLORS.ink } },
  }))
  return (
    <Card title="47 alternatives on the same folds" subtitle="Mean WAPE between tests (x) vs forecasting (y) · lower-left is better · star = selected model family">
      <Plot data={data} layout={{ hovermode: 'closest', xaxis: { title: { text: 'Between tests WAPE %' } }, yaxis: { title: { text: 'Forecast WAPE %' } },
        margin: { l: 56, r: 16, t: 44, b: 48 } }} height={360} />
    </Card>
  )
}

function NormalDay({ v }) {
  const pick = ['A XGBoost raw feats', 'A LightGBM raw feats', 'A CatBoost raw feats', 'A GaussianProcess physics feats',
    'A Ridge physics feats', 'C3 eng-targets Ridge + label feats OOF', 'F2 rec120 + GP-safe water (selected)']
  const rows = pick.map((n) => v.models.find((m) => m.experiment === n)).filter(Boolean)
  const SHORT = { 'A XGBoost raw feats': 'XGBoost', 'A LightGBM raw feats': 'LightGBM', 'A CatBoost raw feats': 'CatBoost',
    'A GaussianProcess physics feats': 'GP (all inputs)', 'A Ridge physics feats': 'Ridge',
    'C3 eng-targets Ridge + label feats OOF': 'C3 Ridge', 'F2 rec120 + GP-safe water (selected)': 'Ours' }
  const label = (n) => SHORT[n] || n
  const bias = [
    { type: 'bar', name: 'Oil', x: rows.map((r) => label(r.experiment)), y: rows.map((r) => r.bias_oil_pct), marker: { color: COLORS.oil } },
    { type: 'bar', name: 'Water', x: rows.map((r) => label(r.experiment)), y: rows.map((r) => r.bias_water_pct), marker: { color: COLORS.water } },
  ]
  const sep = [{ type: 'bar', name: 'Flowline', x: v.separator.map((s) => s.well), y: v.separator.map((s) => s.flowline), marker: { color: '#94a3b8' } },
    { type: 'bar', name: 'Wellhead', x: v.separator.map((s) => s.well), y: v.separator.map((s) => s.wellhead), marker: { color: '#c084fc' } },
    { type: 'bar', name: 'Downhole', x: v.separator.map((s) => s.well), y: v.separator.map((s) => s.downhole), marker: { color: COLORS.accent } }]
  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
      <Card title="The test-separator effect" subtitle="Pressure during tests minus normal days (psi): tests run at a lower flowline pressure">
        <Plot data={sep} layout={{ barmode: 'group', hovermode: 'closest', yaxis: { title: { text: 'psi' } }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
      </Card>
      <Card title="Normal-day check" subtitle="Bias vs physics on ~2,700 normal days (%): models using flowline pressure learn the test-only regime">
        <Plot data={bias} layout={{ barmode: 'group', hovermode: 'closest', yaxis: { title: { text: 'bias %' } }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
      </Card>
    </div>
  )
}

function Physics({ v }) {
  const wells = [...new Set(v.gor.map((g) => g.well))]
  const gor = [{ type: 'scatter', mode: 'markers', name: 'Measured GOR (each test)', x: v.gor.map((g) => g.well), y: v.gor.map((g) => g.gor),
    marker: { color: COLORS.gas, size: 7, opacity: 0.7 } },
  { type: 'scatter', mode: 'markers', name: 'Solution GOR Rs (completion file)', x: wells, y: wells.map((w) => v.gor.find((g) => g.well === w).rs),
    marker: { color: COLORS.ink, size: 16, symbol: 'line-ew-open', line: { width: 3, color: COLORS.ink } } }]
  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
      <Card title="Physics check: gas-oil ratio" subtitle="Above the bubble point the produced GOR must equal Rs — tests scatter around it (measurement noise)">
        <Plot data={gor} layout={{ hovermode: 'closest', yaxis: { title: { text: 'GOR (scf/STB)' } }, margin: { l: 56, r: 16, t: 44, b: 40 } }} height={300} />
      </Card>
      <Card title="Physics check: reservoir pressure" subtitle="End of each build-up vs the VFM reservoir pressure (inferred from well tests only)" bodyClass="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
            {['Well', 'Shut-in', 'Duration', 'Build-up end', 'VFM P_res', 'Difference'].map((h) => <th key={h} className="px-3 py-2.5 font-medium">{h}</th>)}</tr></thead>
          <tbody className="divide-y divide-line">
            {v.buildup_check.map((c) => (
              <tr key={c.well + c.t_shut}>
                <td className="px-3 py-2 font-semibold text-ink">{c.well}</td>
                <td className="num px-3 py-2 text-muted">{c.t_shut}</td>
                <td className="num px-3 py-2 text-muted">{fmt(c.duration_hr, 1)} h</td>
                <td className="num px-3 py-2 text-muted">{fmt(c.p_end_buildup, 1)}</td>
                <td className="num px-3 py-2 text-muted">{fmt(c.p_res_vfm, 1)}</td>
                <td className="px-3 py-2"><Pill tone={Math.abs(c.difference_psi) < 5 ? 'good' : 'neutral'}>{c.difference_psi > 0 ? '+' : ''}{fmt(c.difference_psi, 1)} psi</Pill></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="px-3 pb-3 pt-2 text-[11px] text-faint">Where the build-up reaches the reservoir pressure (pressure support, GL-01) the two agree within a few psi;
          elsewhere the pressure is still rising at the end of the shut-in, so it is a lower bound.</p>
      </Card>
    </div>
  )
}

function DatasetTests({ rows }) {
  const blind = rows.filter((r) => !r.in_training)
  const data = FLUIDS.map((f) => ({ type: 'scatter', mode: 'markers', name: f.label, x: rows.map((r) => r[f.key]), y: rows.map((r) => r[f.key + '_pred']),
    marker: { color: f.color, size: 8, symbol: rows.map((r) => (r.in_training ? 'circle-open' : 'circle')) },
    text: rows.map((r) => `${r.well} ${r.time.slice(0, 10)}${r.in_training ? ' (training)' : ' (blind)'}`),
    hovertemplate: '%{text}<br>measured %{x:.0f} · predicted %{y:.0f}<extra></extra>' }))
  return (
    <Card title="Well tests of the active dataset" subtitle={`${rows.length} tests · ${blind.length} after the training period (filled = blind check, open = used in training)`}>
      <Plot data={data} layout={{ hovermode: 'closest', xaxis: { title: { text: 'Measured' }, type: 'log' }, yaxis: { title: { text: 'Predicted' }, type: 'log' },
        margin: { l: 56, r: 16, t: 44, b: 48 } }} height={320} />
    </Card>
  )
}

export default function VfmValidation() {
  const { data: v, error } = useApi('/api/vfm/validation')
  if (error) return <ErrorBox error={error} />
  if (!v) return <Skeleton className="h-[600px]" />
  return (
    <div className="space-y-5">
      <Methods v={v} />
      <MetricsTable m={v.metrics} />
      <Parity v={v} />
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2"><Horizon v={v} /><PerWell v={v} /></div>
      <OverTime v={v} />
      <Steps rows={v.steps} />
      <Significance rows={v.significance} />
      <Comparison models={v.models} />
      <NormalDay v={v} />
      <Physics v={v} />
      {v.dataset_tests && v.dataset_tests.some((r) => !r.in_training) && <DatasetTests rows={v.dataset_tests} />}
    </div>
  )
}
