// Data page: active dataset, upload of raw files (runs the full pipeline), data-quality report, explorer.
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AlertTriangle, CheckCircle2, CircleDashed, Database, Download, FileSpreadsheet, Loader2, RefreshCw,
  UploadCloud, XCircle,
} from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, Pill, Skeleton } from '../components/ui'
import { COLORS, compact, fmt, useApi } from '../lib'

const STEP_LABEL = {
  validate: 'Validate files', clean: 'Clean sensor data', features: 'Build daily features',
  vfm: 'Virtual flow meter', pta: 'Shut-ins & build-up analysis', export: 'Write Excel results',
}
const TYPE_LABEL = { high_freq: 'Per-minute sensors', choke: 'Choke events', well_tests: 'Well tests',
  pvt: 'Reservoir PVT', completion: 'Well completion' }

// tell the rest of the app (sidebar, other pages) that the active dataset changed
export function announceDatasetChange() { window.dispatchEvent(new Event('dataset-changed')) }

// ---------- active dataset ----------
function DatasetCard({ info, list, onActivate }) {
  return (
    <Card title="Active dataset" subtitle="Every page shows results for this dataset"
      actions={<Pill tone={info.id === 'bundled' ? 'info' : 'good'} dot>{info.label}</Pill>}>
      <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
        <Stat label="Period" value={`${info.period.start.slice(0, 10)} → ${info.period.end.slice(0, 10)}`} />
        <Stat label="Wells" value={info.wells.length} />
        <Stat label="Days" value={info.n_days} />
        <Stat label="Shut-ins / build-up tests" value={`${info.n_shut_ins} / ${info.n_buildup_tests}`} />
      </div>
      {list.length > 1 && (
        <div className="mt-5 border-t border-line pt-4">
          <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Switch dataset</div>
          <div className="flex flex-wrap gap-2">
            {list.map((d) => (
              <button key={d.id} onClick={() => onActivate(d.id)} disabled={d.active}
                className={`rounded-lg border px-3 py-1.5 text-xs transition ${d.active
                  ? 'border-accent/50 bg-accent/10 text-accent' : 'border-line text-muted hover:border-line-strong hover:text-ink'}`}>
                {d.id === 'bundled' ? 'Provided dataset' : d.id} · <span className="num">{d.period.start.slice(0, 10)} → {d.period.end.slice(0, 10)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.1em] text-faint">{label}</div>
      <div className="num mt-1 text-ink">{value}</div>
    </div>
  )
}

// ---------- upload + pipeline progress ----------
function StepRow({ step, state }) {
  const icon = state === 'done' ? <CheckCircle2 size={16} className="text-good" />
    : state === 'running' ? <Loader2 size={16} className="animate-spin text-accent" />
      : state === 'error' ? <XCircle size={16} className="text-bad" />
        : <CircleDashed size={16} className="text-faint" />
  return (
    <li className="flex items-center gap-3 py-1.5 text-[13px]">
      {icon}<span className={state === 'pending' ? 'text-faint' : 'text-ink'}>{STEP_LABEL[step]}</span>
    </li>
  )
}

function Uploader({ onFinished }) {
  const [files, setFiles] = useState([])
  const [drag, setDrag] = useState(false)
  const [uploadPct, setUploadPct] = useState(null)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const input = useRef(null)
  const busy = uploadPct !== null || job?.status === 'running'

  function addFiles(list) {
    const csv = [...list].filter((f) => f.name.toLowerCase().endsWith('.csv'))
    setFiles((old) => [...old.filter((o) => !csv.some((c) => c.name === o.name)), ...csv])
  }

  function start() {
    setError(null); setJob(null); setUploadPct(0)
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    const xhr = new XMLHttpRequest()                     // XHR (not fetch) to show upload progress for big files
    xhr.open('POST', '/api/upload')
    xhr.upload.onprogress = (e) => e.lengthComputable && setUploadPct(Math.round((100 * e.loaded) / e.total))
    xhr.onload = () => {
      setUploadPct(null)
      if (xhr.status !== 200) { setError(JSON.parse(xhr.responseText).detail || xhr.statusText); return }
      setJob(JSON.parse(xhr.responseText))
    }
    xhr.onerror = () => { setUploadPct(null); setError('Upload failed (network)') }
    xhr.send(form)
  }

  // poll the pipeline job
  useEffect(() => {
    if (!job || job.status !== 'running') return
    const t = setInterval(() => {
      fetch(`/api/jobs/${job.id}`).then((r) => r.json()).then((j) => {
        setJob(j)
        if (j.status === 'done') { announceDatasetChange(); onFinished() }
      })
    }, 800)
    return () => clearInterval(t)
  }, [job, onFinished])

  const stepState = (s) => {
    if (!job) return 'pending'
    const i = job.steps.indexOf(s), cur = job.steps.indexOf(job.step)
    if (job.status === 'done') return 'done'
    if (job.status === 'error') return i < cur ? 'done' : i === cur ? 'error' : 'pending'
    return i < cur ? 'done' : i === cur ? 'running' : 'pending'
  }
  const totalMb = files.reduce((s, f) => s + f.size, 0) / 1e6

  return (
    <Card title="Upload well data" subtitle="Raw CSV files, any subset — recognised by their columns, so any file name works">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files) }}
            onClick={() => input.current.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-9 text-center transition
              ${drag ? 'border-accent bg-accent/5' : 'border-line-strong hover:border-accent/60 hover:bg-white/[0.02]'}`}>
            <UploadCloud size={30} className="text-accent" />
            <div className="mt-3 text-sm text-ink">Drop CSV files here or click to choose</div>
            <div className="mt-1 text-xs text-muted">per-minute sensors (required) · choke events · well tests · PVT · completion</div>
            <input ref={input} type="file" accept=".csv" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
          </div>
          {files.length > 0 && (
            <ul className="mt-3 divide-y divide-line rounded-lg border border-line text-[13px]">
              {files.map((f) => (
                <li key={f.name} className="flex items-center justify-between px-3 py-2">
                  <span className="flex items-center gap-2 text-ink"><FileSpreadsheet size={14} className="text-faint" />{f.name}</span>
                  <span className="num text-xs text-faint">{(f.size / 1e6).toFixed(1)} MB</span>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-4 flex items-center gap-3">
            <button onClick={start} disabled={!files.length || busy}
              className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition hover:brightness-110 disabled:opacity-40">
              {busy ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}Run full analysis
            </button>
            {files.length > 0 && !busy && <button onClick={() => setFiles([])} className="text-xs text-muted hover:text-ink">clear</button>}
            {files.length > 0 && <span className="num text-xs text-faint">{files.length} file(s) · {totalMb.toFixed(1)} MB</span>}
          </div>
          {error && <div className="mt-3 rounded-lg border border-bad/30 bg-bad/5 p-3 text-sm text-bad">{error}</div>}
        </div>

        <div className="lg:col-span-2">
          <div className="text-[11px] uppercase tracking-[0.12em] text-faint">Pipeline</div>
          {uploadPct !== null && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted"><span>Uploading</span><span className="num">{uploadPct}%</span></div>
              <div className="mt-1 h-1.5 rounded-full bg-white/5"><div className="h-full rounded-full bg-accent" style={{ width: `${uploadPct}%` }} /></div>
            </div>
          )}
          <ol className="mt-2">{(job?.steps || Object.keys(STEP_LABEL)).map((s) => <StepRow key={s} step={s} state={stepState(s)} />)}</ol>
          {job?.status === 'error' && <div className="mt-2 rounded-lg border border-bad/30 bg-bad/5 p-3 text-xs text-bad">{job.error}</div>}
          {job?.status === 'done' && (
            <div className="mt-3 rounded-lg border border-good/30 bg-good/5 p-3 text-xs text-good">
              Done — the uploaded dataset is now active on every page.
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

// ---------- quality report ----------
function QualityReport({ info }) {
  const cr = info.clean_report
  const sensors = Object.keys(cr.null_codes)
  const acc = info.accuracy_on_uploaded_tests
  return (
    <Card title="Data quality & validation" subtitle="What was recognised, and what the cleaning fixed">
      <table className="w-full text-[13px]">
        <thead><tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.1em] text-faint">
          {['File', 'Recognised as', 'Rows', 'Wells', 'Period'].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">
          {info.files.map((f) => (
            <tr key={f.file}>
              <td className="px-3 py-2 text-ink">{f.file}</td>
              <td className="px-3 py-2">{f.type ? <Pill tone="good">{TYPE_LABEL[f.type]}</Pill> : <Pill tone="warn">not recognised</Pill>}</td>
              <td className="num px-3 py-2 text-muted">{f.rows ? compact(f.rows) : '—'}</td>
              <td className="num px-3 py-2 text-muted">{f.wells ? f.wells.length : '—'}</td>
              <td className="num px-3 py-2 text-muted">{f.start ? `${f.start.slice(0, 10)} → ${f.end.slice(0, 10)}` : 'static'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-2">
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Cleaning (per sensor)</div>
          <table className="w-full text-xs">
            <thead><tr className="text-left text-faint"><th className="py-1 font-medium">Sensor</th><th className="font-medium">Null codes −999.25</th><th className="font-medium">Spikes removed</th></tr></thead>
            <tbody>
              {sensors.map((s) => (
                <tr key={s} className="border-t border-line">
                  <td className="py-1.5 text-muted">{s.replace(/_/g, ' ')}</td>
                  <td className="num text-ink">{fmt(cr.null_codes[s])}</td>
                  <td className="num text-ink">{fmt(cr.spikes_removed[s])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 text-xs text-muted">Duplicate rows dropped: <span className="num text-ink">{fmt(cr.duplicates_dropped)}</span> ·
            shut-in minutes: <span className="num text-ink">{fmt(cr.shut_in_minutes)}</span></div>
        </div>
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Warnings</div>
          {info.warnings.length === 0 ? <div className="flex items-center gap-2 text-sm text-good"><CheckCircle2 size={15} />No warnings</div> : (
            <ul className="space-y-2">{info.warnings.map((w) => (
              <li key={w} className="flex gap-2 text-xs text-muted"><AlertTriangle size={14} className="mt-0.5 shrink-0 text-warn" />{w}</li>))}</ul>
          )}
          {acc && (
            <div className="mt-4">
              <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Model accuracy on the uploaded well tests ({acc.n_tests})</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                {['oil', 'water', 'gas'].map((f) => (
                  <div key={f} className="rounded-lg border border-line p-2">
                    <div className="capitalize text-muted">{f}</div>
                    <div className="num text-ink">WAPE {acc[f].WAPE.toFixed(1)}%</div>
                    <div className="num text-faint">R² {acc[f].R2.toFixed(3)}</div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-[11px] leading-relaxed text-faint">
                {acc.n_in_sample > 0
                  ? `${acc.n_in_sample} of ${acc.n_tests} tests are inside the training period (up to 30 Sep 2025) → in-sample fit, not a blind check.`
                  : 'All tests are after the training period → a blind forecast check.'}
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

// ---------- explorer ----------
function Explorer({ wells }) {
  const [well, setWell] = useState(wells[0])
  const { data, loading, error } = useApi(`/api/data/signals/${well}`)
  let plot = <Skeleton className="h-[520px]" />
  if (error) plot = <ErrorBox error={error} />
  if (data && !loading) {
    const t = data.time
    const traces = [
      { x: t, y: data.dh_pressure_psia, name: 'Downhole pressure', line: { color: COLORS.accent, width: 1.2 }, yaxis: 'y' },
      { x: t, y: data.whp_psia, name: 'WHP', line: { color: '#c084fc', width: 1.2 }, yaxis: 'y2' },
      { x: t, y: data.whp_downstream_psia, name: 'Downstream', line: { color: '#94a3b8', width: 1 }, yaxis: 'y2' },
      { x: t, y: data.gas_lift_rate_mscfd, name: 'Gas lift', line: { color: '#22d3ee', width: 1.2 }, yaxis: 'y3' },
      { x: t, y: data.choke_opening_pct, name: 'Choke %', line: { color: COLORS.oil, width: 1.4, shape: 'hv' }, yaxis: 'y4' },
    ].map((tr) => ({ ...tr, type: 'scatter', mode: 'lines' }))
    const shapes = data.shut_ins.map((s) => {
      const end = new Date(new Date(s.t_shut.replace(' ', 'T')).getTime() + s.duration_hr * 3600e3)
      return { type: 'rect', xref: 'x', yref: 'paper', x0: s.t_shut, x1: end.toISOString().slice(0, 16).replace('T', ' '),
        y0: 0, y1: 1, line: { width: 0 },
        fillcolor: s.event_type === 'planned build-up test' ? 'rgba(245,181,68,0.18)' : 'rgba(141,154,171,0.18)' }
    })
    const layout = {
      grid: { rows: 4, columns: 1, pattern: 'coupled', roworder: 'top to bottom' }, shapes,
      yaxis: { title: { text: 'Pwf (psia)' } }, yaxis2: { title: { text: 'WHP (psia)' } },
      yaxis3: { title: { text: 'Lift gas (Mscf/d)' } }, yaxis4: { title: { text: 'Choke (%)' } },
      margin: { l: 64, r: 20, t: 40, b: 36 },
    }
    plot = <Plot data={traces} layout={layout} height={560} />
  }
  return (
    <Card title="Measurement explorer" subtitle="Hourly averages · amber = planned build-up test · grey = unplanned shut-in"
      actions={<select value={well} onChange={(e) => setWell(e.target.value)}
        className="rounded-lg border border-line bg-panel-2 px-3 py-1.5 text-xs text-ink outline-none focus:border-accent/50">
        {wells.map((w) => <option key={w}>{w}</option>)}</select>}>
      {plot}
    </Card>
  )
}

// ---------- page ----------
export default function Data() {
  const [version, setVersion] = useState(0)                     // bump to reload after a dataset change
  const info = useApi(`/api/dataset?v=${version}`)
  const list = useApi(`/api/datasets?v=${version}`)
  const refresh = () => setVersion((v) => v + 1)

  function activate(id) {
    fetch(`/api/datasets/${id}/activate`, { method: 'POST' }).then(() => { announceDatasetChange(); refresh() })
  }

  if (info.error) return <ErrorBox error={info.error} />
  return (
    <div className="space-y-5">
      {info.data && list.data ? <DatasetCard info={info.data} list={list.data} onActivate={activate} /> : <Skeleton className="h-36" />}
      <Uploader onFinished={refresh} />
      {info.data && (
        <div className="flex flex-wrap gap-3">
          <Link to="/" className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-muted hover:text-ink"><Database size={14} />Overview</Link>
          <Link to="/pta" className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-muted hover:text-ink">Well test analysis →</Link>
          <a href="/api/vfm/export.xlsx" className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-muted hover:text-ink"><Download size={14} />VFM results (Excel)</a>
          <a href="/api/pta/export.xlsx" className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-muted hover:text-ink"><Download size={14} />PTA results (Excel)</a>
        </div>
      )}
      {info.data && <QualityReport info={info.data} />}
      {info.data && <Explorer key={info.data.id} wells={info.data.wells} />}
    </div>
  )
}
