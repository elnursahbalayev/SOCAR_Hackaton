// Reports & export: every deliverable of the active dataset + a printable engineering report.
import { useState } from 'react'
import { Brain, Download, ExternalLink, FileSpreadsheet, FileText, Images, Loader2, Printer } from 'lucide-react'
import { Card, ErrorBox, Pill, Skeleton } from '../components/ui'
import { useApi } from '../lib'

// extra: secondary downloads of the same result, e.g. [{ href: '/api/vfm/export.csv', label: 'CSV' }]
function Deliverable({ icon: Icon, title, text, href, button, tone = 'accent', note, extra = [] }) {
  const [busy, setBusy] = useState(false)
  return (
    <div className="rise flex flex-col rounded-2xl border border-line bg-panel/80 p-5">
      <div className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-full border border-accent/40 bg-bg"><Icon size={18} className="text-accent" /></span>
        <div className="text-[14px] font-semibold text-ink">{title}</div>
      </div>
      <p className="mt-3 flex-1 text-xs leading-relaxed text-muted">{text}</p>
      {note && <div className="mt-2 text-[11px] text-faint">{note}</div>}
      <a href={href} onClick={() => { setBusy(true); setTimeout(() => setBusy(false), 9000) }}
        className={`mt-4 flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition
          ${tone === 'accent' ? 'bg-accent text-on-accent hover:brightness-110' : 'border border-line text-ink hover:border-line-strong'}`}>
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}{button}
      </a>
      {extra.length > 0 && (
        <div className="mt-2 flex gap-2">
          {extra.map((x) => (
            <a key={x.href} href={x.href} title={x.title}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-line px-2 py-1.5 text-[11px] font-semibold text-ink transition hover:border-line-strong">
              <Download size={12} />{x.label}
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Report() {
  const info = useApi('/api/dataset')
  const [price, setPrice] = useState(70)
  const [loaded, setLoaded] = useState(false)
  if (info.error) return <ErrorBox error={info.error} />
  if (!info.data) return <Skeleton className="h-96" />
  const d = info.data
  const reportUrl = `/api/report.html?oil_price=${price}`
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
        <span>Exports for <span className="text-ink">{d.label}</span> · <span className="num">{d.period.start.slice(0, 10)} → {d.period.end.slice(0, 10)}</span> · {d.wells.length} wells</span>
        <Pill tone="info">{d.n_shut_ins} shut-ins · {d.n_buildup_tests} build-up tests</Pill>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <Deliverable icon={FileSpreadsheet} title="VFM results" href="/api/vfm/export.xlsx" button="Excel"
          extra={[{ href: '/api/vfm/export.csv', label: 'CSV', title: 'timestamp, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd' }]}
          text="Daily oil, water and gas per well (date, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd) + model info, validation and daily details." />
        <Deliverable icon={FileSpreadsheet} title="PTA results" href="/api/pta/export.xlsx" button="Excel"
          extra={[{ href: '/api/pta/export.csv', label: 'CSV', title: 'well_name, shut_in_time, permeability_md, skin, delta_p_skin_psi, horner_slope (planned build-up tests)' }]}
          text="Horner slope, skin, permeability and ΔP skin per build-up test + all shut-ins, details, sensitivity and method sheets." />
        <Deliverable icon={Images} title="PTA plots" href="/api/pta/plots.zip" button="ZIP of all plots" tone="line"
          text="History, Horner and log-log derivative plots plus the analytical model match for every shut-in." note="Generated on request (~10 s)" />
        <Deliverable icon={FileText} title="Engineering report" href={`${reportUrl}&download=true`} button="HTML report" tone="line"
          text="Actions, VFM method and validation, per-well table, PTA results with plots, data quality — one self-contained file." />
        <Deliverable icon={Brain} title="Trained VFM model" href="/api/model.pkl" button="vfm_model.pkl" tone="line"
          text="The stored model used for every prediction: per-well physics trends + Gaussian process for water (Python pickle)." />
      </div>

      <Card title="Engineering report · preview" subtitle="Open it in a new tab and use the browser's Print → Save as PDF for a PDF report"
        actions={<div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-muted">Oil price $
            <select value={price} onChange={(e) => { setPrice(Number(e.target.value)); setLoaded(false) }}
              className="rounded-md border border-line bg-panel-2 px-2 py-1 text-ink outline-none">
              {[50, 60, 70, 80, 90].map((p) => <option key={p}>{p}</option>)}</select>/bbl</label>
          <a href={reportUrl} target="_blank" rel="noreferrer" className="flex items-center gap-2 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent hover:brightness-110">
            <Printer size={14} />Open / print to PDF<ExternalLink size={12} /></a>
        </div>}
        bodyClass="p-0">
        <div className="relative">
          {!loaded && <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-muted"><Loader2 size={16} className="animate-spin" />Building the report (plots are drawn on the server)…</div>}
          <iframe key={reportUrl} src={reportUrl} title="report" onLoad={() => setLoaded(true)}
            className={`h-[900px] w-full rounded-b-2xl bg-[#ffffff] transition-opacity ${loaded ? 'opacity-100' : 'opacity-0'}`} />
        </div>
      </Card>
    </div>
  )
}
