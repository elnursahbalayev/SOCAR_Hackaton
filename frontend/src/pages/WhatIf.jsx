// What-if for one well: the inflow curve (IPR) and the oil gained by lowering the flowing pressure or removing skin.
import { useEffect, useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'
import Plot from '../components/Plot'
import { Card, ErrorBox, Skeleton } from '../components/ui'
import { COLORS, compact, fmt, useApi } from '../lib'

function Slider({ label, value, min, max, step, onChange, format, hint }) {
  return (
    <label className="block" title={hint}>
      <div className="flex justify-between text-xs"><span className="text-muted">{label}</span><span className="num text-ink">{format(value)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 w-full accent-[var(--color-accent)]" />
    </label>
  )
}

export default function WhatIf({ well }) {
  const { data, error } = useApi(`/api/vfm/well/${well}/ipr`)
  const [dPwf, setDPwf] = useState(0)
  const [skinNew, setSkinNew] = useState(null)
  const [price, setPrice] = useState(70)
  useEffect(() => { setDPwf(0); setSkinNew(null) }, [well])
  if (error) return <ErrorBox error={error} />
  if (!data) return <Skeleton className="h-96" />

  const { J, p_res: pRes, pb, pwf_now: pwfNow, water_cut_now: wc, rs, skin, ln_term: lnTerm } = data
  // Keep the well above the bubble point with a 100 psi margin (below Pb gas comes out of solution near the well).
  const maxDrop = Math.max(0, Math.min(800, Math.floor((pwfNow - pb - 100) / 25) * 25))
  const sNew = skin == null ? null : (skinNew ?? skin)
  // Darcy: J is proportional to 1 / (ln(re/rw) - 0.75 + s)  ->  removing skin raises J
  const Jnew = skin == null ? J : J * (lnTerm + skin) / (lnTerm + sNew)
  const pwfNew = pwfNow - dPwf
  const qNow = J * (pRes - pwfNow)            // oil at today's operating point: q = J (P_res - Pwf)
  const qNew = Jnew * (pRes - pwfNew)
  const gain = qNew - qNow
  const waterGain = (gain * wc) / (100 - wc)  // same water cut -> water rises with the liquid
  const gasGain = (gain * rs) / 1000          // gas = oil x Rs

  const scale = Jnew / J
  const traces = [
    { x: data.curve.oil, y: data.curve.pwf, name: 'Inflow today (IPR)', type: 'scatter', mode: 'lines', line: { color: COLORS.oil, width: 2 } },
  ]
  if (Math.abs(scale - 1) > 1e-6) {
    traces.push({ x: data.curve.oil.map((q) => q * scale), y: data.curve.pwf, name: 'Inflow after skin removal', type: 'scatter',
      mode: 'lines', line: { color: COLORS.oil, width: 2, dash: 'dash' } })
  }
  traces.push({ x: [qNow], y: [pwfNow], name: 'Operating point today', type: 'scatter', mode: 'markers',
    marker: { color: COLORS.ink, size: 11, symbol: 'circle' } })
  if (gain > 0.5) {
    traces.push({ x: [qNew], y: [pwfNew], name: 'What-if operating point', type: 'scatter', mode: 'markers',
      marker: { color: COLORS.accent, size: 13, symbol: 'star' } })
  }
  const layout = {
    hovermode: 'closest', margin: { l: 64, r: 20, t: 40, b: 44 },
    xaxis: { title: { text: 'Oil rate (STB/d)' }, rangemode: 'tozero' },
    yaxis: { title: { text: 'Flowing bottomhole pressure Pwf (psia)' }, rangemode: 'tozero' },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: pb, y1: pb, line: { color: COLORS.gas, width: 1, dash: 'dot' } }],
    annotations: [{ xref: 'paper', x: 1, y: pb, text: `bubble point ${fmt(pb)} psia`, showarrow: false, xanchor: 'right', yanchor: 'bottom',
      font: { size: 10, color: COLORS.gas } }],
  }

  return (
    <Card title={`${well} · inflow & what-if`} subtitle="IPR: q = J (P_res − Pwf) above the bubble point, Vogel below · move the sliders to test an action"
      actions={<SlidersHorizontal size={15} className="text-faint" />}>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="lg:col-span-3"><Plot data={traces} layout={layout} height={340} /></div>
        <div className="space-y-4 lg:col-span-2">
          <Slider label="Lower flowing pressure by (more lift gas / bigger choke)" value={dPwf} min={0} max={maxDrop} step={25}
            onChange={setDPwf} format={(v) => `${fmt(v)} psi → Pwf ${fmt(pwfNow - v)} psia`}
            hint={`Limited to Pwf ≥ bubble point + 100 psi (${fmt(pb + 100)} psia)`} />
          {skin != null ? (
            <Slider label={`Skin after stimulation (today ${skin.toFixed(2)})`} value={sNew} min={Math.min(-3, skin)} max={Math.max(skin, 0)}
              step={0.1} onChange={setSkinNew} format={(v) => `${v.toFixed(1)} → J ${J.toFixed(2)} → ${Jnew.toFixed(2)} STB/d/psi`}
              hint="J = k h / (141.2 Bo μ (ln(re/rw) − 0.75 + s)): lower skin, higher productivity index" />
          ) : (
            <div className="rounded-lg border border-line p-3 text-xs text-muted">No build-up test for this well, so skin is unknown — run one to evaluate stimulation.</div>
          )}
          <div className="grid grid-cols-3 gap-2">
            {[['Oil', gain, 'STB/d', COLORS.oil], ['Water', waterGain, 'STB/d', COLORS.water], ['Gas', gasGain, 'Mscf/d', COLORS.gas]].map(([l, v, u, c]) => (
              <div key={l} className="rounded-xl border border-line p-3">
                <div className="text-[11px] text-muted">{l}</div>
                <div className="num text-lg" style={{ color: c }}>{v >= 0 ? '+' : ''}{fmt(v)}</div>
                <div className="text-[10px] text-faint">{u}</div>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between rounded-xl border border-accent/30 bg-accent/[0.06] p-3">
            <div>
              <div className="text-[11px] text-muted">Value of the extra oil</div>
              <div className="num text-lg text-ink">${compact(gain * price)}/day · ${compact(gain * price * 365)}/year</div>
            </div>
            <label className="flex items-center gap-1 text-xs text-muted">$<input value={price} onChange={(e) => setPrice(Number(e.target.value) || 0)}
              className="num w-12 rounded-md border border-line bg-panel-2 px-1.5 py-0.5 text-right text-ink" />/bbl</label>
          </div>
          <p className="text-[11px] leading-relaxed text-faint">
            Today: oil {fmt(qNow)} STB/d at Pwf {fmt(pwfNow)} psia, P<sub>res</sub> {fmt(pRes)} psia, J {J.toFixed(2)} STB/d/psi.
            The what-if assumes the same reservoir pressure and water cut; how much lift gas or choke change reaches the new Pwf is not modelled.
          </p>
        </div>
      </div>
    </Card>
  )
}
