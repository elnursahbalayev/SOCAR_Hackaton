// Thin wrapper around Plotly.js with the app's theme (dark or light) applied to every chart.
import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import { LIGHT_SWAP, useTheme } from '../lib'

// Neutral chart colours per theme (axes, grid, text, hover box)
const THEME_STYLE = {
  dark: { grid: 'rgba(255,255,255,0.05)', zero: 'rgba(255,255,255,0.08)', axis: 'rgba(255,255,255,0.1)',
    tick: '#8d9aab', ink: '#e7edf4', hoverBg: '#131a23', hoverLine: 'rgba(255,255,255,0.12)' },
  light: { grid: 'rgba(15,23,42,0.07)', zero: 'rgba(15,23,42,0.12)', axis: 'rgba(15,23,42,0.18)',
    tick: '#536073', ink: '#101826', hoverBg: '#ffffff', hoverLine: 'rgba(15,23,42,0.15)' },
}

function baseStyle(t) {
  const axis = { gridcolor: t.grid, zerolinecolor: t.zero, linecolor: t.axis,
    tickfont: { color: t.tick, size: 11 }, title: { font: { color: t.tick, size: 11 } } }
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter Variable, sans-serif', color: t.ink, size: 12 },
    margin: { l: 56, r: 24, t: 44, b: 44 },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: t.hoverBg, bordercolor: t.hoverLine, font: { color: t.ink, size: 12 } },
    // legend sits ABOVE the plotting area so it never covers data
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { color: t.tick, size: 11 }, bgcolor: 'rgba(0,0,0,0)' },
    modebar: { bgcolor: 'rgba(0,0,0,0)', color: t.tick, activecolor: t.ink },
  }
  return { axis, layout }
}

// Build a FRESH layout object for every render. Plotly writes computed values (axis type, range)
// back into the layout it receives, so sharing objects between charts would leak settings across them.
function mergeLayout(layout, theme) {
  const { axis, layout: base } = baseStyle(THEME_STYLE[theme])
  const out = structuredClone({ ...base, ...layout })
  out.xaxis = { ...structuredClone(axis), ...structuredClone(layout.xaxis || {}) }
  out.yaxis = { ...structuredClone(axis), ...structuredClone(layout.yaxis || {}) }
  for (const key of Object.keys(layout)) {
    if (/^[xy]axis\d+$/.test(key)) out[key] = { ...structuredClone(axis), ...structuredClone(layout[key]) }
  }
  if (layout.legend) out.legend = { ...structuredClone(base.legend), ...structuredClone(layout.legend) }
  return out
}

// Light mode: replace the dark-theme colours used in traces / shapes by their light-theme equivalents.
function swapColors(value) {
  if (typeof value === 'string') return LIGHT_SWAP[value.toLowerCase()] || value
  if (Array.isArray(value)) return value.map(swapColors)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, swapColors(v)]))
  }
  return value
}

export default function Plot({ data, layout = {}, height = 320, className = '' }) {
  const ref = useRef(null)
  const { theme } = useTheme()
  useEffect(() => {
    const light = theme === 'light'
    Plotly.react(ref.current, light ? swapColors(data) : data, mergeLayout(light ? swapColors(layout) : layout, theme),
      { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] })
  }, [data, layout, theme])
  useEffect(() => {
    const node = ref.current
    return () => Plotly.purge(node)
  }, [])
  return <div ref={ref} className={className} style={{ height, width: '100%' }} />
}
