// Small shared helpers: API fetching, number formatting, colours.
import { createContext, useContext, useEffect, useState } from 'react'

export const COLORS = {
  oil: '#34c77b', water: '#4c8dff', gas: '#f0605d', accent: '#f5b544',
  ink: '#e7edf4', muted: '#8d9aab', faint: '#5d6877', line: 'rgba(255,255,255,0.07)',
}

// Fetch JSON from the Python backend. Returns { data, error, loading }.
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    setState({ data: null, error: null, loading: true })
    if (!path) return () => { alive = false }          // nothing to load yet (e.g. waiting for a selection)
    fetch(path)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }))
    return () => { alive = false }
  }, [path])
  return state
}

// 12345.6 -> "12,346"   (digits = decimals)
export function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

// 3838008 -> "3.84M"
export function compact(value) {
  if (value === null || value === undefined) return '—'
  return Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value)
}

// +3.2 -> "+3.2%"
export function pct(value, digits = 1) {
  if (value === null || value === undefined) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}%`
}

// ---------- theme (dark / light) ----------
// The choice is stored in the browser and applied as <html data-theme="...">, which switches the CSS tokens.
export const ThemeContext = createContext({ theme: 'light', toggle: () => {} })
export const useTheme = () => useContext(ThemeContext)

export function useThemeState() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'light')
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('wellmind-theme', theme) } catch { /* private mode: just don't remember */ }
  }, [theme])
  return { theme, toggle: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')) }
}

// Charts are written with the dark-theme colours; in light mode Plot.jsx swaps them for these
// (near-white lines and overlays would vanish on a white background, bright colours need more contrast).
export const LIGHT_SWAP = {
  '#e7edf4': '#101826', '#8d9aab': '#536073', '#5d6877': '#8590a0',
  '#34c77b': '#1a9a56', '#4c8dff': '#2f6fdb', '#f0605d': '#d64545', '#f5b544': '#d18a12',
  '#7dd3fc': '#0e8fc4', '#c084fc': '#8b46d8', '#22d3ee': '#0e9fb5', '#94a3b8': '#64748b', '#4a5566': '#b4bdc9',
  'rgba(231,237,244,0.35)': 'rgba(16,24,38,0.35)', 'rgba(255,255,255,0.04)': 'rgba(15,23,42,0.05)',
}
