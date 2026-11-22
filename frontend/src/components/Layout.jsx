// App shell: sidebar navigation + top bar with the engineering workflow stepper.
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Activity, Check, ChevronRight, Database, FileText, LayoutDashboard, Lightbulb, Moon, Sun, Waves,
} from 'lucide-react'
import { useTheme } from '../lib'

export const PAGES = [
  { path: '/', label: 'Overview', icon: LayoutDashboard, title: 'Field Overview',
    subtitle: 'Production, reservoir health and actions at a glance' },
  { path: '/data', label: 'Data', icon: Database, step: 1, title: 'Field Data',
    subtitle: 'Upload, quality control and measurement explorer' },
  { path: '/vfm', label: 'Virtual Flow Meter', icon: Activity, step: 2, title: 'Virtual Flow Metering',
    subtitle: 'Daily oil, water and gas rates from sensor data' },
  { path: '/pta', label: 'Well Test Analysis', icon: Waves, step: 3, title: 'Pressure Transient Analysis',
    subtitle: 'Automatic build-up interpretation for every shut-in' },
  { path: '/insights', label: 'Insights', icon: Lightbulb, step: 4, title: 'Insights & Actions',
    subtitle: 'Where value is lost or can be gained' },
  { path: '/report', label: 'Report', icon: FileText, step: 5, title: 'Reports & Export',
    subtitle: 'Deliverables and engineering report' },
]

function Logo() {
  return (
    <div className="flex items-center gap-3 px-2">
      <img src="/logo.svg" alt="" className="h-9 w-9 rounded-lg shadow-[0_0_24px_rgba(245,181,68,0.35)]" />
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight text-ink">WellMind</div>
        <div className="text-[11px] text-muted">Integrated Well Performance</div>
      </div>
    </div>
  )
}

// Active dataset (provided or uploaded); refreshes when the Data page announces a change
function DatasetPanel() {
  const [info, setInfo] = useState(null)
  useEffect(() => {
    const load = () => fetch('/api/dataset').then((r) => r.json()).then(setInfo).catch(() => setInfo(null))
    load()
    window.addEventListener('dataset-changed', load)
    return () => window.removeEventListener('dataset-changed', load)
  }, [])
  return (
    <NavLink to="/data" className="mt-auto block space-y-3 rounded-xl border border-line bg-white/[0.02] p-3.5 text-[11px] transition hover:border-line-strong">
      <div className="flex items-center justify-between">
        <span className="text-muted">Dataset</span>
        {info ? <span className="flex items-center gap-1.5 text-good"><span className="live-dot h-1.5 w-1.5 rounded-full bg-good" />{info.id === 'bundled' ? 'Provided' : 'Uploaded'}</span>
          : <span className="text-faint">…</span>}
      </div>
      <div className="text-ink">{info ? `${info.wells.length} gas-lifted wells · ${info.n_days} days` : 'Loading'}</div>
      <div className="num text-faint">{info ? `${info.period.start.slice(0, 10)} → ${info.period.end.slice(0, 10)}` : ''}</div>
      <div className="border-t border-line pt-2 text-faint">VFM model <span className="text-muted">v5 · physics + GP</span></div>
    </NavLink>
  )
}

function Sidebar() {
  return (
    <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-line bg-panel/60 px-3 py-5 backdrop-blur-md lg:flex">
      <Logo />
      <nav className="mt-8 flex flex-col gap-0.5">
        <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">Workspace</div>
        {PAGES.map(({ path, label, icon: Icon, step }) => (
          <NavLink key={path} to={path} end
            className={({ isActive }) => `group flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition
              ${isActive ? 'bg-white/[0.06] text-ink shadow-[inset_2px_0_0_0_var(--color-accent)]' : 'text-muted hover:bg-white/[0.03] hover:text-ink'}`}>
            {({ isActive }) => (<>
              <Icon size={17} className={isActive ? 'text-accent' : 'text-faint group-hover:text-muted'} />
              <span className="flex-1">{label}</span>
              {step && <span className="num text-[10px] text-faint">0{step}</span>}
            </>)}
          </NavLink>
        ))}
      </nav>
      <DatasetPanel />
    </aside>
  )
}

function Stepper({ current }) {
  const steps = PAGES.filter((p) => p.step)
  return (
    <ol className="flex items-center gap-1 text-[12px]">
      {steps.map((s, i) => {
        const done = current > s.step
        const active = current === s.step
        return (
          <li key={s.path} className="flex items-center gap-1">
            <NavLink to={s.path} className={`flex items-center gap-2 rounded-full px-2.5 py-1 transition
              ${active ? 'bg-accent/10 text-accent ring-1 ring-accent/30' : done ? 'text-ink hover:bg-white/5' : 'text-faint hover:bg-white/5 hover:text-muted'}`}>
              <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold
                ${active ? 'bg-accent text-on-accent' : done ? 'bg-good/15 text-good' : 'bg-white/5 text-faint'}`}>
                {done ? <Check size={11} strokeWidth={3} /> : s.step}
              </span>
              <span className="hidden xl:inline">{s.label}</span>
            </NavLink>
            {i < steps.length - 1 && <ChevronRight size={14} className="text-faint/60" />}
          </li>
        )
      })}
    </ol>
  )
}

// Dark / light switch (remembered in the browser)
function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const dark = theme === 'dark'
  return (
    <button onClick={toggle} title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="flex h-8 w-8 items-center justify-center rounded-full border border-line text-muted transition hover:border-line-strong hover:text-ink">
      {dark ? <Sun size={15} /> : <Moon size={15} />}
    </button>
  )
}

export default function Layout() {
  const { pathname } = useLocation()
  const page = PAGES.find((p) => p.path === pathname) || PAGES[0]
  return (
    <div className="flex min-h-full">
      <Sidebar />
      <div className="grid-bg flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-line bg-bg/70 backdrop-blur-xl">
          <div className="flex flex-wrap items-center justify-between gap-4 px-8 py-4">
            <div>
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-faint">
                <span>Gas-lift field</span><ChevronRight size={12} /><span className="text-muted">{page.label}</span>
              </div>
              <h1 className="mt-1 text-xl font-semibold tracking-tight text-ink">{page.title}</h1>
              <p className="text-[13px] text-muted">{page.subtitle}</p>
            </div>
            <div className="flex items-center gap-4">
              <Stepper current={page.step || 0} />
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1600px] flex-1 px-8 py-7">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
