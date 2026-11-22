import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout'
import { ThemeContext, useThemeState } from './lib'
import Data from './pages/Data'
import Insights from './pages/Insights'
import Overview from './pages/Overview'
import Pta from './pages/Pta'
import Report from './pages/Report'
import Vfm from './pages/Vfm'

function App({ children }) {
  return <ThemeContext.Provider value={useThemeState()}>{children}</ThemeContext.Provider>
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Overview />} />
            <Route path="/data" element={<Data />} />
            <Route path="/vfm" element={<Vfm />} />
            <Route path="/pta" element={<Pta />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/report" element={<Report />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </App>
  </StrictMode>,
)
