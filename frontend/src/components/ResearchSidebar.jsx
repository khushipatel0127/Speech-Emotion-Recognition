import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Activity, AudioLines, Database, History, Mic2 } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Overview', icon: Activity, end: true },
  { to: '/analyze', label: 'Analyze', icon: Mic2 },
  { to: '/history', label: 'Records', icon: History },
]

export default function ResearchSidebar() {
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    let mounted = true
    fetch('/api/health')
      .then((response) => response.json())
      .then((data) => { if (mounted && data.models_loaded) setModelInfo(data.model) })
      .catch(() => {})
    return () => { mounted = false }
  }, [])

  const phase2Active = modelInfo?.kind === 'phase2_wavlm'
  return (
    <aside className="sidebar-research">
      <div className="brand-research">
        <div className="brand-mark-research"><AudioLines size={19} strokeWidth={1.8} /></div>
        <div>
          <div className="brand-name-research">SpeechSense</div>
          <div className="brand-subtitle-research">AUDIO EMOTION LAB</div>
        </div>
      </div>
      <nav className="side-nav-research" aria-label="Primary navigation">
        <div className="nav-label-research">Workspace</div>
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item-research${isActive ? ' active' : ''}`}>
            <Icon size={16} strokeWidth={1.8} /><span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="model-status-research">
        <div className="status-line-research"><span className="status-dot-research" />{phase2Active ? 'Inference ready' : 'Baseline ready'}</div>
        <div className="status-meta-research">
          <Database size={11} style={{ marginRight: 6, verticalAlign: -2 }} />
          {phase2Active ? 'WavLM-Large / 6 classes' : 'Acoustic ensemble / 8 classes'}<br />16 kHz mono input
        </div>
      </div>
    </aside>
  )
}
