import React from 'react'
import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Mic, History, AudioWaveform, Activity } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/analyze', label: 'Analyze Speech', icon: Mic, end: false },
  { to: '/history', label: 'Analysis History', icon: History, end: false },
]

export default function Sidebar() {
  return (
    <aside
      style={{
        width: '260px',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        backgroundColor: '#0F172A',
        color: '#F8FAFC',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px',
        boxSizing: 'border-box',
        zIndex: 50,
        borderRight: '1px solid #1E293B',
      }}
    >
      {/* Brand Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '4px 8px 24px', borderBottom: '1px solid #1E293B' }}>
        <div
          style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(37, 99, 235, 0.35)',
            flexShrink: 0,
          }}
        >
          <AudioWaveform size={22} color="#FFFFFF" strokeWidth={2.2} />
        </div>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 800, letterSpacing: '-0.02em', color: '#FFFFFF', lineHeight: 1.2 }}>
            SpeechSense
          </h1>
          <p style={{ fontSize: '11px', fontWeight: 500, color: '#94A3B8', marginTop: '2px', letterSpacing: '0.01em' }}>
            Speech Emotion Recognition
          </p>
        </div>
      </div>

      {/* Nav Menu */}
      <nav style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
        <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748B', padding: '0 12px 6px' }}>
          Menu
        </div>
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 16px',
              borderRadius: '10px',
              fontSize: '14px',
              fontWeight: isActive ? 600 : 500,
              textDecoration: 'none',
              transition: 'all 0.15s ease',
              backgroundColor: isActive ? '#2563EB' : 'transparent',
              color: isActive ? '#FFFFFF' : '#94A3B8',
              boxShadow: isActive ? '0 4px 12px rgba(37, 99, 235, 0.3)' : 'none',
            })}
            onMouseEnter={(e) => {
              if (!e.currentTarget.getAttribute('aria-current')) {
                e.currentTarget.style.backgroundColor = '#1E293B'
                e.currentTarget.style.color = '#F8FAFC'
              }
            }}
            onMouseLeave={(e) => {
              if (!e.currentTarget.getAttribute('aria-current')) {
                e.currentTarget.style.backgroundColor = 'transparent'
                e.currentTarget.style.color = '#94A3B8'
              }
            }}
          >
            <Icon size={18} strokeWidth={2} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer Info */}
      <div
        style={{
          marginTop: 'auto',
          padding: '16px',
          borderRadius: '12px',
          backgroundColor: '#1E293B',
          border: '1px solid #334155',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#10B981',
              boxShadow: '0 0 8px #10B981',
            }}
          />
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#E2E8F0' }}>
            Ensemble Model Active
          </span>
        </div>
        <p style={{ fontSize: '11px', color: '#94A3B8', lineHeight: 1.4 }}>
          SVM + HistGradientBoosting<br />
          84 Acoustic Features (16 kHz)
        </p>
      </div>
    </aside>
  )
}
