import React, { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  Mic,
  ArrowRight,
  Zap,
  Layers,
  BarChart2,
  Database,
  History,
  Clock,
  Eye,
  Loader2,
  AudioWaveform,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react'
import EmotionBadge from '../components/EmotionBadge.jsx'

export default function Overview() {
  const navigate = useNavigate()
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [backendHealth, setBackendHealth] = useState({ ok: true, modelsLoaded: true })

  useEffect(() => {
    let isMounted = true

    // Check health & fetch recent analyses
    Promise.all([
      fetch('/api/health')
        .then((res) => res.json())
        .catch(() => ({ status: 'error', models_loaded: false })),
      fetch('/api/history?per_page=5')
        .then((res) => res.json())
        .catch(() => ({ items: [] })),
    ]).then(([healthData, historyData]) => {
      if (isMounted) {
        setBackendHealth({
          ok: healthData.status === 'ok',
          modelsLoaded: !!healthData.models_loaded,
        })
        setRecentAnalyses(historyData.items || [])
        setLoading(false)
      }
    })

    return () => {
      isMounted = false
    }
  }, [])

  return (
    <div className="fade-in">
      {/* Backend Alert if offline */}
      {!backendHealth.ok && (
        <div
          style={{
            backgroundColor: '#FEF2F2',
            border: '1px solid #FECACA',
            borderRadius: '12px',
            padding: '14px 18px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '24px',
            color: '#B91C1C',
          }}
        >
          <AlertTriangle size={20} color="#DC2626" />
          <div style={{ fontSize: '13px', lineHeight: 1.5 }}>
            <strong>Backend API is not responding.</strong> Ensure FastAPI is running on port 8000 using{' '}
            <code style={{ background: '#FEE2E2', padding: '2px 6px', borderRadius: '4px' }}>
              .venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --port 8000
            </code>
          </div>
        </div>
      )}

      {/* Hero Banner */}
      <div
        style={{
          background: 'linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 45%, #2563EB 100%)',
          borderRadius: '20px',
          padding: '36px 44px',
          color: '#FFFFFF',
          boxShadow: '0 10px 25px -5px rgba(37, 99, 235, 0.25), 0 8px 10px -6px rgba(37, 99, 235, 0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '32px',
          marginBottom: '32px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Background glow circle */}
        <div
          style={{
            position: 'absolute',
            right: '-60px',
            top: '-60px',
            width: '280px',
            height: '280px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />

        <div style={{ flex: 1, zIndex: 1, maxWidth: '640px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: 'rgba(255, 255, 255, 0.16)',
              backdropFilter: 'blur(8px)',
              padding: '4px 12px',
              borderRadius: '9999px',
              fontSize: '12px',
              fontWeight: 600,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
              color: '#DBEAFE',
              marginBottom: '16px',
            }}
          >
            <Zap size={14} color="#93C5FD" />
            Machine Learning Audio Intelligence
          </div>

          <h2
            style={{
              fontSize: '32px',
              fontWeight: 800,
              lineHeight: 1.2,
              letterSpacing: '-0.02em',
              marginBottom: '14px',
            }}
          >
            Understand the emotion in a voice.
          </h2>

          <p
            style={{
              fontSize: '15px',
              color: '#BFDBFE',
              lineHeight: 1.6,
              marginBottom: '26px',
            }}
          >
            SpeechSense processes speech recordings to extract 84 acoustic features across MFCCs, energy, and zero-crossing rates, delivering multi-class probability scores via an ensemble of Calibrated SVM and HistGradientBoosting.
          </p>

          <button
            onClick={() => navigate('/analyze')}
            style={{
              backgroundColor: '#FFFFFF',
              color: '#1D4ED8',
              border: 'none',
              borderRadius: '12px',
              padding: '13px 26px',
              fontSize: '15px',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '10px',
              boxShadow: '0 4px 14px rgba(0, 0, 0, 0.12)',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)'
              e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.18)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 4px 14px rgba(0, 0, 0, 0.12)'
            }}
          >
            <Mic size={18} strokeWidth={2.2} />
            <span>Analyze Speech</span>
            <ArrowRight size={16} strokeWidth={2.2} />
          </button>
        </div>

        {/* Audio Waveform Graphic */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            height: '110px',
            padding: '16px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '16px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            zIndex: 1,
            flexShrink: 0,
          }}
        >
          {[32, 54, 78, 95, 60, 42, 85, 100, 72, 45, 88, 65, 40, 75, 50, 30].map((h, i) => (
            <div
              key={i}
              style={{
                width: '6px',
                height: `${h}%`,
                borderRadius: '4px',
                backgroundColor: '#FFFFFF',
                opacity: 0.4 + (h / 100) * 0.6,
                animation: `waveBar 1.2s ease-in-out infinite alternate`,
                animationDelay: `${i * 0.08}s`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Feature Cards Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '20px',
          marginBottom: '36px',
        }}
      >
        {[
          {
            icon: Zap,
            color: '#2563EB',
            bg: '#EFF6FF',
            title: 'Real-Time Inference',
            desc: 'Processes WAV, MP3, M4A, MP4, and live microphone audio within milliseconds, generating complete predictions instantly.',
          },
          {
            icon: Layers,
            color: '#7C3AED',
            bg: '#F5F3FF',
            title: 'Dual-Model Ensemble',
            desc: '50/50 soft voting ensemble combining Calibrated Support Vector Machine and HistGradientBoosting.',
          },
          {
            icon: BarChart2,
            color: '#059669',
            bg: '#ECFDF5',
            title: '84 Acoustic Features',
            desc: 'Extracts 40 MFCC means, 40 MFCC standard deviations, RMS energy, and Zero-Crossing Rates at 16 kHz.',
          },
          {
            icon: Database,
            color: '#D97706',
            bg: '#FFFBEB',
            title: 'Persistent SQLite Storage',
            desc: 'Preserves every analysis record, probability breakdown, and audio file for playback and report export.',
          },
        ].map((feat) => {
          const Icon = feat.icon
          return (
            <div
              key={feat.title}
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
                boxShadow: '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)',
                transition: 'transform 0.15s ease, box-shadow 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.06)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)'
              }}
            >
              <div
                style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '12px',
                  backgroundColor: feat.bg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '16px',
                }}
              >
                <Icon size={22} color={feat.color} />
              </div>
              <h3
                style={{
                  fontSize: '16px',
                  fontWeight: 700,
                  color: '#0F172A',
                  marginBottom: '8px',
                }}
              >
                {feat.title}
              </h3>
              <p
                style={{
                  fontSize: '13px',
                  color: '#64748B',
                  lineHeight: 1.6,
                }}
              >
                {feat.desc}
              </p>
            </div>
          )
        })}
      </div>

      {/* Recent Analyses Card */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '16px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          padding: '28px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '20px',
            borderBottom: '1px solid #F1F5F9',
            paddingBottom: '16px',
          }}
        >
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0F172A' }}>
              Recent Analyses
            </h3>
            <p style={{ fontSize: '13px', color: '#64748B', marginTop: '2px' }}>
              Latest speech predictions processed by the ensemble model
            </p>
          </div>
          <Link
            to="/history"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: 600,
              color: '#2563EB',
              textDecoration: 'none',
              padding: '6px 12px',
              borderRadius: '8px',
              transition: 'background-color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#EFF6FF')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            <span>View All</span>
            <ArrowRight size={14} />
          </Link>
        </div>

        {loading ? (
          <div
            style={{
              padding: '48px 0',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              color: '#94A3B8',
            }}
          >
            <Loader2 size={28} className="spin" color="#2563EB" />
            <span style={{ fontSize: '13px' }}>Loading analysis records...</span>
          </div>
        ) : recentAnalyses.length === 0 ? (
          /* Empty State */
          <div
            style={{
              padding: '48px 24px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <div
              style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                backgroundColor: '#EFF6FF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}
            >
              <AudioWaveform size={32} color="#2563EB" />
            </div>
            <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A', marginBottom: '6px' }}>
              No speech analyses yet
            </h4>
            <p style={{ fontSize: '13px', color: '#64748B', maxWidth: '380px', marginBottom: '20px' }}>
              Upload your voice recording (WAV, MP3, M4A, MP4) or record live with your microphone to detect emotion and populate history.
            </p>
            <button
              onClick={() => navigate('/analyze')}
              style={{
                backgroundColor: '#2563EB',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '10px',
                padding: '10px 20px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Mic size={16} />
              <span>Analyze Speech</span>
            </button>
          </div>
        ) : (
          /* Table of recent items */
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #E2E8F0', color: '#64748B' }}>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Audio File</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Predicted Emotion</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Confidence</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Duration</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600 }}>Analyzed At</th>
                  <th style={{ padding: '12px 16px', fontWeight: 600, textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {recentAnalyses.map((item) => {
                  const probs = typeof item.probabilities === 'string'
                    ? JSON.parse(item.probabilities)
                    : item.probabilities || {}
                  const topScore = Math.max(...Object.values(probs), 0)
                  const dateStr = new Date(item.created_at).toLocaleString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })

                  return (
                    <tr
                      key={item.id}
                      style={{
                        borderBottom: '1px solid #F1F5F9',
                        transition: 'background-color 0.15s ease',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#F8FAFC')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <td style={{ padding: '14px 16px', fontWeight: 600, color: '#0F172A' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Mic size={16} color="#64748B" />
                          <span style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {item.filename}
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '14px 16px' }}>
                        <EmotionBadge emotion={item.predicted_emotion} showIcon={true} />
                      </td>
                      <td style={{ padding: '14px 16px', fontWeight: 600, color: '#0F172A' }}>
                        {(topScore * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '14px 16px', color: '#64748B' }}>
                        {item.duration_seconds.toFixed(1)}s
                      </td>
                      <td style={{ padding: '14px 16px', color: '#64748B', whiteSpace: 'nowrap' }}>
                        {dateStr}
                      </td>
                      <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                        <button
                          onClick={() => navigate(`/history/${item.id}`)}
                          style={{
                            backgroundColor: '#EFF6FF',
                            color: '#2563EB',
                            border: '1px solid #BFDBFE',
                            borderRadius: '8px',
                            padding: '6px 12px',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#2563EB'
                            e.currentTarget.style.color = '#FFFFFF'
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#EFF6FF'
                            e.currentTarget.style.color = '#2563EB'
                          }}
                        >
                          <Eye size={13} />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
