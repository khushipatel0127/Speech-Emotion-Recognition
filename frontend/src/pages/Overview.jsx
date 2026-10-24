import React, { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  Mic,
  ArrowRight,
  Zap,
  Layers,
  BarChart2,
  Database,
  Eye,
  Loader2,
  AudioWaveform,
  AlertTriangle,
  Clock,
  CalendarDays,
  ShieldQuestion,
} from 'lucide-react'
import EmotionBadge from '../components/EmotionBadge.jsx'
import { EMOTION_COLORS } from '../components/emotionTheme.js'

export default function Overview() {
  const navigate = useNavigate()
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [backendHealth, setBackendHealth] = useState({ ok: true, modelsLoaded: true, model: null })
  const [stats, setStats] = useState({
    total: 0,
    recent_count: 0,
    average_duration_seconds: 0,
    uncertain_count: 0,
    emotion_distribution: {},
  })

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
      fetch('/api/stats')
        .then((res) => res.json())
        .catch(() => ({
          total: 0,
          recent_count: 0,
          average_duration_seconds: 0,
          uncertain_count: 0,
          emotion_distribution: {},
        })),
    ]).then(([healthData, historyData, statsData]) => {
      if (isMounted) {
        setBackendHealth({
          ok: healthData.status === 'ok',
          modelsLoaded: !!healthData.models_loaded,
          model: healthData.model || null,
        })
        setRecentAnalyses(historyData.items || [])
        setStats(statsData)
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
      {(!backendHealth.ok || !backendHealth.modelsLoaded) && (
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
            <strong>{backendHealth.ok ? 'The prediction model is unavailable.' : 'Backend API is not responding.'}</strong>{' '}
            Ensure FastAPI is running on port 8000 using{' '}
            <code style={{ background: '#FEE2E2', padding: '2px 6px', borderRadius: '4px' }}>
              .venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --port 8000
            </code>
          </div>
        </div>
      )}

      {/* Hero Banner */}
      <div
        className="speechsense-hero"
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
          className="speechsense-hero-waveform"
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

      {/* Database-backed dashboard statistics */}
      <div
        className="overview-stats-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.05fr) minmax(320px, 0.95fr)',
          gap: '20px',
          marginBottom: '28px',
        }}
      >
        <div
          className="overview-kpi-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
            gap: '14px',
          }}
        >
          {[
            {
              label: 'Total Analyses',
              value: stats.total,
              detail: 'Stored in SQLite',
              icon: Database,
              color: '#2563EB',
              background: '#EFF6FF',
            },
            {
              label: 'Last 7 Days',
              value: stats.recent_count,
              detail: 'Recent predictions',
              icon: CalendarDays,
              color: '#7C3AED',
              background: '#F5F3FF',
            },
            {
              label: 'Average Audio',
              value: `${Number(stats.average_duration_seconds || 0).toFixed(1)}s`,
              detail: 'Recording duration',
              icon: Clock,
              color: '#059669',
              background: '#ECFDF5',
            },
            {
              label: 'Low-Certainty',
              value: stats.uncertain_count,
              detail: 'Flagged predictions',
              icon: ShieldQuestion,
              color: '#D97706',
              background: '#FFFBEB',
            },
          ].map((card) => {
            const Icon = card.icon
            return (
              <div
                key={card.label}
                style={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #E2E8F0',
                  borderRadius: '14px',
                  padding: '18px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '14px',
                }}
              >
                <div
                  style={{
                    width: '42px',
                    height: '42px',
                    flexShrink: 0,
                    borderRadius: '11px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: card.background,
                  }}
                >
                  <Icon size={20} color={card.color} />
                </div>
                <div>
                  <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F172A', lineHeight: 1.1 }}>
                    {card.value}
                  </div>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: '#475569', marginTop: '4px' }}>
                    {card.label}
                  </div>
                  <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>{card.detail}</div>
                </div>
              </div>
            )
          })}
        </div>

        <div
          style={{
            backgroundColor: '#FFFFFF',
            border: '1px solid #E2E8F0',
            borderRadius: '14px',
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          }}
        >
          <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A' }}>Emotion Distribution</h3>
          <p style={{ fontSize: '12px', color: '#64748B', marginTop: '3px', marginBottom: '16px' }}>
            Real predictions saved in analysis history
          </p>
          {Object.keys(stats.emotion_distribution || {}).length === 0 ? (
            <div
              style={{
                minHeight: '120px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#94A3B8',
                fontSize: '12px',
              }}
            >
              Analyze audio to populate this chart.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {Object.entries(stats.emotion_distribution).map(([emotion, count]) => {
                const percentage = stats.total > 0 ? (count / stats.total) * 100 : 0
                const color = EMOTION_COLORS[emotion] || '#64748B'
                return (
                  <div key={emotion} title={`${count} ${emotion} prediction${count === 1 ? '' : 's'}`}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '11px',
                        marginBottom: '4px',
                      }}
                    >
                      <span style={{ color: '#475569', fontWeight: 600, textTransform: 'capitalize' }}>{emotion}</span>
                      <span style={{ color: '#64748B' }}>{count} · {percentage.toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '7px', borderRadius: '999px', backgroundColor: '#F1F5F9', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${percentage}%`,
                          height: '100%',
                          borderRadius: '999px',
                          backgroundColor: color,
                          transition: 'width 0.35s ease',
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
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
            desc: 'Processes WAV, MP3, M4A, MP4, and live microphone audio through one validated backend prediction path.',
          },
          {
            icon: Layers,
            color: '#7C3AED',
            bg: '#F5F3FF',
            title: backendHealth.model?.kind === 'phase2_wavlm' ? 'WavLM-Large Model' : 'Dual-Model Ensemble',
            desc: backendHealth.model?.kind === 'phase2_wavlm'
              ? 'Frozen pretrained speech representations with a validated six-emotion Logistic Regression classifier.'
              : '50/50 soft voting ensemble combining Calibrated Support Vector Machine and HistGradientBoosting.',
          },
          {
            icon: BarChart2,
            color: '#059669',
            bg: '#ECFDF5',
            title: backendHealth.model?.kind === 'phase2_wavlm' ? '1,024-D Speech Embeddings' : '84 Acoustic Features',
            desc: backendHealth.model?.kind === 'phase2_wavlm'
              ? 'Uses mean-pooled WavLM-Large embeddings from 16 kHz mono speech with speaker-independent evaluation.'
              : 'Extracts 40 MFCC means, 40 MFCC standard deviations, RMS energy, and Zero-Crossing Rates at 16 kHz.',
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
