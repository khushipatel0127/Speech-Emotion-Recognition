import React, { useEffect, useState } from 'react'
import { ArrowRight, CircleAlert, Database, Loader2, Mic2 } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import EmotionBadge from '../components/EmotionBadge.jsx'
import { EMOTION_COLORS } from '../components/emotionTheme.js'

const EMPTY_STATS = { total: 0, recent_count: 0, average_duration_seconds: 0, uncertain_count: 0, emotion_distribution: {} }

function parseScores(value) {
  try { return typeof value === 'string' ? JSON.parse(value) : value || {} } catch { return {} }
}

export default function OverviewResearch() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState(null)
  const [recent, setRecent] = useState([])
  const [stats, setStats] = useState(EMPTY_STATS)

  useEffect(() => {
    let mounted = true
    Promise.all([
      fetch('/api/health').then((response) => response.json()),
      fetch('/api/history?per_page=5').then((response) => response.json()),
      fetch('/api/stats').then((response) => response.json()),
    ]).then(([healthData, historyData, statsData]) => {
      if (!mounted) return
      setHealth(healthData)
      setRecent(historyData.items || [])
      setStats(statsData || EMPTY_STATS)
    }).catch(() => {
      if (mounted) setHealth({ status: 'error', models_loaded: false })
    }).finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [])

  const model = health?.model
  const metrics = [
    ['Stored analyses', stats.total, 'SQLite records'],
    ['Last 7 days', stats.recent_count, 'Recent runs'],
    ['Mean duration', `${Number(stats.average_duration_seconds || 0).toFixed(1)}s`, 'Processed audio'],
    ['Flagged uncertain', stats.uncertain_count, 'Low score or margin'],
  ]

  return (
    <div className="research-page">
      <header className="research-heading">
        <div>
          <div className="research-kicker">Speech emotion research workspace</div>
          <h1 className="research-title">Overview</h1>
          <p className="research-description">Inspect recorded analyses, deployment status and measured model behaviour.</p>
        </div>
        <div className="research-note">MODEL SCORES ARE ESTIMATES, NOT GUARANTEED CONFIDENCE</div>
      </header>

      {health && (!health.models_loaded || health.status !== 'ok') && (
        <div className="lab-alert lab-alert-error" style={{ marginBottom: 18 }}>
          <CircleAlert size={17} />Backend or model unavailable. Start FastAPI on port 8000 and refresh this page.
        </div>
      )}

      <section className="metric-grid-research" aria-label="Prediction statistics">
        {metrics.map(([label, value, detail]) => (
          <div className="metric-research" key={label}>
            <div className="eyebrow">{label}</div>
            <div className="metric-value-research">{value}</div>
            <div className="metric-detail-research">{detail}</div>
          </div>
        ))}
      </section>

      <div className="overview-grid-research" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.35fr) minmax(280px, .65fr)', gap: 18, marginTop: 18 }}>
        <section className="lab-panel">
          <div className="lab-panel-header">
            <div>
              <div className="lab-panel-title">Recent analyses</div>
              <div className="lab-panel-meta">Newest stored predictions</div>
            </div>
            <Link className="lab-button lab-button-ghost" to="/history">All records <ArrowRight size={14} /></Link>
          </div>
          {loading ? (
            <div className="lab-panel-body" style={{ minHeight: 220, display: 'grid', placeItems: 'center', color: 'var(--muted)' }}><Loader2 className="spin" size={22} /></div>
          ) : recent.length === 0 ? (
            <div className="lab-panel-body" style={{ minHeight: 220, display: 'grid', placeItems: 'center', textAlign: 'center' }}>
              <div><Mic2 size={24} color="var(--accent)" /><p style={{ marginTop: 10, fontSize: 13, fontWeight: 700 }}>No analyses stored</p><p style={{ marginTop: 4, color: 'var(--muted)', fontSize: 11 }}>Record or upload speech to begin.</p></div>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table className="lab-table-research">
                <thead><tr><th>Recording</th><th>Prediction</th><th>Score</th><th>Model</th><th /></tr></thead>
                <tbody>
                  {recent.map((item) => {
                    const scores = parseScores(item.probabilities)
                    const top = Math.max(...Object.values(scores), 0)
                    return (
                      <tr key={item.id}>
                        <td><div style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 650 }}>{item.filename}</div><div className="mono" style={{ color: 'var(--faint)', fontSize: 9, marginTop: 3 }}>{Number(item.duration_seconds || 0).toFixed(1)} SEC</div></td>
                        <td><EmotionBadge emotion={item.predicted_emotion} showIcon /></td>
                        <td className="mono">{(top * 100).toFixed(1)}%</td>
                        <td className="mono" style={{ color: 'var(--faint)', fontSize: 9 }}>{item.model_version?.includes('phase2') ? 'PHASE 2' : 'BASELINE'}</td>
                        <td><button className="lab-button lab-button-ghost" onClick={() => navigate(`/history/${item.id}`)} style={{ minHeight: 30, padding: '0 10px' }}>Open</button></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <div style={{ display: 'grid', gap: 18, alignContent: 'start' }}>
          <section className="lab-panel">
            <div className="lab-panel-header"><div><div className="lab-panel-title">Active system</div><div className="lab-panel-meta">Runtime configuration</div></div><span className="status-dot-research" /></div>
            <div className="lab-panel-body" style={{ display: 'grid', gap: 13 }}>
              {[
                ['Model', model?.backbone || 'Historical ensemble'],
                ['Output classes', model?.classes?.length || 0],
                ['Device', model?.device || 'CPU'],
                ['Public holdout', model?.kind === 'phase2_wavlm' ? '77.54% accuracy' : 'Historical baseline'],
              ].map(([label, value]) => <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 18, paddingBottom: 10, borderBottom: '1px solid #e8e9e6' }}><span className="eyebrow">{label}</span><span className="mono" style={{ fontSize: 10, textAlign: 'right' }}>{value}</span></div>)}
            </div>
          </section>

          <section className="lab-panel">
            <div className="lab-panel-header"><div className="lab-panel-title">Prediction distribution</div></div>
            <div className="lab-panel-body" style={{ display: 'grid', gap: 11 }}>
              {Object.keys(stats.emotion_distribution || {}).length === 0 ? <span style={{ color: 'var(--muted)', fontSize: 11 }}>No stored results.</span> : Object.entries(stats.emotion_distribution).map(([emotion, count]) => {
                const percentage = stats.total ? (count / stats.total) * 100 : 0
                return <div key={emotion}><div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 10 }}><span style={{ textTransform: 'capitalize', fontWeight: 700 }}>{emotion}</span><span className="mono">{count} / {percentage.toFixed(0)}%</span></div><div className="score-track-research"><div className="score-fill-research" style={{ '--score-width': `${percentage}%`, '--score-color': EMOTION_COLORS[emotion] || '#65747a' }} /></div></div>
              })}
            </div>
          </section>
        </div>
      </div>

      <div className="lab-alert lab-alert-warning" style={{ marginTop: 18 }}><CircleAlert size={17} /><div><strong>Evaluation context.</strong> The 77.54% result is from held-out acted datasets. Natural microphone performance is still under evaluation and may be substantially lower.</div></div>
    </div>
  )
}
