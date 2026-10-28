import React, { useEffect, useRef, useState } from 'react'
import { AlertCircle, ArrowLeft, Download, Loader2, Pause, Play } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { EMOTION_COLORS } from '../components/emotionTheme.js'

function formatSize(bytes) {
  if (!bytes) return '0 B'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function DetailResearch() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef(null)

  useEffect(() => {
    fetch(`/api/history/${id}`).then((response) => { if (response.status === 404) throw new Error('not-found'); return response.json() }).then(setData).catch((error) => { if (error.message === 'not-found') setNotFound(true) }).finally(() => setLoading(false))
  }, [id])

  if (loading) return <div style={{ minHeight: '60vh', display: 'grid', placeItems: 'center' }}><Loader2 className="spin" size={25} color="var(--accent)" /></div>
  if (notFound || !data) return <div className="research-page"><button className="lab-button lab-button-ghost" onClick={() => navigate('/history')}><ArrowLeft size={13} />Back to records</button><div className="lab-alert lab-alert-error" style={{ marginTop: 18 }}><AlertCircle size={17} />This analysis record does not exist or has been deleted.</div></div>

  let rawScores = {}
  try { rawScores = typeof data.probabilities === 'string' ? JSON.parse(data.probabilities) : data.probabilities || {} } catch { rawScores = {} }
  const scores = Object.entries(rawScores).map(([emotion, score]) => ({ emotion: emotion.toLowerCase(), score, color: EMOTION_COLORS[emotion.toLowerCase()] || '#65747a' })).sort((a, b) => b.score - a.score)
  const prediction = data.predicted_emotion?.toLowerCase()
  const color = EMOTION_COLORS[prediction] || 'var(--accent)'
  const phase2 = data.model_version?.includes('phase2')
  const timestamp = new Date(data.created_at).toLocaleString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const download = () => {
    const link = document.createElement('a')
    link.href = `/api/report/${id}`
    link.download = `speechsense_report_${id.slice(0, 8)}.txt`
    document.body.appendChild(link); link.click(); document.body.removeChild(link)
  }

  return (
    <div className="research-page">
      <header className="research-heading">
        <div><button className="lab-button lab-button-ghost" onClick={() => navigate('/history')} style={{ marginBottom: 15 }}><ArrowLeft size={13} />Back to records</button><div className="research-kicker">Stored inference result</div><h1 className="research-title">Analysis detail</h1></div>
        <button className="lab-button lab-button-secondary" onClick={download}><Download size={14} />Download report</button>
      </header>

      <div className="analysis-grid-research" style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, .72fr) minmax(0, 1.28fr)', gap: 18, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 18 }}>
          <section className="lab-panel"><div className="lab-panel-body" style={{ padding: 22 }}><div className="eyebrow">Predicted acoustic emotion</div><div style={{ marginTop: 10, color, fontSize: 40, lineHeight: 1, fontWeight: 730, letterSpacing: '-.045em', textTransform: 'capitalize' }}>{prediction}</div><div className="mono" style={{ marginTop: 15, fontSize: 20 }}>{((scores[0]?.score || 0) * 100).toFixed(1)}% <span style={{ color: 'var(--faint)', fontSize: 9 }}>TOP MODEL SCORE</span></div>{data.is_uncertain && <div className="lab-alert lab-alert-warning" style={{ marginTop: 17 }}><AlertCircle size={15} />{data.uncertainty_note || 'Interpret this result cautiously.'}</div>}</div></section>
          <section className="lab-panel"><div className="lab-panel-header"><div className="lab-panel-title">Audio playback</div></div><div className="lab-panel-body"><audio ref={audioRef} src={`/audio/${id}`} onEnded={() => setPlaying(false)} /><button className="lab-button lab-button-secondary" onClick={() => { if (playing) audioRef.current?.pause(); else audioRef.current?.play(); setPlaying(!playing) }}>{playing ? <Pause size={14} /> : <Play size={14} />}{playing ? 'Pause recording' : 'Play recording'}</button></div></section>
          <section className="lab-panel"><div className="lab-panel-header"><div className="lab-panel-title">Recording metadata</div></div><div className="lab-panel-body" style={{ display: 'grid', gap: 12 }}>{[
            ['Filename', data.filename], ['Duration', `${Number(data.duration_seconds || 0).toFixed(2)} seconds`], ['File size', formatSize(data.file_size_bytes)], ['Model', data.model_version || 'Historical record'], ['Processing', data.processing_time_ms != null ? `${Number(data.processing_time_ms).toFixed(0)} ms` : 'Unavailable'], ['Timestamp', timestamp],
          ].map(([label, value]) => <div key={label} style={{ paddingBottom: 10, borderBottom: '1px solid #e8e9e6' }}><div className="eyebrow">{label}</div><div style={{ marginTop: 6, overflowWrap: 'anywhere', fontSize: 11 }}>{value}</div></div>)}</div></section>
        </div>

        <section className="lab-panel"><div className="lab-panel-header"><div><div className="lab-panel-title">Complete score distribution</div><div className="lab-panel-meta">{phase2 ? 'WavLM-Large embeddings / six-class logistic classifier' : 'Historical acoustic-feature ensemble'}</div></div></div><div className="lab-panel-body"><div style={{ display: 'grid', gap: 15 }}>{scores.map(({ emotion, score, color: scoreColor }) => <div className="score-row-research" key={emotion}><div className="score-label-research"><span className="score-dot-research" style={{ '--score-color': scoreColor }} />{emotion}</div><div className="score-track-research"><div className="score-fill-research" style={{ '--score-width': `${score * 100}%`, '--score-color': scoreColor }} /></div><div className="score-value-research">{(score * 100).toFixed(1)}%</div></div>)}</div><div className="lab-alert lab-alert-info" style={{ marginTop: 24 }}><AlertCircle size={16} /><div><strong>Architecture note.</strong> {phase2 ? 'The deployed model mean-pools 1,024-dimensional WavLM-Large speech embeddings and applies a validated six-class logistic classifier.' : 'This record was produced by the preserved 84-feature SVM and HistGradientBoosting baseline.'} Scores are model outputs rather than verified psychological confidence.</div></div></div></section>
      </div>
    </div>
  )
}
