import React, { useCallback, useEffect, useState } from 'react'
import { AlertCircle, ChevronLeft, ChevronRight, Eye, Loader2, Search, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import EmotionBadge from '../components/EmotionBadge.jsx'

const ALL_EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

function scoresFor(item) {
  try { return typeof item.probabilities === 'string' ? JSON.parse(item.probabilities) : item.probabilities || {} } catch { return {} }
}

export default function HistoryResearch() {
  const navigate = useNavigate()
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [search, setSearch] = useState('')
  const [emotion, setEmotion] = useState('')
  const [loading, setLoading] = useState(true)
  const [clearOpen, setClearOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [working, setWorking] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    const query = new URLSearchParams({ page: String(page), per_page: '10' })
    if (search.trim()) query.set('search', search.trim())
    if (emotion) query.set('emotion', emotion)
    try {
      const response = await fetch(`/api/history?${query}`)
      const data = await response.json()
      setItems(data.items || [])
      setTotal(data.total || 0)
      setPages(data.pages || 1)
    } catch {
      setItems([]); setTotal(0); setPages(1)
    } finally { setLoading(false) }
  }, [page, search, emotion])

  useEffect(() => { load() }, [load])

  const clearAll = async () => {
    setWorking(true)
    try { await fetch('/api/history', { method: 'DELETE' }); setClearOpen(false); setPage(1); await load() } finally { setWorking(false) }
  }

  const deleteOne = async () => {
    if (!deleteTarget) return
    setWorking(true)
    try {
      await fetch(`/api/history/${deleteTarget.id}`, { method: 'DELETE' })
      setDeleteTarget(null)
      if (items.length === 1 && page > 1) setPage((value) => value - 1)
      else await load()
    } finally { setWorking(false) }
  }

  return (
    <div className="research-page">
      <header className="research-heading">
        <div><div className="research-kicker">Persistent prediction archive</div><h1 className="research-title">Analysis records</h1><p className="research-description">Search stored recordings, inspect model versions and reopen complete result details.</p></div>
        <div className="research-note">{total} SQLITE RECORD{total === 1 ? '' : 'S'}<br />PAGE {page} OF {pages}</div>
      </header>

      {(clearOpen || deleteTarget) && (
        <div className="modal-backdrop-research" role="dialog" aria-modal="true">
          <div className="modal-panel-research">
            <AlertCircle size={20} color="#b64444" />
            <h2 style={{ marginTop: 13, fontSize: 17 }}>{clearOpen ? 'Delete all analysis records?' : 'Delete this analysis record?'}</h2>
            <p style={{ marginTop: 8, color: 'var(--muted)', fontSize: 12, lineHeight: 1.6 }}>{clearOpen ? `This removes all ${total} database records and stored audio files.` : `This removes “${deleteTarget?.filename}” and its stored audio file.`} This cannot be undone.</p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}><button className="lab-button lab-button-ghost" disabled={working} onClick={() => { setClearOpen(false); setDeleteTarget(null) }}>Cancel</button><button className="lab-button" disabled={working} onClick={clearOpen ? clearAll : deleteOne} style={{ color: 'white', background: '#b64444', borderColor: '#b64444' }}>{working ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}Delete</button></div>
          </div>
        </div>
      )}

      <section className="lab-panel">
        <div className="lab-panel-header">
          <div className="lab-panel-title">Filter records</div>
          {total > 0 && <button className="lab-button lab-button-ghost" style={{ color: '#a63f3f' }} onClick={() => setClearOpen(true)}><Trash2 size={13} />Clear archive</button>}
        </div>
        <div className="lab-panel-body" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 210px', gap: 10 }}>
          <label style={{ position: 'relative' }}><Search size={14} style={{ position: 'absolute', left: 12, top: 13, color: 'var(--faint)' }} /><input className="field-research" value={search} onChange={(event) => { setSearch(event.target.value); setPage(1) }} placeholder="Search by recording filename" style={{ paddingLeft: 36 }} /></label>
          <select className="field-research" value={emotion} onChange={(event) => { setEmotion(event.target.value); setPage(1) }}><option value="">All emotions</option>{ALL_EMOTIONS.map((value) => <option key={value} value={value}>{value[0].toUpperCase() + value.slice(1)}</option>)}</select>
        </div>
      </section>

      <section className="lab-panel" style={{ marginTop: 14 }}>
        {loading ? <div className="lab-panel-body" style={{ minHeight: 260, display: 'grid', placeItems: 'center' }}><Loader2 size={22} className="spin" color="var(--accent)" /></div> : items.length === 0 ? <div className="lab-panel-body" style={{ minHeight: 260, display: 'grid', placeItems: 'center', textAlign: 'center' }}><div><div className="eyebrow">No matching records</div><p style={{ marginTop: 9, color: 'var(--muted)', fontSize: 12 }}>Change the filters or analyze a recording.</p></div></div> : <div style={{ overflowX: 'auto' }}><table className="lab-table-research"><thead><tr><th>Recording</th><th>Prediction</th><th>Top score</th><th>Duration</th><th>Model</th><th>Timestamp</th><th /></tr></thead><tbody>{items.map((item) => {
          const scores = scoresFor(item)
          const top = Math.max(...Object.values(scores), 0)
          const date = new Date(item.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })
          return <tr key={item.id}><td><div style={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 650 }}>{item.filename}</div></td><td><EmotionBadge emotion={item.predicted_emotion} showIcon /></td><td className="mono">{(top * 100).toFixed(1)}%</td><td className="mono">{Number(item.duration_seconds || 0).toFixed(1)}s</td><td className="mono" style={{ color: 'var(--faint)', fontSize: 9 }}>{item.model_version?.includes('phase2') ? 'PHASE 2' : 'BASELINE'}</td><td style={{ whiteSpace: 'nowrap', color: 'var(--muted)' }}>{date}</td><td><div style={{ display: 'flex', gap: 6 }}><button className="lab-button lab-button-ghost" style={{ minHeight: 31, padding: '0 9px' }} onClick={() => navigate(`/history/${item.id}`)}><Eye size={13} />Open</button><button className="lab-button lab-button-ghost" style={{ minWidth: 31, minHeight: 31, padding: 0, color: '#a63f3f' }} onClick={() => setDeleteTarget(item)} aria-label={`Delete ${item.filename}`}><Trash2 size={13} /></button></div></td></tr>
        })}</tbody></table></div>}
      </section>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 13 }}><span className="mono" style={{ color: 'var(--faint)', fontSize: 9 }}>SHOWING {items.length} OF {total}</span><div style={{ display: 'flex', gap: 7 }}><button className="lab-button lab-button-ghost" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}><ChevronLeft size={13} />Previous</button><button className="lab-button lab-button-ghost" disabled={page >= pages} onClick={() => setPage((value) => value + 1)}>Next<ChevronRight size={13} /></button></div></div>
    </div>
  )
}
