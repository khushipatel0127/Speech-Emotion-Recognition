import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Search,
  Filter,
  Trash2,
  Eye,
  ChevronLeft,
  ChevronRight,
  Loader2,
  AudioWaveform,
  AlertCircle,
  Mic,
} from 'lucide-react'
import EmotionBadge from '../components/EmotionBadge.jsx'

const ALL_EMOTIONS = [
  'angry',
  'calm',
  'disgust',
  'fearful',
  'happy',
  'neutral',
  'sad',
  'surprised',
]

export default function History() {
  const navigate = useNavigate()
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [search, setSearch] = useState('')
  const [selectedEmotion, setSelectedEmotion] = useState('')
  const [loading, setLoading] = useState(true)
  const [showClearModal, setShowClearModal] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [deletingId, setDeletingId] = useState(null)

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    const params = new URLSearchParams()
    params.set('page', page)
    params.set('per_page', 10)
    if (search.trim()) params.set('search', search.trim())
    if (selectedEmotion.trim()) params.set('emotion', selectedEmotion.trim())

    try {
      const res = await fetch(`/api/history?${params.toString()}`)
      const data = await res.json()
      setItems(data.items || [])
      setTotal(data.total || 0)
      setPages(data.pages || 1)
    } catch {
      setItems([])
      setTotal(0)
      setPages(1)
    } finally {
      setLoading(false)
    }
  }, [page, search, selectedEmotion])

  useEffect(() => {
    // Fetching remote history intentionally updates loading and result state.
    // oxlint-disable-next-line react/set-state-in-effect
    fetchHistory()
  }, [fetchHistory])

  const handleClearHistory = async () => {
    setClearing(true)
    try {
      await fetch('/api/history', { method: 'DELETE' })
      setShowClearModal(false)
      setPage(1)
      await fetchHistory()
    } catch (err) {
      console.error('Failed to clear history:', err)
    } finally {
      setClearing(false)
    }
  }

  const handleDeleteEntry = async () => {
    if (!deleteTarget) return
    setDeletingId(deleteTarget.id)
    try {
      const res = await fetch(`/api/history/${deleteTarget.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete request failed')
      setDeleteTarget(null)
      if (items.length === 1 && page > 1) {
        setPage((current) => current - 1)
      } else {
        await fetchHistory()
      }
    } catch (err) {
      console.error('Failed to delete history entry:', err)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="fade-in">
      {/* Page Header & Actions */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '28px',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <h1 style={{ fontSize: '26px', fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em' }}>
            Analysis History
          </h1>
          <p style={{ fontSize: '14px', color: '#64748B', marginTop: '4px' }}>
            {total} total speech prediction record{total !== 1 ? 's' : ''} stored in SQLite
          </p>
        </div>

        {total > 0 && (
          <button
            onClick={() => setShowClearModal(true)}
            style={{
              backgroundColor: '#FEF2F2',
              color: '#DC2626',
              border: '1px solid #FECACA',
              borderRadius: '10px',
              padding: '10px 16px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#DC2626'
              e.currentTarget.style.color = '#FFFFFF'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#FEF2F2'
              e.currentTarget.style.color = '#DC2626'
            }}
          >
            <Trash2 size={16} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Confirmation Modal */}
      {showClearModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
          }}
        >
          <div
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '16px',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.2)',
              maxWidth: '420px',
              width: '100%',
              padding: '28px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                backgroundColor: '#FEF2F2',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
              }}
            >
              <AlertCircle size={26} color="#DC2626" />
            </div>

            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0F172A', marginBottom: '8px' }}>
              Clear Entire Analysis History?
            </h3>

            <p style={{ fontSize: '13px', color: '#64748B', lineHeight: 1.5, marginBottom: '24px' }}>
              This will permanently delete all <strong>{total}</strong> prediction records and their stored audio files from the SQLite database. This action cannot be reversed.
            </p>

            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={() => setShowClearModal(false)}
                disabled={clearing}
                style={{
                  flex: 1,
                  padding: '11px',
                  borderRadius: '10px',
                  border: '1px solid #CBD5E1',
                  backgroundColor: '#FFFFFF',
                  color: '#475569',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleClearHistory}
                disabled={clearing}
                style={{
                  flex: 1,
                  padding: '11px',
                  borderRadius: '10px',
                  border: 'none',
                  backgroundColor: '#DC2626',
                  color: '#FFFFFF',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: clearing ? 'not-allowed' : 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                {clearing ? <Loader2 size={16} className="spin" /> : <Trash2 size={16} />}
                <span>{clearing ? 'Clearing...' : 'Delete All'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-entry-title"
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '16px',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.2)',
              maxWidth: '420px',
              width: '100%',
              padding: '28px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                backgroundColor: '#FEF2F2',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
              }}
            >
              <Trash2 size={25} color="#DC2626" />
            </div>
            <h3
              id="delete-entry-title"
              style={{ fontSize: '18px', fontWeight: 700, color: '#0F172A', marginBottom: '8px' }}
            >
              Delete This Analysis?
            </h3>
            <p style={{ fontSize: '13px', color: '#64748B', lineHeight: 1.5, marginBottom: '24px' }}>
              The record and stored audio for <strong>{deleteTarget.filename}</strong> will be permanently deleted.
            </p>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={Boolean(deletingId)}
                style={{
                  flex: 1,
                  padding: '11px',
                  borderRadius: '10px',
                  border: '1px solid #CBD5E1',
                  backgroundColor: '#FFFFFF',
                  color: '#475569',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteEntry}
                disabled={Boolean(deletingId)}
                style={{
                  flex: 1,
                  padding: '11px',
                  borderRadius: '10px',
                  border: 'none',
                  backgroundColor: '#DC2626',
                  color: '#FFFFFF',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: deletingId ? 'not-allowed' : 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                {deletingId ? <Loader2 size={16} className="spin" /> : <Trash2 size={16} />}
                <span>{deletingId ? 'Deleting...' : 'Delete Analysis'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '14px',
          border: '1px solid #E2E8F0',
          padding: '16px 20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
          marginBottom: '20px',
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {/* Search input */}
        <div style={{ flex: 1, minWidth: '240px', position: 'relative' }}>
          <Search
            size={18}
            color="#94A3B8"
            style={{
              position: 'absolute',
              left: '14px',
              top: '50%',
              transform: 'translateY(-50%)',
              pointerEvents: 'none',
            }}
          />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setPage(1)
            }}
            placeholder="Search audio recording by filename..."
            style={{
              width: '100%',
              padding: '10px 14px 10px 42px',
              borderRadius: '10px',
              border: '1px solid #E2E8F0',
              fontSize: '14px',
              outline: 'none',
              backgroundColor: '#F8FAFC',
              boxSizing: 'border-box',
              transition: 'border-color 0.15s ease',
            }}
            onFocus={(e) => (e.target.style.borderColor = '#2563EB')}
            onBlur={(e) => (e.target.style.borderColor = '#E2E8F0')}
          />
        </div>

        {/* Emotion dropdown */}
        <div style={{ minWidth: '180px', position: 'relative' }}>
          <Filter
            size={16}
            color="#94A3B8"
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)',
              pointerEvents: 'none',
            }}
          />
          <select
            value={selectedEmotion}
            onChange={(e) => {
              setSelectedEmotion(e.target.value)
              setPage(1)
            }}
            style={{
              width: '100%',
              padding: '10px 14px 10px 36px',
              borderRadius: '10px',
              border: '1px solid #E2E8F0',
              fontSize: '13px',
              fontWeight: 500,
              outline: 'none',
              backgroundColor: '#F8FAFC',
              cursor: 'pointer',
              color: '#0F172A',
            }}
          >
            <option value="">All Emotions</option>
            {ALL_EMOTIONS.map((em) => (
              <option key={em} value={em}>
                {em.charAt(0).toUpperCase() + em.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {(search || selectedEmotion) && (
          <button
            onClick={() => {
              setSearch('')
              setSelectedEmotion('')
              setPage(1)
            }}
            style={{
              backgroundColor: '#F1F5F9',
              border: 'none',
              color: '#64748B',
              padding: '9px 14px',
              borderRadius: '8px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* History Table Container */}
      <div
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '16px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          overflow: 'hidden',
        }}
      >
        {loading ? (
          <div
            style={{
              padding: '64px 0',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              color: '#94A3B8',
            }}
          >
            <Loader2 size={32} className="spin" color="#2563EB" />
            <span style={{ fontSize: '14px' }}>Loading history records...</span>
          </div>
        ) : items.length === 0 ? (
          /* Empty / No Results */
          <div
            style={{
              padding: '64px 24px',
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
                backgroundColor: '#F8FAFC',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}
            >
              <AudioWaveform size={28} color="#94A3B8" />
            </div>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A', marginBottom: '6px' }}>
              No prediction records found
            </h3>
            <p style={{ fontSize: '13px', color: '#64748B', maxWidth: '360px', marginBottom: '20px' }}>
              {search || selectedEmotion
                ? 'Try adjusting your search query or emotion filter.'
                : 'Upload audio on the Analyze Speech page to start recording history.'}
            </p>
            {!search && !selectedEmotion && (
              <button
                onClick={() => navigate('/analyze')}
                style={{
                  backgroundColor: '#2563EB',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '10px 20px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Mic size={16} />
                <span>Analyze Audio</span>
              </button>
            )}
          </div>
        ) : (
          /* Table of results */
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                <thead>
                  <tr style={{ backgroundColor: '#F8FAFC', borderBottom: '1px solid #E2E8F0', color: '#64748B' }}>
                    <th style={{ padding: '14px 20px', fontWeight: 600 }}>Filename</th>
                    <th style={{ padding: '14px 20px', fontWeight: 600 }}>Predicted Emotion</th>
                    <th style={{ padding: '14px 20px', fontWeight: 600 }}>Top Probability</th>
                    <th style={{ padding: '14px 20px', fontWeight: 600 }}>Duration</th>
                    <th style={{ padding: '14px 20px', fontWeight: 600 }}>Analyzed Date</th>
                    <th style={{ padding: '14px 20px', fontWeight: 600, textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, index) => {
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
                          borderBottom: index < items.length - 1 ? '1px solid #F1F5F9' : 'none',
                          transition: 'background-color 0.15s ease',
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#F8FAFC')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        <td style={{ padding: '16px 20px', fontWeight: 600, color: '#0F172A' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Mic size={16} color="#64748B" />
                            <span
                              style={{
                                maxWidth: '240px',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                              title={item.filename}
                            >
                              {item.filename}
                            </span>
                          </div>
                        </td>
                        <td style={{ padding: '16px 20px' }}>
                          <EmotionBadge emotion={item.predicted_emotion} showIcon={true} />
                        </td>
                        <td style={{ padding: '16px 20px', fontWeight: 700, color: '#0F172A' }}>
                          {(topScore * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '16px 20px', color: '#64748B' }}>
                          {item.duration_seconds.toFixed(1)}s
                        </td>
                        <td style={{ padding: '16px 20px', color: '#64748B', whiteSpace: 'nowrap' }}>
                          {dateStr}
                        </td>
                        <td style={{ padding: '16px 20px', textAlign: 'right' }}>
                          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                          <button
                            onClick={() => navigate(`/history/${item.id}`)}
                            style={{
                              backgroundColor: '#EFF6FF',
                              color: '#2563EB',
                              border: '1px solid #BFDBFE',
                              borderRadius: '8px',
                              padding: '7px 14px',
                              fontSize: '12px',
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
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
                            <Eye size={14} />
                            <span>View Result</span>
                          </button>
                          <button
                            onClick={() => setDeleteTarget(item)}
                            aria-label={`Delete analysis for ${item.filename}`}
                            title="Delete analysis"
                            style={{
                              backgroundColor: '#FEF2F2',
                              color: '#DC2626',
                              border: '1px solid #FECACA',
                              borderRadius: '8px',
                              width: '34px',
                              height: '32px',
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Trash2 size={14} />
                          </button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination footer */}
            {pages > 1 && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '16px 20px',
                  borderTop: '1px solid #E2E8F0',
                  backgroundColor: '#FFFFFF',
                  fontSize: '13px',
                  color: '#64748B',
                }}
              >
                <div>
                  Page <strong>{page}</strong> of <strong>{pages}</strong>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    style={{
                      padding: '7px 12px',
                      borderRadius: '8px',
                      border: '1px solid #E2E8F0',
                      backgroundColor: page === 1 ? '#F8FAFC' : '#FFFFFF',
                      color: page === 1 ? '#CBD5E1' : '#475569',
                      cursor: page === 1 ? 'not-allowed' : 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      fontWeight: 600,
                      fontSize: '12px',
                    }}
                  >
                    <ChevronLeft size={14} />
                    <span>Previous</span>
                  </button>

                  <button
                    onClick={() => setPage((p) => Math.min(pages, p + 1))}
                    disabled={page === pages}
                    style={{
                      padding: '7px 12px',
                      borderRadius: '8px',
                      border: '1px solid #E2E8F0',
                      backgroundColor: page === pages ? '#F8FAFC' : '#FFFFFF',
                      color: page === pages ? '#CBD5E1' : '#475569',
                      cursor: page === pages ? 'not-allowed' : 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      fontWeight: 600,
                      fontSize: '12px',
                    }}
                  >
                    <span>Next</span>
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
