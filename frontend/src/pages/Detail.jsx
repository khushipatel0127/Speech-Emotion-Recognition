import React, { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Download,
  Play,
  Pause,
  Loader2,
  FileText,
  Clock,
  HardDrive,
  Calendar,
  AlertCircle,
  Info,
} from 'lucide-react'
import { EMOTION_COLORS, EMOTION_ICONS } from '../components/emotionTheme.js'

export default function Detail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef(null)

  useEffect(() => {
    fetch(`/api/history/${id}`)
      .then((res) => {
        if (res.status === 404) throw new Error('Not found')
        return res.json()
      })
      .then((json) => {
        setData(json)
      })
      .catch((err) => {
        if (err.message === 'Not found') {
          setNotFound(true)
        }
      })
      .finally(() => setLoading(false))
  }, [id])

  const togglePlayAudio = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleDownloadReport = () => {
    const link = document.createElement('a')
    link.href = `/api/report/${id}`
    link.download = `speechsense_report_${id.slice(0, 8)}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  }

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '60vh',
          gap: '14px',
          color: '#64748B',
        }}
      >
        <Loader2 size={36} className="spin" color="#2563EB" />
        <p style={{ fontSize: '14px' }}>Loading analysis result...</p>
      </div>
    )
  }

  if (notFound || !data) {
    return (
      <div className="fade-in" style={{ padding: '20px 0' }}>
        <button
          onClick={() => navigate('/history')}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#2563EB',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            marginBottom: '24px',
          }}
        >
          <ArrowLeft size={16} />
          <span>Back to Analysis History</span>
        </button>

        <div
          style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '16px',
            border: '1px solid #E2E8F0',
            padding: '48px 24px',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              backgroundColor: '#FEF2F2',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
            }}
          >
            <AlertCircle size={28} color="#DC2626" />
          </div>
          <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0F172A', marginBottom: '8px' }}>
            Analysis Record Not Found
          </h3>
          <p style={{ fontSize: '14px', color: '#64748B', marginBottom: '24px' }}>
            The requested prediction ID does not exist or has been deleted from the database.
          </p>
          <button
            onClick={() => navigate('/history')}
            style={{
              backgroundColor: '#2563EB',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '10px',
              padding: '10px 20px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Return to History
          </button>
        </div>
      </div>
    )
  }

  const rawProbs = typeof data.probabilities === 'string'
    ? JSON.parse(data.probabilities)
    : data.probabilities || {}

  const sortedProbs = Object.entries(rawProbs)
    .map(([emotion, prob]) => ({
      emotion: emotion.toLowerCase(),
      score: prob,
      percentage: (prob * 100).toFixed(1),
      color: EMOTION_COLORS[emotion.toLowerCase()] || '#64748B',
      icon: EMOTION_ICONS[emotion.toLowerCase()] || '🎤',
    }))
    .sort((a, b) => b.score - a.score)

  const topEmotion = data.predicted_emotion?.toLowerCase() || ''
  const topScore = sortedProbs.length > 0 ? sortedProbs[0].percentage : '0.0'
  const emotionThemeColor = EMOTION_COLORS[topEmotion] || '#2563EB'
  const dateFormatted = new Date(data.created_at).toLocaleString(undefined, {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

  return (
    <div className="fade-in">
      {/* Top Navigation & Action Bar */}
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
        <button
          onClick={() => navigate('/history')}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#2563EB',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
            padding: '6px 0',
          }}
        >
          <ArrowLeft size={16} />
          <span>Back to Analysis History</span>
        </button>

        <button
          onClick={handleDownloadReport}
          style={{
            backgroundColor: '#2563EB',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '10px',
            padding: '10px 18px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 2px 8px rgba(37, 99, 235, 0.25)',
            transition: 'background-color 0.15s ease',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#1D4ED8')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#2563EB')}
        >
          <Download size={15} />
          <span>Download Analysis Report</span>
        </button>
      </div>

      {/* Grid Layout: Left Column (Emotion + Audio + Metadata) & Right Column (Probabilities) */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1.25fr',
          gap: '24px',
          alignItems: 'start',
        }}
      >
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Main Emotion Result Card */}
          <div
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '16px',
              border: `1px solid ${emotionThemeColor}33`,
              boxShadow: '0 4px 14px rgba(0,0,0,0.04)',
              padding: '28px',
              textAlign: 'center',
              background: `linear-gradient(135deg, ${emotionThemeColor}14 0%, #FFFFFF 100%)`,
            }}
          >
            <div style={{ fontSize: '56px', marginBottom: '8px' }}>
              {EMOTION_ICONS[topEmotion] || '🎤'}
            </div>

            <p style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748B' }}>
              Predicted Emotion
            </p>

            <h2
              style={{
                fontSize: '34px',
                fontWeight: 800,
                color: emotionThemeColor,
                textTransform: 'capitalize',
                margin: '4px 0 10px',
                letterSpacing: '-0.02em',
              }}
            >
              {topEmotion}
            </h2>

            <p style={{ fontSize: '15px', color: '#0F172A', fontWeight: 600 }}>
              Top Model Score:{' '}
              <span style={{ color: emotionThemeColor, fontSize: '18px', fontWeight: 800 }}>
                {topScore}%
              </span>
            </p>

            {Boolean(data.is_uncertain) && (
              <div
                style={{
                  margin: '14px auto 0',
                  maxWidth: '430px',
                  padding: '10px 12px',
                  borderRadius: '10px',
                  border: '1px solid #FDE68A',
                  backgroundColor: '#FFFBEB',
                  color: '#92400E',
                  fontSize: '12px',
                  lineHeight: 1.5,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  textAlign: 'left',
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '1px' }} />
                <span>The top model score is low or close to another emotion. Interpret this result cautiously.</span>
              </div>
            )}
          </div>

          {/* Audio Player Card */}
          <div
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '16px',
              border: '1px solid #E2E8F0',
              boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
              padding: '22px',
            }}
          >
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', marginBottom: '14px' }}>
              Audio Playback
            </h3>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                backgroundColor: '#F8FAFC',
                padding: '12px',
                borderRadius: '12px',
                border: '1px solid #E2E8F0',
              }}
            >
              <button
                onClick={togglePlayAudio}
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: '50%',
                  backgroundColor: '#2563EB',
                  color: '#FFFFFF',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  boxShadow: '0 2px 6px rgba(37, 99, 235, 0.3)',
                }}
              >
                {isPlaying ? <Pause size={16} fill="#FFFFFF" /> : <Play size={16} fill="#FFFFFF" style={{ marginLeft: '2px' }} />}
              </button>

              <audio
                ref={audioRef}
                src={`/audio/${data.id}`}
                onEnded={() => setIsPlaying(false)}
                controls
                style={{
                  flex: 1,
                  height: '36px',
                  outline: 'none',
                }}
              />
            </div>
          </div>

          {/* Metadata Card */}
          <div
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '16px',
              border: '1px solid #E2E8F0',
              boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
              padding: '24px',
            }}
          >
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', marginBottom: '16px' }}>
              Recording Details
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {[
                { icon: FileText, label: 'Filename', value: data.filename },
                { icon: Clock, label: 'Duration', value: `${data.duration_seconds.toFixed(2)} seconds` },
                { icon: HardDrive, label: 'File Size', value: formatFileSize(data.file_size_bytes) },
                { icon: Info, label: 'Model Version', value: data.model_version || 'Historical model (version unavailable)' },
                { icon: Clock, label: 'Processing Time', value: data.processing_time_ms != null ? `${data.processing_time_ms.toFixed(0)} ms` : 'Unavailable for older record' },
                { icon: Calendar, label: 'Analysis Timestamp', value: dateFormatted },
              ].map(({ icon: Icon, label, value }) => (
                <div key={label} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      backgroundColor: '#F1F5F9',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <Icon size={16} color="#64748B" />
                  </div>
                  <div style={{ overflow: 'hidden' }}>
                    <p style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#94A3B8' }}>
                      {label}
                    </p>
                    <p style={{ fontSize: '13px', fontWeight: 600, color: '#0F172A', wordBreak: 'break-all', marginTop: '1px' }}>
                      {value}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Probability Breakdown */}
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
              marginBottom: '22px',
            }}
          >
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A' }}>
                Complete Emotion Probability Distribution
              </h3>
              <p style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                Combined 50/50 Soft Voting of Calibrated SVM & HistGradientBoosting
              </p>
            </div>
          </div>

          {/* Probability Bars */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {sortedProbs.map((item) => (
              <div key={item.emotion}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '13px',
                    marginBottom: '5px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>{item.icon}</span>
                    <span style={{ fontWeight: 600, color: '#0F172A', textTransform: 'capitalize' }}>
                      {item.emotion}
                    </span>
                  </div>
                  <span style={{ fontWeight: 700, color: item.color }}>
                    {item.percentage}%
                  </span>
                </div>

                {/* Bar Track */}
                <div
                  style={{
                    width: '100%',
                    height: '10px',
                    borderRadius: '9999px',
                    backgroundColor: '#F1F5F9',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${item.percentage}%`,
                      height: '100%',
                      backgroundColor: item.color,
                      borderRadius: '9999px',
                      transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Model Information Notice */}
          <div
            style={{
              marginTop: '28px',
              backgroundColor: '#F8FAFC',
              border: '1px solid #E2E8F0',
              borderRadius: '12px',
              padding: '16px',
              display: 'flex',
              gap: '12px',
              fontSize: '12px',
              color: '#64748B',
              lineHeight: 1.6,
            }}
          >
            <Info size={18} color="#64748B" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong style={{ color: '#0F172A' }}>Model Architecture Note:</strong> Feature extraction generated 84 acoustic values from audio resampled to 16,000 Hz mono (40 MFCC means, 40 MFCC standard deviations, RMS energy mean & std, and Zero-Crossing Rate mean & std). Classification uses soft voting over calibrated probabilities.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
