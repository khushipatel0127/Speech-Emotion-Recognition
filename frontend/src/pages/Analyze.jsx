import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Upload,
  Mic,
  X,
  Play,
  Pause,
  Square,
  Radio,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Info,
  AudioWaveform,
  Volume2,
  FileAudio,
  History,
  RotateCcw,
} from 'lucide-react'
import EmotionBadge, { EMOTION_COLORS, EMOTION_ICONS } from '../components/EmotionBadge.jsx'

const SUPPORTED_EXTS = ['.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.webm', '.aac', '.flac']
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024 // 25 MB

export default function Analyze() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('upload') // 'upload' | 'record'
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [audioDuration, setAudioDuration] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // In-browser recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const recordingTimerRef = useRef(null)

  const audioRef = useRef(null)
  const fileInputRef = useRef(null)

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  }

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const validateAndSelectFile = useCallback((selectedFile) => {
    setError(null)
    setResult(null)
    setIsPlaying(false)

    if (!selectedFile) return

    const ext = '.' + selectedFile.name.split('.').pop().toLowerCase()
    const isSupported = SUPPORTED_EXTS.includes(ext) || selectedFile.type.startsWith('audio/') || selectedFile.type === 'video/mp4'

    if (!isSupported) {
      setError(`Unsupported format. Supported formats: WAV, MP3, M4A, MP4, OGG, WebM.`)
      return
    }

    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setError('File size limit exceeded: Audio file must be under 25 MB.')
      return
    }

    if (selectedFile.size === 0) {
      setError('Empty file: The selected audio file has 0 bytes.')
      return
    }

    setFile(selectedFile)
    const url = URL.createObjectURL(selectedFile)
    setAudioUrl(url)
  }, [])

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0])
    }
  }

  const handleRemoveFile = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
    }
    setFile(null)
    setAudioUrl(null)
    setAudioDuration(null)
    setResult(null)
    setError(null)
    setIsPlaying(false)
    if (audioRef.current) {
      audioRef.current.pause()
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const togglePlayAudio = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleAudioLoadedMetadata = () => {
    if (audioRef.current) {
      setAudioDuration(audioRef.current.duration)
    }
  }

  const handleAudioEnded = () => {
    setIsPlaying(false)
  }

  // --- Browser Microphone Recording Controls ---
  const startRecording = async () => {
    setError(null)
    setResult(null)
    handleRemoveFile()

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone access is not supported by your browser.')
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []

      // Preferred mime types
      let mimeType = ''
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus'
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm'
      } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
        mimeType = 'audio/ogg'
      }

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })

        // Determine extension
        const ext = recorder.mimeType.includes('ogg') ? '.ogg' : '.webm'
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
        const recordedFile = new File([audioBlob], `mic_recording_${timestamp}${ext}`, {
          type: audioBlob.type,
        })

        setFile(recordedFile)
        const url = URL.createObjectURL(recordedFile)
        setAudioUrl(url)

        // Stop all audio tracks to release microphone
        stream.getTracks().forEach((track) => track.stop())
      }

      recorder.start(100) // 100ms chunking
      setIsRecording(true)
      setRecordingSeconds(0)

      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 179) {
            // Auto stop at 3 minutes
            stopRecording()
            return 180
          }
          return prev + 1
        })
      }, 1000)
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Microphone permission was denied. Please allow microphone access in your browser settings.')
      } else {
        setError(`Could not access microphone: ${err.message}`)
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
    setIsRecording(false)
  }

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to predict emotion from audio.')
      }

      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred during prediction.')
    } finally {
      setLoading(false)
    }
  }

  // Sorted probability scores
  const sortedProbabilities = result?.probabilities
    ? Object.entries(result.probabilities)
        .map(([emotion, prob]) => ({
          emotion: emotion.toLowerCase(),
          score: prob,
          percentage: (prob * 100).toFixed(1),
          color: EMOTION_COLORS[emotion.toLowerCase()] || '#64748B',
          icon: EMOTION_ICONS[emotion.toLowerCase()] || '🎤',
        }))
        .sort((a, b) => b.score - a.score)
    : []

  const topScore = sortedProbabilities.length > 0 ? sortedProbabilities[0].percentage : '0.0'
  const predictedEmotion = result?.predicted_emotion?.toLowerCase() || ''
  const emotionThemeColor = EMOTION_COLORS[predictedEmotion] || '#2563EB'

  return (
    <div className="fade-in">
      {/* Page Header */}
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '26px', fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em' }}>
          Analyze Speech
        </h1>
        <p style={{ fontSize: '14px', color: '#64748B', marginTop: '4px' }}>
          Upload an audio or video recording (WAV, MP3, M4A, MP4, WebM) or record directly with your microphone.
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: result ? '1fr 1.15fr' : '1fr',
          gap: '28px',
          alignItems: 'start',
        }}
      >
        {/* Input Card (Tabs for File Upload vs Microphone) */}
        <div
          style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '16px',
            border: '1px solid #E2E8F0',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            padding: '24px',
          }}
        >
          {/* Mode Switcher Tabs */}
          <div
            style={{
              display: 'flex',
              gap: '8px',
              backgroundColor: '#F1F5F9',
              padding: '4px',
              borderRadius: '12px',
              marginBottom: '20px',
            }}
          >
            <button
              onClick={() => {
                if (isRecording) stopRecording()
                setActiveTab('upload')
              }}
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: activeTab === 'upload' ? '#FFFFFF' : 'transparent',
                color: activeTab === 'upload' ? '#0F172A' : '#64748B',
                fontWeight: activeTab === 'upload' ? 700 : 500,
                fontSize: '13px',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: activeTab === 'upload' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                transition: 'all 0.15s ease',
              }}
            >
              <Upload size={16} />
              <span>Upload Audio / Video</span>
            </button>

            <button
              onClick={() => {
                handleRemoveFile()
                setActiveTab('record')
              }}
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: activeTab === 'record' ? '#FFFFFF' : 'transparent',
                color: activeTab === 'record' ? '#0F172A' : '#64748B',
                fontWeight: activeTab === 'record' ? 700 : 500,
                fontSize: '13px',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: activeTab === 'record' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                transition: 'all 0.15s ease',
              }}
            >
              <Mic size={16} color={activeTab === 'record' ? '#DC2626' : undefined} />
              <span>Record Microphone</span>
            </button>
          </div>

          {/* TAB 1: FILE UPLOAD */}
          {activeTab === 'upload' && !file && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? '#2563EB' : '#CBD5E1'}`,
                borderRadius: '14px',
                backgroundColor: isDragging ? '#EFF6FF' : '#F8FAFC',
                padding: '40px 20px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                if (!isDragging) {
                  e.currentTarget.style.borderColor = '#94A3B8'
                  e.currentTarget.style.backgroundColor = '#F1F5F9'
                }
              }}
              onMouseLeave={(e) => {
                if (!isDragging) {
                  e.currentTarget.style.borderColor = '#CBD5E1'
                  e.currentTarget.style.backgroundColor = '#F8FAFC'
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.m4a,.mp4,.ogg,.webm,.aac,.flac,audio/*,video/mp4"
                style={{ display: 'none' }}
                onChange={handleFileInputChange}
              />
              <div
                style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '50%',
                  backgroundColor: '#EFF6FF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 16px',
                  boxShadow: '0 2px 8px rgba(37, 99, 235, 0.15)',
                }}
              >
                <Upload size={24} color="#2563EB" />
              </div>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', marginBottom: '6px' }}>
                Drag & drop your voice recording here
              </h3>
              <p style={{ fontSize: '13px', color: '#64748B', marginBottom: '18px' }}>
                or click to browse your files
              </p>
              <span
                style={{
                  display: 'inline-block',
                  backgroundColor: '#2563EB',
                  color: '#FFFFFF',
                  padding: '9px 20px',
                  borderRadius: '10px',
                  fontSize: '13px',
                  fontWeight: 600,
                  boxShadow: '0 2px 8px rgba(37, 99, 235, 0.25)',
                }}
              >
                Browse Files
              </span>
              <p style={{ fontSize: '11px', color: '#64748B', marginTop: '16px', lineHeight: 1.4 }}>
                <strong>Supported formats:</strong> WAV, MP3, M4A, MP4 (voice & video), OGG, WebM<br />
                Max size: 25 MB · Automatic FFmpeg 16 kHz conversion
              </p>
            </div>
          )}

          {/* TAB 2: IN-BROWSER MICROPHONE RECORDER */}
          {activeTab === 'record' && !file && (
            <div
              style={{
                borderRadius: '14px',
                backgroundColor: '#F8FAFC',
                border: '1px solid #E2E8F0',
                padding: '40px 24px',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  width: '72px',
                  height: '72px',
                  borderRadius: '50%',
                  backgroundColor: isRecording ? '#FEF2F2' : '#EFF6FF',
                  border: isRecording ? '3px solid #EF4444' : '3px solid #BFDBFE',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 18px',
                  transition: 'all 0.3s ease',
                  boxShadow: isRecording ? '0 0 20px rgba(239, 68, 68, 0.4)' : 'none',
                }}
              >
                <Mic size={32} color={isRecording ? '#DC2626' : '#2563EB'} />
              </div>

              {isRecording ? (
                <div>
                  <div
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px',
                      backgroundColor: '#FEF2F2',
                      color: '#DC2626',
                      padding: '4px 12px',
                      borderRadius: '9999px',
                      fontSize: '12px',
                      fontWeight: 700,
                      marginBottom: '12px',
                    }}
                  >
                    <span
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: '#DC2626',
                        animation: 'spin 1.5s infinite',
                      }}
                    />
                    <span>RECORDING LIVE SPEECH...</span>
                  </div>

                  <div style={{ fontSize: '32px', fontWeight: 800, color: '#0F172A', fontVariantNumeric: 'tabular-nums', marginBottom: '16px' }}>
                    {formatTime(recordingSeconds)}{' '}
                    <span style={{ fontSize: '14px', color: '#94A3B8', fontWeight: 500 }}>/ 03:00 max</span>
                  </div>

                  <button
                    onClick={stopRecording}
                    style={{
                      backgroundColor: '#DC2626',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '12px',
                      padding: '12px 28px',
                      fontSize: '14px',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px',
                      boxShadow: '0 4px 12px rgba(220, 38, 38, 0.3)',
                    }}
                  >
                    <Square size={16} fill="#FFFFFF" />
                    <span>Stop Recording</span>
                  </button>
                </div>
              ) : (
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A', marginBottom: '6px' }}>
                    Speak into your microphone
                  </h3>
                  <p style={{ fontSize: '13px', color: '#64748B', maxWidth: '320px', margin: '0 auto 20px' }}>
                    Record your voice directly inside the browser. SpeechSense will convert your audio and predict your emotion.
                  </p>

                  <button
                    onClick={startRecording}
                    style={{
                      backgroundColor: '#2563EB',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '12px',
                      padding: '12px 28px',
                      fontSize: '14px',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '8px',
                      boxShadow: '0 4px 12px rgba(37, 99, 235, 0.25)',
                    }}
                  >
                    <Mic size={16} />
                    <span>Start Recording</span>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* AUDIO PREVIEW & PREDICTION TRIGGER */}
          {file && (
            <div>
              <div
                style={{
                  backgroundColor: '#F8FAFC',
                  border: '1px solid #E2E8F0',
                  borderRadius: '14px',
                  padding: '18px',
                  marginBottom: '20px',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                    gap: '12px',
                    marginBottom: '16px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div
                      style={{
                        width: '44px',
                        height: '44px',
                        borderRadius: '10px',
                        backgroundColor: '#EFF6FF',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <FileAudio size={22} color="#2563EB" />
                    </div>
                    <div>
                      <p
                        style={{
                          fontSize: '14px',
                          fontWeight: 700,
                          color: '#0F172A',
                          wordBreak: 'break-all',
                        }}
                      >
                        {file.name}
                      </p>
                      <p style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>
                        {formatFileSize(file.size)}
                        {audioDuration !== null && ` · ${audioDuration.toFixed(1)}s duration`}
                        <span style={{ marginLeft: '6px', color: '#2563EB', fontWeight: 600 }}>
                          ({file.name.split('.').pop().toUpperCase()})
                        </span>
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={handleRemoveFile}
                    title="Remove file"
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: '6px',
                      borderRadius: '6px',
                      color: '#94A3B8',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#EF4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = '#94A3B8')}
                  >
                    <X size={18} />
                  </button>
                </div>

                {/* Player Controls */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    backgroundColor: '#FFFFFF',
                    padding: '10px 14px',
                    borderRadius: '10px',
                    border: '1px solid #E2E8F0',
                  }}
                >
                  <button
                    onClick={togglePlayAudio}
                    style={{
                      width: '36px',
                      height: '36px',
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
                    src={audioUrl}
                    onLoadedMetadata={handleAudioLoadedMetadata}
                    onEnded={handleAudioEnded}
                    controls
                    style={{
                      flex: 1,
                      height: '34px',
                      outline: 'none',
                    }}
                  />
                </div>
              </div>

              {/* Predict Button */}
              <button
                onClick={handlePredict}
                disabled={loading}
                style={{
                  width: '100%',
                  backgroundColor: loading ? '#93C5FD' : '#2563EB',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '12px',
                  padding: '14px',
                  fontSize: '15px',
                  fontWeight: 700,
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  boxShadow: '0 4px 12px rgba(37, 99, 235, 0.25)',
                  transition: 'background-color 0.2s ease',
                }}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="spin" />
                    <span>Converting with FFmpeg & Predicting Emotion...</span>
                  </>
                ) : (
                  <>
                    <Mic size={18} />
                    <span>Predict Emotion</span>
                  </>
                )}
              </button>

              <button
                onClick={handleRemoveFile}
                disabled={loading}
                style={{
                  width: '100%',
                  marginTop: '10px',
                  backgroundColor: 'transparent',
                  color: '#64748B',
                  border: '1px solid #E2E8F0',
                  borderRadius: '10px',
                  padding: '10px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                <RotateCcw size={14} />
                <span>Choose or Record Different Audio</span>
              </button>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div
              style={{
                marginTop: '20px',
                backgroundColor: '#FEF2F2',
                border: '1px solid #FECACA',
                borderRadius: '12px',
                padding: '14px 16px',
                display: 'flex',
                gap: '10px',
                color: '#B91C1C',
                fontSize: '13px',
                lineHeight: 1.5,
              }}
            >
              <AlertCircle size={18} color="#DC2626" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Error:</strong> {error}
              </div>
            </div>
          )}
        </div>

        {/* Prediction Results Display */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Predicted Emotion Card */}
            <div
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '16px',
                border: `1px solid ${emotionThemeColor}33`,
                boxShadow: '0 4px 14px rgba(0,0,0,0.05)',
                padding: '28px',
                textAlign: 'center',
                background: `linear-gradient(135deg, ${emotionThemeColor}12 0%, #FFFFFF 100%)`,
                position: 'relative',
              }}
            >
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: '#ECFDF5',
                  color: '#059669',
                  border: '1px solid #A7F3D0',
                  borderRadius: '9999px',
                  padding: '4px 12px',
                  fontSize: '12px',
                  fontWeight: 600,
                  marginBottom: '16px',
                }}
              >
                <CheckCircle2 size={14} color="#059669" />
                Prediction Complete
              </div>

              <div style={{ fontSize: '56px', marginBottom: '8px' }}>
                {EMOTION_ICONS[predictedEmotion] || '🎤'}
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
                {predictedEmotion}
              </h2>

              <p style={{ fontSize: '15px', color: '#0F172A', fontWeight: 600 }}>
                Ensemble Confidence:{' '}
                <span style={{ color: emotionThemeColor, fontSize: '18px', fontWeight: 800 }}>
                  {topScore}%
                </span>
              </p>
            </div>

            {/* Probability Breakdown Card */}
            <div
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '16px',
                border: '1px solid #E2E8F0',
                boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                padding: '24px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '18px',
                }}
              >
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A' }}>
                  Emotion Probability Scores
                </h3>
                <span style={{ fontSize: '12px', color: '#64748B' }}>
                  8 Emotions (Soft-Voting)
                </span>
              </div>

              {/* Horizontal Probability Bars */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {sortedProbabilities.map((item) => (
                  <div key={item.emotion}>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: '13px',
                        marginBottom: '4px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span>{item.icon}</span>
                        <span style={{ fontWeight: 600, color: '#0F172A', textTransform: 'capitalize' }}>
                          {item.emotion}
                        </span>
                      </div>
                      <span style={{ fontWeight: 700, color: item.color }}>
                        {item.percentage}%
                      </span>
                    </div>

                    {/* Progress track */}
                    <div
                      style={{
                        width: '100%',
                        height: '9px',
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
                          transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Disclaimer */}
              <div
                style={{
                  marginTop: '20px',
                  backgroundColor: '#F8FAFC',
                  border: '1px solid #E2E8F0',
                  borderRadius: '10px',
                  padding: '12px',
                  display: 'flex',
                  gap: '10px',
                  fontSize: '12px',
                  color: '#64748B',
                  lineHeight: 1.5,
                }}
              >
                <Info size={16} color="#64748B" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>
                  <strong>Disclaimer:</strong> Predictions are statistical estimates calculated by a machine learning model based on acoustic features and may not reflect an individual's true emotional state.
                </span>
              </div>

              {/* View in History Button */}
              <button
                onClick={() => navigate(`/history/${result.id}`)}
                style={{
                  marginTop: '16px',
                  width: '100%',
                  backgroundColor: '#EFF6FF',
                  color: '#2563EB',
                  border: '1px solid #BFDBFE',
                  borderRadius: '10px',
                  padding: '11px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
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
                <History size={16} />
                <span>View Full Details & Download Report</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
