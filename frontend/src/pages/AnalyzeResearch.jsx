import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, FileAudio, History, Loader2, Mic2, Pause, Play, RotateCcw, Square, Upload, X } from 'lucide-react'
import AudioWaveform from '../components/AudioWaveform.jsx'
import { EMOTION_COLORS } from '../components/emotionTheme.js'

const SUPPORTED_EXTS = ['.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.webm', '.aac', '.flac']
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024

function formatSize(bytes) {
  if (!bytes) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export default function AnalyzeResearch() {
  const navigate = useNavigate()
  const [mode, setMode] = useState('upload')
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [duration, setDuration] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [recording, setRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const audioRef = useRef(null)
  const inputRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)

  const clearSelection = useCallback(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    if (audioRef.current) audioRef.current.pause()
    if (inputRef.current) inputRef.current.value = ''
    setFile(null)
    setAudioUrl(null)
    setDuration(null)
    setPlaying(false)
    setResult(null)
    setError(null)
  }, [audioUrl])

  const selectFile = useCallback((selected) => {
    setError(null)
    setResult(null)
    setPlaying(false)
    if (!selected) return
    const ext = `.${selected.name.split('.').pop().toLowerCase()}`
    const supported = SUPPORTED_EXTS.includes(ext) || selected.type.startsWith('audio/') || selected.type === 'video/mp4'
    if (!supported) { setError('Unsupported media. Use WAV, MP3, M4A, MP4, OGG, WebM, AAC or FLAC.'); return }
    if (selected.size > MAX_UPLOAD_BYTES) { setError('The selected file exceeds the 25 MB upload limit.'); return }
    if (selected.size === 0) { setError('The selected file is empty.'); return }
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setDuration(null)
    setFile(selected)
    setAudioUrl(URL.createObjectURL(selected))
  }, [audioUrl])

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop()
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setRecording(false)
  }, [])

  const startRecording = async () => {
    clearSelection()
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone recording is unavailable in this browser.')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      let mimeType = ''
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mimeType = 'audio/webm;codecs=opus'
      else if (MediaRecorder.isTypeSupported('audio/webm')) mimeType = 'audio/webm'
      else if (MediaRecorder.isTypeSupported('audio/ogg')) mimeType = 'audio/ogg'
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => { if (event.data?.size) chunksRef.current.push(event.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const ext = recorder.mimeType.includes('ogg') ? '.ogg' : '.webm'
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
        const recorded = new File([blob], `mic_recording_${stamp}${ext}`, { type: blob.type })
        setFile(recorded)
        setAudioUrl(URL.createObjectURL(recorded))
        stream.getTracks().forEach((track) => track.stop())
      }
      recorder.start(100)
      setRecordingSeconds(0)
      setRecording(true)
      timerRef.current = setInterval(() => setRecordingSeconds((value) => {
        if (value >= 179) { stopRecording(); return 180 }
        return value + 1
      }), 1000)
    } catch (caught) {
      setError(caught.name === 'NotAllowedError' ? 'Microphone permission was denied.' : caught.message)
    }
  }

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (recorderRef.current?.state !== 'inactive') recorderRef.current?.stop()
  }, [])

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'The server could not analyze this recording.')
      setResult(data)
      if (Number.isFinite(data.duration_seconds) && data.duration_seconds > 0) setDuration(data.duration_seconds)
    } catch (caught) {
      setError(caught.message || 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const scores = result?.probabilities || result?.emotion_scores || {}
  const ranked = Object.entries(scores).map(([emotion, score]) => ({ emotion: emotion.toLowerCase(), score, color: EMOTION_COLORS[emotion.toLowerCase()] || '#65747a' })).sort((a, b) => b.score - a.score)
  const prediction = result?.predicted_emotion?.toLowerCase()
  const predictionColor = EMOTION_COLORS[prediction] || 'var(--accent)'

  return (
    <div className="research-page">
      <header className="research-heading">
        <div>
          <div className="research-kicker">Acquisition and inference</div>
          <h1 className="research-title">Analyze speech</h1>
          <p className="research-description">Provide one clear speech segment. Audio is converted to 16 kHz mono before model inference.</p>
        </div>
        <div className="research-note">SUPPORTED: WAV / MP3 / M4A / MP4 / OGG / WEBM<br />MAXIMUM: 25 MB / 20 SEC FOR PHASE 2</div>
      </header>

      {error && <div className="lab-alert lab-alert-error" style={{ marginBottom: 18 }}><AlertCircle size={17} /><span>{error}</span></div>}

      <div className="analysis-grid-research" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, .9fr) minmax(0, 1.1fr)', gap: 18, alignItems: 'start' }}>
        <section className="lab-panel">
          <div className="lab-panel-header">
            <div><div className="lab-panel-title">01 / Audio source</div><div className="lab-panel-meta">Select file or capture microphone</div></div>
            <div style={{ display: 'flex', border: '1px solid var(--line)', borderRadius: 5, padding: 3, background: 'var(--paper)' }}>
              <button onClick={() => { setMode('upload'); clearSelection() }} className="lab-button" style={{ minHeight: 31, padding: '0 11px', border: 0, background: mode === 'upload' ? 'white' : 'transparent', boxShadow: mode === 'upload' ? '0 0 0 1px var(--line-dark)' : 'none' }}><Upload size={13} />Upload</button>
              <button onClick={() => { setMode('record'); clearSelection() }} className="lab-button" style={{ minHeight: 31, padding: '0 11px', border: 0, background: mode === 'record' ? 'white' : 'transparent', boxShadow: mode === 'record' ? '0 0 0 1px var(--line-dark)' : 'none' }}><Mic2 size={13} />Record</button>
            </div>
          </div>
          <div className="lab-panel-body">
            {!file && mode === 'upload' && (
              <button type="button" onClick={() => inputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files?.[0]) }} style={{ width: '100%', minHeight: 192, display: 'grid', placeItems: 'center', padding: 24, cursor: 'pointer', color: 'var(--ink-2)', background: dragging ? '#eaf5f2' : '#f8f7f3', border: `1px ${dragging ? 'solid' : 'dashed'} ${dragging ? 'var(--accent)' : 'var(--line-dark)'}`, borderRadius: 5 }}>
                <div><Upload size={24} color="var(--accent)" /><div style={{ marginTop: 12, fontSize: 13, fontWeight: 750 }}>Choose or drop a recording</div><div className="mono" style={{ marginTop: 7, color: 'var(--faint)', fontSize: 9 }}>AUDIO OR VIDEO WITH AN AUDIO STREAM</div></div>
              </button>
            )}
            <input ref={inputRef} type="file" hidden accept="audio/*,video/mp4,video/webm,.wav,.mp3,.m4a,.mp4,.ogg,.webm,.aac,.flac" onChange={(event) => selectFile(event.target.files?.[0])} />

            {!file && mode === 'record' && (
              <div style={{ minHeight: 192, display: 'grid', placeItems: 'center', textAlign: 'center', border: '1px dashed var(--line-dark)', borderRadius: 5, background: recording ? '#fff8f4' : '#f8f7f3' }}>
                {recording ? <div><div className="mono" style={{ color: '#b64b42', fontSize: 28, letterSpacing: '-.04em' }}>{formatTime(recordingSeconds)}</div><div className="eyebrow" style={{ marginTop: 8, color: '#b64b42' }}>Recording in progress</div><button className="lab-button lab-button-secondary" onClick={stopRecording} style={{ marginTop: 18 }}><Square size={13} fill="currentColor" />Stop recording</button></div> : <div><Mic2 size={25} color="var(--accent)" /><div style={{ marginTop: 12, fontSize: 13, fontWeight: 750 }}>Capture microphone speech</div><div className="mono" style={{ marginTop: 7, color: 'var(--faint)', fontSize: 9 }}>RECOMMENDED: 3–8 SECONDS OF CLEAR SPEECH</div><button className="lab-button lab-button-primary" onClick={startRecording} style={{ marginTop: 18 }}><Mic2 size={14} />Start recording</button></div>}
              </div>
            )}

            {file && (
              <div style={{ display: 'grid', gap: 14 }}>
                <AudioWaveform file={file} />
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, paddingBottom: 14, borderBottom: '1px solid var(--line)' }}>
                  <button className="lab-button lab-button-secondary" onClick={() => { if (!audioRef.current) return; if (playing) audioRef.current.pause(); else audioRef.current.play(); setPlaying(!playing) }} style={{ minWidth: 40, padding: 0 }}>{playing ? <Pause size={14} /> : <Play size={14} />}</button>
                  <FileAudio size={17} color="var(--accent)" />
                  <div style={{ minWidth: 0, flex: 1 }}><div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, fontWeight: 700 }}>{file.name}</div><div className="mono" style={{ marginTop: 3, color: 'var(--faint)', fontSize: 9 }}>{formatSize(file.size)}{Number.isFinite(duration) ? ` / ${duration.toFixed(1)} SEC` : ''}</div></div>
                  <button className="lab-button lab-button-ghost" onClick={clearSelection} style={{ minWidth: 34, minHeight: 34, padding: 0 }} title="Remove recording"><X size={14} /></button>
                </div>
                <audio ref={audioRef} src={audioUrl || undefined} onLoadedMetadata={() => { const value = audioRef.current?.duration; setDuration(Number.isFinite(value) && value > 0 ? value : null) }} onEnded={() => setPlaying(false)} />
                <button className="lab-button lab-button-primary" onClick={analyze} disabled={loading}>{loading ? <><Loader2 size={14} className="spin" />Extracting speech representation…</> : <><Mic2 size={14} />Run emotion analysis</>}</button>
                <button className="lab-button lab-button-ghost" onClick={clearSelection}><RotateCcw size={13} />Choose another recording</button>
              </div>
            )}
          </div>
        </section>

        <section className="lab-panel">
          <div className="lab-panel-header"><div><div className="lab-panel-title">02 / Model output</div><div className="lab-panel-meta">Six-class probability distribution</div></div>{result && <span className="mono" style={{ color: 'var(--accent)', fontSize: 9 }}>INFERENCE COMPLETE</span>}</div>
          {!result ? (
            <div className="lab-panel-body" style={{ minHeight: 390, display: 'grid', placeItems: 'center', textAlign: 'center' }}>
              <div style={{ maxWidth: 330 }}><AudioWaveform file={null} bins={48} /><div className="eyebrow" style={{ marginTop: 20 }}>Awaiting recording</div><p style={{ marginTop: 9, color: 'var(--muted)', fontSize: 12, lineHeight: 1.65 }}>Select a recording and run inference to view the predicted class, model-score distribution and uncertainty status.</p></div>
            </div>
          ) : (
            <div className="lab-panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 22, alignItems: 'end', paddingBottom: 18, borderBottom: '1px solid var(--line)' }}>
                <div><div className="eyebrow">Predicted acoustic emotion</div><div style={{ marginTop: 8, color: predictionColor, fontSize: 38, fontWeight: 730, lineHeight: 1, letterSpacing: '-.045em', textTransform: 'capitalize' }}>{prediction}</div></div>
                <div style={{ textAlign: 'right' }}><div className="eyebrow">Top model score</div><div className="mono" style={{ marginTop: 8, fontSize: 27, fontWeight: 650 }}>{(ranked[0]?.score * 100 || 0).toFixed(1)}%</div></div>
              </div>

              {result.is_uncertain && <div className="lab-alert lab-alert-warning" style={{ marginTop: 15 }}><AlertCircle size={16} /><span>{result.uncertainty_note || 'The leading scores are close. Interpret the result cautiously.'}</span></div>}

              <div style={{ display: 'grid', gap: 13, marginTop: 20 }}>
                {ranked.map(({ emotion, score, color }) => <div className="score-row-research" key={emotion}><div className="score-label-research"><span className="score-dot-research" style={{ '--score-color': color }} />{emotion}</div><div className="score-track-research"><div className="score-fill-research" style={{ '--score-width': `${score * 100}%`, '--score-color': color }} /></div><div className="score-value-research">{(score * 100).toFixed(1)}%</div></div>)}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1, marginTop: 22, background: 'var(--line)', border: '1px solid var(--line)' }}>
                {[
                  ['Duration', `${Number(result.duration_seconds || 0).toFixed(2)} s`],
                  ['Processing', `${Number(result.processing_time_ms || 0).toFixed(0)} ms`],
                  ['Status', result.is_uncertain ? 'Uncertain' : 'Resolved'],
                ].map(([label, value]) => <div key={label} style={{ padding: 12, background: 'white' }}><div className="eyebrow">{label}</div><div className="mono" style={{ marginTop: 7, fontSize: 10 }}>{value}</div></div>)}
              </div>
              <div className="mono" style={{ marginTop: 12, color: 'var(--faint)', fontSize: 9 }}>MODEL / {result.model_version}</div>
              <div style={{ display: 'flex', gap: 9, marginTop: 18, flexWrap: 'wrap' }}><button className="lab-button lab-button-secondary" onClick={() => navigate(`/history/${result.id}`)}>Open full record</button><button className="lab-button lab-button-ghost" onClick={() => navigate('/history')}><History size={13} />Analysis history</button></div>
            </div>
          )}
        </section>
      </div>

      <div className="lab-alert lab-alert-info" style={{ marginTop: 18 }}><AlertCircle size={16} /><div><strong>Interpretation note.</strong> Scores describe the model output distribution. They are not verified psychological measurements and can be confidently wrong on speech unlike the training datasets.</div></div>
    </div>
  )
}
