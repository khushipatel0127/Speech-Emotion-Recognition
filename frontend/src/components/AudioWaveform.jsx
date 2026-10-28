import React, { useEffect, useMemo, useState } from 'react'

const FALLBACK = [0.2, 0.42, 0.35, 0.68, 0.46, 0.82, 0.38, 0.58, 0.9, 0.52, 0.76, 0.33, 0.62, 0.44, 0.7, 0.28]

export default function AudioWaveform({ file, bins = 72 }) {
  const [peaks, setPeaks] = useState(null)
  useEffect(() => {
    let cancelled = false
    if (!file) { setPeaks(null); return undefined }
    const decode = async () => {
      let context
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext
        if (!AudioContext) return
        context = new AudioContext()
        const buffer = await context.decodeAudioData(await file.arrayBuffer())
        const channel = buffer.getChannelData(0)
        const block = Math.max(1, Math.floor(channel.length / bins))
        const values = Array.from({ length: bins }, (_, index) => {
          let sum = 0
          const start = index * block
          const end = Math.min(channel.length, start + block)
          for (let i = start; i < end; i += 1) sum += Math.abs(channel[i])
          return sum / Math.max(1, end - start)
        })
        const max = Math.max(...values, 0.001)
        if (!cancelled) setPeaks(values.map((value) => Math.max(0.05, value / max)))
      } catch {
        if (!cancelled) setPeaks(null)
      } finally {
        if (context) context.close().catch(() => {})
      }
    }
    decode()
    return () => { cancelled = true }
  }, [file, bins])

  const points = useMemo(() => {
    const source = peaks || Array.from({ length: bins }, (_, i) => FALLBACK[i % FALLBACK.length])
    return source.map((value, index) => {
      const x = (index / Math.max(1, source.length - 1)) * 1000
      const half = value * 42
      return `${x},${52 - half} ${x},${52 + half}`
    })
  }, [peaks, bins])

  return (
    <div className="waveform-frame" aria-label={file ? 'Audio waveform preview' : 'Waveform placeholder'}>
      <svg className="waveform-svg" viewBox="0 0 1000 104" preserveAspectRatio="none" role="img">
        <title>{file ? 'Waveform generated from selected audio' : 'No audio selected'}</title>
        {points.map((point, index) => <polyline key={index} points={point} stroke={file ? '#0f766e' : '#aab6b2'} strokeWidth="5" opacity={file ? .82 : .42} />)}
      </svg>
    </div>
  )
}
