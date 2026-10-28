import React from 'react'
import { EMOTION_COLORS } from './emotionTheme.js'

export default function EmotionBadge({ emotion, showIcon = false, size = 'md' }) {
  const normEmotion = emotion ? emotion.toLowerCase().trim() : 'neutral'
  const color = EMOTION_COLORS[normEmotion] || '#64748B'
  const padding = size === 'sm' ? '3px 7px' : size === 'lg' ? '6px 10px' : '4px 8px'

  return (
    <span
      className="emotion-badge-research"
      style={{ '--badge-color': color, padding }}
    >
      {showIcon && <span className="emotion-badge-dot-research" />}
      <span>{normEmotion}</span>
    </span>
  )
}
