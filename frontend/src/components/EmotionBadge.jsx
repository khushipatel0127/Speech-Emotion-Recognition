import React from 'react'
import { EMOTION_COLORS, EMOTION_ICONS } from './emotionTheme.js'

export default function EmotionBadge({ emotion, showIcon = false, size = 'md' }) {
  const normEmotion = emotion ? emotion.toLowerCase().trim() : 'neutral'
  const color = EMOTION_COLORS[normEmotion] || '#64748B'
  const icon = EMOTION_ICONS[normEmotion] || '🎤'

  const isSmall = size === 'sm'
  const isLarge = size === 'lg'

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        backgroundColor: `${color}18`,
        color: color,
        border: `1px solid ${color}33`,
        borderRadius: '9999px',
        padding: isSmall ? '2px 8px' : isLarge ? '6px 16px' : '4px 12px',
        fontSize: isSmall ? '11px' : isLarge ? '14px' : '12px',
        fontWeight: 600,
        textTransform: 'capitalize',
        letterSpacing: '0.01em',
        whiteSpace: 'nowrap',
      }}
    >
      {showIcon && <span>{icon}</span>}
      <span>{normEmotion}</span>
    </span>
  )
}
