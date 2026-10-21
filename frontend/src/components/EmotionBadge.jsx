import React from 'react'

export const EMOTION_COLORS = {
  angry: '#EF4444',
  calm: '#3B82F6',
  disgust: '#8B5CF6',
  fearful: '#F97316',
  happy: '#EAB308',
  neutral: '#64748B',
  sad: '#0284C7',
  surprised: '#EC4899',
}

export const EMOTION_ICONS = {
  angry: '😠',
  calm: '😌',
  disgust: '🤢',
  fearful: '😨',
  happy: '😄',
  neutral: '😐',
  sad: '😢',
  surprised: '😲',
}

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
