import React from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar.jsx'

export default function Layout() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F8FAFC' }}>
      <Sidebar />
      <main
        style={{
          flex: 1,
          marginLeft: '260px',
          padding: '36px 48px',
          maxWidth: 'calc(100vw - 260px)',
          minHeight: '100vh',
          boxSizing: 'border-box',
          overflowX: 'hidden',
        }}
      >
        <div style={{ maxWidth: '1180px', margin: '0 auto' }}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
