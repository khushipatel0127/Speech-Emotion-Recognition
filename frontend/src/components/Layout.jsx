import React from 'react'
import { Outlet } from 'react-router-dom'
import ResearchSidebar from './ResearchSidebar.jsx'

export default function Layout() {
  return (
    <div className="app-shell">
      <ResearchSidebar />
      <main className="app-main">
        <div className="app-content">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
