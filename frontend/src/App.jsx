import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import Overview from './pages/OverviewResearch.jsx'
import Analyze from './pages/AnalyzeResearch.jsx'
import History from './pages/HistoryResearch.jsx'
import Detail from './pages/DetailResearch.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="history" element={<History />} />
        <Route path="history/:id" element={<Detail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
