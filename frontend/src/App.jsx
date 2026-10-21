import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import Overview from './pages/Overview.jsx'
import Analyze from './pages/Analyze.jsx'
import History from './pages/History.jsx'
import Detail from './pages/Detail.jsx'

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
