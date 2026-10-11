import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MotionConfig } from 'motion/react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { STATIC } from './components/motion'
import { StreamProvider } from './hooks/useStream'
import AnalyzePage from './pages/AnalyzePage'
import EvaluationPage from './pages/EvaluationPage'
import IndexPage from './pages/IndexPage'
import Overview from './pages/Overview'
import SignalDesk from './pages/SignalDesk'
import StressPage from './pages/StressPage'

const qc = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1, placeholderData: (prev: unknown) => prev } } })

export default function App() {
  return (
    <MotionConfig reducedMotion={STATIC ? 'always' : 'user'}>
    <QueryClientProvider client={qc}>
      <StreamProvider>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/desk" element={<SignalDesk />} />
              <Route path="/analyze" element={<AnalyzePage />} />
              <Route path="/index" element={<IndexPage />} />
              <Route path="/stress" element={<StressPage />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
              <Route path="/label" element={<Navigate to="/evaluation" replace />} />
              <Route path="/quality" element={<Navigate to="/evaluation" replace />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </StreamProvider>
    </QueryClientProvider>
    </MotionConfig>
  )
}
