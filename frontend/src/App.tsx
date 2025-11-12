import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import { LoginPage } from './pages/LoginPage';
import { AdminDashboard } from './pages/AdminDashboard';
import { OwnerDashboard } from './pages/OwnerDashboard';
import { ReportsPage } from './pages/ReportsPage';
import { PageLayout } from './components/PageLayout';

const Loading = () => (
  <PageLayout>
    <p>Loading...</p>
  </PageLayout>
);

export default function App() {
  const { user, loading } = useAuth();
  if (loading) return <Loading />;

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/admin"
        element={user?.role === 'ADMIN' ? <AdminDashboard /> : <Navigate to="/login" replace />}
      />
      <Route
        path="/owner"
        element={user?.role === 'OWNER' ? <OwnerDashboard /> : <Navigate to="/login" replace />}
      />
      <Route
        path="/reports"
        element={user?.role === 'ADMIN' ? <ReportsPage /> : <Navigate to="/login" replace />}
      />
      <Route
        path="/"
        element={
          user ? (
            user.role === 'ADMIN' ? (
              <Navigate to="/admin" replace />
            ) : user.role === 'OWNER' ? (
              <Navigate to="/owner" replace />
            ) : (
              <PageLayout>
                <p>Viewer mode coming soon.</p>
              </PageLayout>
            )
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
    </Routes>
  );
}
