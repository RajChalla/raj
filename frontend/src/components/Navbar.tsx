import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className="navbar">
      <h1>FC26 Auction</h1>
      <nav>
        {user?.role === 'ADMIN' && (
          <>
            <Link to="/admin">Admin</Link>
            <Link to="/reports">Reports</Link>
          </>
        )}
        {user?.role === 'OWNER' && <Link to="/owner">My Team</Link>}
      </nav>
      <div className="auth">
        {user ? (
          <button onClick={logout}>Logout ({user.displayName})</button>
        ) : (
          <Link to="/login">Login</Link>
        )}
      </div>
    </header>
  );
}
