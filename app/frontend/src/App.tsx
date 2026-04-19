import React from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  Outlet,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { loadSession } from "./api/client";
import { ChatProvider } from "./context/ChatContext";
import Navbar from "./components/Navbar";
import LoginPage from "./pages/LoginPage";
import ChatPage from "./pages/ChatPage";
import IngestPage from "./pages/IngestPage";
import AuditPage from "./pages/AuditPage";
import AdminUsersPage from "./pages/AdminUsersPage";

// Guard: redirect to /login if no session exists
function RequireAuth() {
  const session = loadSession();
  if (!session) return <Navigate to="/login" replace />;
  return (
    <div className="flex h-full">
      <Navbar />
      <main className="flex-1 overflow-hidden flex flex-col bg-slate-950">
        <Outlet />
      </main>
    </div>
  );
}

function RequireAdmin() {
  const session = loadSession();
  if (!session) return <Navigate to="/login" replace />;
  if (session.role !== "admin") return <Navigate to="/chat" replace />;
  return <Outlet />;
}

function RequireNonGuest() {
  const session = loadSession();
  if (!session) return <Navigate to="/login" replace />;
  if (session.role === "guest") return <Navigate to="/chat" replace />;
  return <Outlet />;
}

function RequireAuditRole() {
  const session = loadSession();
  if (!session) return <Navigate to="/login" replace />;
  if (session.role !== "admin" && session.role !== "auditor") {
    return <Navigate to="/chat" replace />;
  }
  return <Outlet />;
}

function ForceLoginLanding() {
  const navigate = useNavigate();
  const location = useLocation();
  const didRedirect = React.useRef(false);

  React.useEffect(() => {
    if (didRedirect.current) return;
    didRedirect.current = true;

    if (location.pathname !== "/login") {
      navigate("/login", { replace: true });
    }
  }, [location.pathname, navigate]);

  return null;
}

export default function App() {
  return (
    <ChatProvider>
      <BrowserRouter>
        <ForceLoginLanding />
        <Routes>
          {/* Always land on the sign-in page when opening the app */}
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
            <Route path="/chat"   element={<ChatPage />} />
            <Route element={<RequireNonGuest />}>
              <Route path="/ingest" element={<IngestPage />} />
            </Route>
            <Route element={<RequireAuditRole />}>
              <Route path="/audit"  element={<AuditPage />} />
            </Route>
            <Route element={<RequireAdmin />}>
              <Route path="/admin/users" element={<AdminUsersPage />} />
            </Route>
          </Route>
          {/* Default redirect */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    </ChatProvider>
  );
}
