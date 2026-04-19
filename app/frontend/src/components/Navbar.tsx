import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearSession, loadSession } from "../api/client";
import { useChatContext } from "../context/ChatContext";

const ROLE_COLORS: Record<string, string> = {
  admin:   "bg-red-500/20 text-red-300 border-red-500/30",
  analyst: "bg-brand-500/20 text-brand-300 border-brand-500/30",
  auditor: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  guest:   "bg-slate-500/20 text-slate-300 border-slate-500/30",
};

interface NavItemProps {
  to: string;
  label: string;
  icon: React.ReactNode;
}

function NavItem({ to, label, icon }: NavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
          isActive
            ? "bg-brand-600/20 text-brand-300 border border-brand-500/30"
            : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
        }`
      }
    >
      {icon}
      {label}
    </NavLink>
  );
}

export default function Navbar() {
  const navigate = useNavigate();
  const session = loadSession();
  const { resetChat } = useChatContext();

  const handleLogout = () => {
    resetChat();
    clearSession();
    navigate("/login");
  };

  const roleClasses =
    session ? (ROLE_COLORS[session.role] ?? ROLE_COLORS.guest) : "";

  return (
    <nav className="h-full flex flex-col bg-slate-900 border-r border-slate-800 w-56 shrink-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-none">SL-RAG</p>
            <p className="text-[10px] text-slate-500 mt-0.5">ISRO DocIntel</p>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <NavItem
          to="/chat"
          label="Chat"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          }
        />
        {session?.role !== "guest" && (
          <>
            <NavItem
              to="/ingest"
              label="Documents"
              icon={
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              }
            />
            {(session?.role === "admin" || session?.role === "auditor") && (
              <NavItem
                to="/audit"
                label="Audit Log"
                icon={
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                }
              />
            )}
          </>
        )}
        {session?.role === "admin" && (
          <NavItem
            to="/admin/users"
            label="Users"
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 20h5V4H2v16h5m10 0v-2a4 4 0 00-4-4H9a4 4 0 00-4 4v2m12 0H7m8-12a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            }
          />
        )}
      </div>

      {/* User section */}
      {session && (
        <div className="px-3 py-4 border-t border-slate-800 space-y-3">
          <div className="px-2">
            <p className="text-xs font-semibold text-slate-200 truncate">{session.user_id}</p>
            <span className={`badge rounded-md text-[10px] mt-1 border ${roleClasses}`}>
              {session.role}
            </span>
          </div>
          <button
            onClick={handleLogout}
            className="w-full btn-ghost text-xs justify-start gap-2 text-slate-500 hover:text-red-400"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            Sign out
          </button>
        </div>
      )}
    </nav>
  );
}
