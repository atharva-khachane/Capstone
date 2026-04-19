import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { login, fetchStatus, saveSession } from "../api/client";

function PipelineStatusBanner({ ready, running }: { ready: boolean; running: boolean }) {
  if (ready) return null;

  return (
    <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 mb-5">
      <span className="mt-0.5 shrink-0">
        {running ? (
          <svg className="w-4 h-4 text-amber-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        ) : (
          <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
        )}
      </span>
      <div>
        <p className="text-xs font-semibold text-amber-300">
          {running ? "Pipeline is warming up…" : "Pipeline not ready"}
        </p>
        <p className="text-[11px] text-amber-400/80 mt-0.5">
          {running
            ? "Embedding documents — this takes ~2 min on first start. You can sign in now and wait inside."
            : "Backend is reachable but pipeline has not started yet."}
        </p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // Role is derived from the account on the backend.

  // Poll pipeline status every 4 s so the banner updates automatically
  const { data: status } = useQuery({
    queryKey: ["pipelineStatus"],
    queryFn: fetchStatus,
    refetchInterval: 4000,
    staleTime: 0,
  });

  const pipelineReady   = status?.ready ?? false;
  const pipelineRunning = status?.ingest_running ?? false;

  const mutation = useMutation({
    mutationFn: () => login(userId.trim() || "guest", password),
    onSuccess: (session) => {
      saveSession(session);
      // If pipeline isn't ready yet, navigate to chat — it shows its own loading overlay
      navigate("/chat");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!password) return;
    mutation.mutate();
  };

  // Extract a readable error message from Axios errors
  const errorMsg = (() => {
    if (!mutation.isError) return null;
    const err = mutation.error as { response?: { data?: { detail?: string }; status?: number }; message?: string };
    if (err?.response?.data?.detail) return err.response.data.detail;
    if (err?.response?.status === 401) return "Incorrect user ID or password.";
    if (err?.response?.status === 500) return "Server error — backend may still be starting up.";
    return err?.message ?? "Login failed. Check that the backend is running.";
  })();

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-brand-900/20 via-slate-950 to-slate-950 pointer-events-none" />

      <div className="relative w-full max-w-md">
        {/* Logo block */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-brand-600 rounded-2xl mb-4 shadow-lg shadow-brand-600/30">
            <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white">SL-RAG</h1>
          <p className="text-sm text-slate-400 mt-1">ISRO Document Intelligence Platform</p>
        </div>

        {/* Login card */}
        <div className="card shadow-2xl shadow-black/50">
          <h2 className="text-base font-semibold text-slate-200 mb-5">Sign in to continue</h2>

          {/* Pipeline loading banner */}
          <PipelineStatusBanner ready={pipelineReady} running={pipelineRunning} />

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* User ID */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                User ID
              </label>
              <input
                type="text"
                className="input"
                placeholder="e.g. admin, analyst, auditor, guest"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                autoComplete="username"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  className="input pr-10"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  onClick={() => setShowPassword((v) => !v)}
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {/* Role selector */}
              {/* Role selector removed */}

            {/* Error */}
            {errorMsg && (
              <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                {errorMsg}
              </p>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={mutation.isPending || !password}
              className="btn-primary w-full py-2.5"
            >
              {mutation.isPending ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Signing in…
                </span>
              ) : (
                "Sign in"
              )}
            </button>
          </form>
        </div>

        {/* Demo credentials hint */}
        <div className="mt-4 card bg-slate-900/60 border-slate-800">
          <p className="text-[10px] text-slate-500 font-medium mb-2">Demo credentials</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {[
              ["admin",   "admin123"],
              ["analyst", "analyst123"],
              ["auditor", "auditor123"],
              ["guest",   "guest"],
            ].map(([uid, pw]) => (
              <button
                key={uid}
                type="button"
                className="text-left text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
                onClick={() => { setUserId(uid); setPassword(pw); setSelectedRole(uid === "admin" ? "admin" : uid); }}
              >
                <span className="text-slate-400">{uid}</span>
                <span className="text-slate-600"> / </span>
                <span className="font-mono">{pw}</span>
              </button>
            ))}
          </div>
        </div>

        <p className="text-center text-[10px] text-slate-600 mt-4">
          SL-RAG v1.0 · Secure offline pipeline · ISRO
        </p>
      </div>
    </div>
  );
}
