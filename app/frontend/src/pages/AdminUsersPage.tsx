import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createUser, deleteUser, fetchRoles, fetchUsers, loadSession, clearSession } from "../api/client";
import type { RoleOption } from "../api/types";
import { useChatContext } from "../context/ChatContext";

export default function AdminUsersPage() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const session = loadSession();
  const { resetChat } = useChatContext();
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [selectedRole, setSelectedRole] = useState<string>("guest");

  const ROLE_ORDER = ["admin", "auditor", "analyst", "guest"] as const;
  const ROLE_RANK: Record<string, number> = useMemo(
    () => ({ guest: 0, analyst: 1, auditor: 2, admin: 3 }),
    []
  );

  const usersQuery = useQuery({
    queryKey: ["admin-users"],
    queryFn: fetchUsers,
  });

  const rolesQuery = useQuery<RoleOption[]>({
    queryKey: ["roles"],
    queryFn: fetchRoles,
  });

  const sortedRoles = useMemo(() => {
    const roles = (rolesQuery.data ?? []).map((r) => r.role);
    const unique = Array.from(new Set(roles));
    const ranked = unique.filter((r) => r in ROLE_RANK).sort((a, b) => ROLE_RANK[b] - ROLE_RANK[a]);

    // Fallback: always show the known hierarchy.
    return ranked.length ? ranked : Array.from(ROLE_ORDER);
  }, [rolesQuery.data, ROLE_RANK]);

  const mutation = useMutation({
    mutationFn: () =>
      createUser({
        user_id: userId.trim(),
        password,
        role: selectedRole,
      }),
    onSuccess: async () => {
      setUserId("");
      setPassword("");
      setSelectedRole("guest");
      await qc.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });

  const errorMsg = (() => {
    if (!mutation.isError) return null;
    const err = mutation.error as {
      response?: { data?: { detail?: string } };
      message?: string;
    };
    return err?.response?.data?.detail ?? err?.message ?? "Failed to create user.";
  })();

  const deleteMutation = useMutation({
    mutationFn: (uid: string) => deleteUser(uid),
    onSuccess: async (_data, uid) => {
      await qc.invalidateQueries({ queryKey: ["admin-users"] });

      const deleted = (uid || "").toLowerCase();
      const current = (session?.user_id || "").toLowerCase();
      if (deleted && current && deleted === current) {
        resetChat();
        clearSession();
        navigate("/login", { replace: true });
      }
    },
  });

  const deleteErrorMsg = (() => {
    if (!deleteMutation.isError) return null;
    const err = deleteMutation.error as {
      response?: { data?: { detail?: string } };
      message?: string;
    };
    return err?.response?.data?.detail ?? err?.message ?? "Failed to delete user.";
  })();

  const selectRole = (role: string) => setSelectedRole(role);

  return (
    <div className="h-full overflow-y-auto p-6 lg:p-8 bg-slate-950 text-slate-100">
      <div className="max-w-6xl mx-auto space-y-6">
        <header>
          <h1 className="text-2xl font-bold">User Management</h1>
          <p className="text-sm text-slate-400 mt-1">
            Admin-only controls to create users and assign access roles.
          </p>
        </header>

        <section className="card">
          <h2 className="text-base font-semibold mb-4">Create User</h2>
          <form
            className="grid grid-cols-1 lg:grid-cols-12 gap-4"
            onSubmit={(e) => {
              e.preventDefault();
              mutation.mutate();
            }}
          >
            <div className="lg:col-span-4">
              <label className="block text-xs text-slate-400 mb-1">User ID</label>
              <input
                className="input"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="e.g. mission_analyst"
                minLength={3}
                maxLength={64}
                required
              />
            </div>

            <div className="lg:col-span-4">
              <label className="block text-xs text-slate-400 mb-1">Password</label>
              <input
                type="password"
                className="input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Minimum 6 characters"
                minLength={6}
                maxLength={256}
                required
              />
            </div>

            <div className="lg:col-span-12">
              <label className="block text-xs text-slate-400 mb-2">Role</label>
              <div className="flex flex-wrap gap-2">
                {(sortedRoles.length ? sortedRoles : ["admin", "auditor", "analyst", "guest"]).map((role) => {
                  const active = selectedRole === role;
                  return (
                    <button
                      key={role}
                      type="button"
                      onClick={() => selectRole(role)}
                      className={
                        active
                          ? "px-3 py-1.5 rounded-md text-xs border border-brand-500/40 bg-brand-500/20 text-brand-200"
                          : "px-3 py-1.5 rounded-md text-xs border border-slate-700 bg-slate-800 text-slate-300 hover:border-slate-500"
                      }
                    >
                      {role}
                    </button>
                  );
                })}
              </div>
            </div>

            {errorMsg && (
              <p className="lg:col-span-12 text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                {errorMsg}
              </p>
            )}

            {mutation.isSuccess && (
              <p className="lg:col-span-12 text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
                User created successfully.
              </p>
            )}

            <div className="lg:col-span-12">
              <button
                type="submit"
                className="btn-primary"
                disabled={mutation.isPending || !userId.trim() || !password || !selectedRole}
              >
                {mutation.isPending ? "Creating..." : "Create User"}
              </button>
            </div>
          </form>
        </section>

        <section className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold">Existing Users</h2>
            <button
              type="button"
              className="btn-ghost text-xs"
              onClick={() => usersQuery.refetch()}
            >
              Refresh
            </button>
          </div>

          {usersQuery.isLoading && <p className="text-sm text-slate-400">Loading users...</p>}

          {usersQuery.isError && (
            <p className="text-sm text-red-300">
              Could not load users. Make sure you are logged in as admin.
            </p>
          )}

          {usersQuery.data && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-800">
                    <th className="py-2 pr-4">User ID</th>
                    <th className="py-2 pr-4">Role</th>
                    <th className="py-2 pr-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {usersQuery.data.map((u) => (
                    (() => {
                      const uid = (u.user_id || "").toLowerCase();
                      const isSelf = uid && uid === (session?.user_id || "").toLowerCase();
                      const isDefaultAdmin = uid === "admin";
                      const canDelete = !isDefaultAdmin;
                      const deleteTitle = isSelf
                        ? "Deleting yourself will log you out."
                        : isDefaultAdmin
                          ? "Default admin cannot be deleted."
                          : "Delete this user";

                      return (
                    <tr key={u.user_id} className="border-b border-slate-900/80">
                      <td className="py-2 pr-4 font-medium text-slate-200">{u.user_id}</td>
                      <td className="py-2 pr-4">
                        <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] text-slate-300">
                          {u.role}
                        </span>
                      </td>
                      <td className="py-2 pr-4">
                        <button
                          type="button"
                          title={deleteTitle}
                          className={
                            canDelete
                              ? "btn-ghost text-xs text-red-300 hover:text-red-200"
                              : "btn-ghost text-xs text-slate-500 cursor-not-allowed"
                          }
                          disabled={!canDelete || deleteMutation.isPending}
                          onClick={() => {
                            if (!canDelete) return;
                            const ok = window.confirm(`Delete user '${u.user_id}'? This cannot be undone.`);
                            if (!ok) return;
                            deleteMutation.mutate(u.user_id);
                          }}
                        >
                          {deleteMutation.isPending ? "Deleting..." : "Delete"}
                        </button>
                      </td>
                    </tr>
                      );
                    })()
                  ))}
                </tbody>
              </table>

              {deleteErrorMsg && (
                <p className="mt-4 text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                  {deleteErrorMsg}
                </p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
