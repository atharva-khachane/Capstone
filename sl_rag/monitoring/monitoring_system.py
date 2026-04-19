"""
Monitoring & Governance System (Layer 6).

Per ANTIGRAVITY_PROMPT.md Phase 6, provides:
  - Query logging with timestamps and latency
  - Tamper-evident audit trail (SHA-256 hash chain)
  - Document access logging
  - Security event tracking (PII detections, unusual patterns)
  - Performance metrics
  - Drift detection stub

Storage: SQLite (encrypted via EncryptionManager if provided).
Retention: 7 years (configurable).

100% OFFLINE - No external APIs.
"""

import sqlite3
import hashlib
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class MonitoringSystem:
    """Comprehensive monitoring and governance for ISRO document RAG."""

    def __init__(
        self,
        db_path: str = "./storage/audit_logs/monitoring.db",
        encryption_manager=None,
        retention_years: int = 7,
        encrypt_at_rest: bool = False,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encryption_manager = encryption_manager
        self.retention_years = retention_years
        self.encrypt_at_rest = encrypt_at_rest
        if self.encrypt_at_rest and not self.encryption_manager:
            raise ValueError("encrypt_at_rest=True requires encryption_manager")

        self._runtime_db_path = (
            self.db_path.with_suffix(self.db_path.suffix + ".runtime")
            if self.encrypt_at_rest else self.db_path
        )
        self._previous_hash = "GENESIS"
        self._sensitive_access_counts: Dict[str, int] = {}
        self._drift_baseline: Dict[str, List[float]] = {}
        if self.encrypt_at_rest:
            self._hydrate_runtime_db()
        self._init_db()
        # BUG FIX: seed _previous_hash from existing audit trail so the hash
        # chain stays intact across multiple MonitoringSystem instantiations.
        self._seed_previous_hash()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(str(self._runtime_db_path))
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            query_text  TEXT NOT NULL,
            query_hash  TEXT NOT NULL,
            user_id     TEXT,
            role        TEXT,
            session_id  TEXT,
            domains     TEXT,
            num_results INTEGER,
            latency_ms  REAL,
            confidence  REAL,
            metadata    TEXT
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS document_access_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            user_id     TEXT,
            role        TEXT,
            session_id  TEXT,
            doc_id      TEXT NOT NULL,
            chunk_ids   TEXT,
            query_hash  TEXT,
            action      TEXT DEFAULT 'retrieve'
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            severity    TEXT NOT NULL,
            details     TEXT,
            resolved    INTEGER DEFAULT 0
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT NOT NULL,
            operation      TEXT NOT NULL,
            details        TEXT,
            previous_hash  TEXT NOT NULL,
            current_hash   TEXT NOT NULL
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value       REAL NOT NULL,
            metadata    TEXT
        )""")

        # Backward-compatible schema migration for older DB files.
        self._ensure_column(c, "query_log", "user_id", "TEXT")
        self._ensure_column(c, "query_log", "role", "TEXT")
        self._ensure_column(c, "query_log", "session_id", "TEXT")
        self._ensure_column(c, "document_access_log", "user_id", "TEXT")
        self._ensure_column(c, "document_access_log", "role", "TEXT")
        self._ensure_column(c, "document_access_log", "session_id", "TEXT")

        conn.commit()
        conn.close()
        if self.encrypt_at_rest:
            self._persist_encrypted_db()
        print(f"[MONITOR] Database initialised at {self.db_path}")

    def _seed_previous_hash(self) -> None:
        """Seed _previous_hash from the last audit_trail row (if any).

        Without this, every new MonitoringSystem instance resets the chain
        to 'GENESIS', which breaks verify_audit_chain() when the DB already
        has entries from a previous run.
        """
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("SELECT current_hash FROM audit_trail ORDER BY id DESC LIMIT 1")
            row = c.fetchone()
            conn.close()
            if row:
                self._previous_hash = row[0]
        except Exception:
            pass  # Fresh DB or unreadable — keep GENESIS

    def _hydrate_runtime_db(self) -> None:
        """Restore runtime sqlite DB from encrypted-at-rest bytes if present."""
        if not self.db_path.exists():
            return
        encrypted_bytes = self.db_path.read_bytes()
        try:
            decrypted = self.encryption_manager.decrypt_text(encrypted_bytes).encode("latin1")
            self._runtime_db_path.write_bytes(decrypted)
        except Exception:
            # Backward compatibility: treat existing db_path as plaintext sqlite.
            self._runtime_db_path.write_bytes(encrypted_bytes)

    def _persist_encrypted_db(self) -> None:
        """Persist runtime sqlite DB to encrypted-at-rest file."""
        if not self._runtime_db_path.exists():
            return
        plain = self._runtime_db_path.read_bytes()
        encrypted = self.encryption_manager.encrypt_text(plain.decode("latin1"))
        self.db_path.write_bytes(encrypted)

    def _connect(self):
        return sqlite3.connect(str(self._runtime_db_path))

    @staticmethod
    def _ensure_column(cursor, table: str, column: str, column_type: str) -> None:
        """Add a column if it does not already exist."""
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------

    def log_query(
        self,
        query: str,
        domains: List[str],
        num_results: int,
        latency_ms: float,
        confidence: float = 0.0,
        user_id: str = "",
        role: str = "",
        session_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> int:
        now = datetime.now().isoformat()
        q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO query_log "
            "(timestamp, query_text, query_hash, user_id, role, session_id, domains, num_results, latency_ms, confidence, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (now, query, q_hash, user_id, role, session_id, json.dumps(domains),
             num_results, latency_ms, confidence, json.dumps(metadata or {})),
        )
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        if self.encrypt_at_rest:
            self._persist_encrypted_db()

        self._append_audit("query", f"query_hash={q_hash}")
        return row_id

    # ------------------------------------------------------------------
    # Document access logging
    # ------------------------------------------------------------------

    def log_document_access(
        self,
        doc_id: str,
        chunk_ids: List[str],
        query_hash: str = "",
        user_id: str = "",
        role: str = "",
        session_id: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO document_access_log "
            "(timestamp, user_id, role, session_id, doc_id, chunk_ids, query_hash, action) VALUES (?,?,?,?,?,?,?,?)",
            (now, user_id, role, session_id, doc_id, json.dumps(chunk_ids), query_hash, "retrieve"),
        )
        conn.commit()
        conn.close()
        if self.encrypt_at_rest:
            self._persist_encrypted_db()

    # ------------------------------------------------------------------
    # Security events
    # ------------------------------------------------------------------

    def log_security_event(
        self,
        event_type: str,
        severity: str = "info",
        details: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO security_events "
            "(timestamp, event_type, severity, details) VALUES (?,?,?,?)",
            (now, event_type, severity, details),
        )
        conn.commit()
        conn.close()
        if self.encrypt_at_rest:
            self._persist_encrypted_db()

        self._append_audit(f"security_{severity}", f"{event_type}: {details[:100]}")

    # ------------------------------------------------------------------
    # Tamper-evident audit trail (hash chain)
    # ------------------------------------------------------------------

    def _append_audit(self, operation: str, details: str) -> None:
        """Atomically append a new audit entry, reading the last hash inside
        the same exclusive transaction to prevent chain forking when multiple
        processes write concurrently to the same SQLite database."""
        now = datetime.now().isoformat()

        conn = self._connect()
        # EXCLUSIVE transaction: no other writer can insert between our SELECT
        # and INSERT, preventing concurrent processes from forking the chain.
        conn.execute("BEGIN EXCLUSIVE")
        c = conn.cursor()

        # Read the true last hash from the DB (not from in-memory cache) so
        # concurrent processes never see the same _previous_hash.
        c.execute("SELECT current_hash FROM audit_trail ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        previous_hash = row[0] if row else "GENESIS"

        payload = f"{now}|{operation}|{details}|{previous_hash}"
        current_hash = hashlib.sha256(payload.encode()).hexdigest()

        c.execute(
            "INSERT INTO audit_trail "
            "(timestamp, operation, details, previous_hash, current_hash) "
            "VALUES (?,?,?,?,?)",
            (now, operation, details, previous_hash, current_hash),
        )
        conn.commit()
        conn.close()
        if self.encrypt_at_rest:
            self._persist_encrypted_db()

        self._previous_hash = current_hash

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit trail."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT timestamp, operation, details, previous_hash, current_hash "
                  "FROM audit_trail ORDER BY id")
        rows = c.fetchall()
        conn.close()

        if not rows:
            return {"valid": True, "entries": 0, "broken_at": None}

        prev = "GENESIS"
        for i, (ts, op, det, ph, ch) in enumerate(rows):
            if ph != prev:
                return {"valid": False, "entries": len(rows), "broken_at": i}
            payload = f"{ts}|{op}|{det}|{ph}"
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if ch != expected:
                return {"valid": False, "entries": len(rows), "broken_at": i}
            prev = ch

        return {"valid": True, "entries": len(rows), "broken_at": None}

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        now = datetime.now().isoformat()
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO performance_metrics (timestamp, metric_name, value, metadata) "
            "VALUES (?,?,?,?)",
            (now, name, value, json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_query_stats(self, last_n: int = 100) -> Dict[str, Any]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "SELECT latency_ms, confidence, num_results FROM query_log "
            "ORDER BY id DESC LIMIT ?", (last_n,),
        )
        rows = c.fetchall()
        conn.close()

        if not rows:
            return {"total_queries": 0}

        latencies = [r[0] for r in rows if r[0] is not None]
        confidences = [r[1] for r in rows if r[1] is not None]

        return {
            "total_queries": len(rows),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "avg_results": round(sum(r[2] for r in rows) / len(rows), 1),
        }

    def get_security_summary(self) -> Dict[str, int]:
        conn = self._connect()  # BUG FIX: was sqlite3.connect(str(self.db_path)) — wrong path when encrypted
        c = conn.cursor()
        c.execute("SELECT severity, COUNT(*) FROM security_events GROUP BY severity")
        rows = c.fetchall()
        conn.close()
        return {sev: cnt for sev, cnt in rows}

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a report suitable for government audit requirements."""
        audit = self.verify_audit_chain()
        q_stats = self.get_query_stats()
        sec = self.get_security_summary()

        return {
            "generated_at": datetime.now().isoformat(),
            "audit_trail_integrity": audit,
            "query_statistics": q_stats,
            "security_events": sec,
            "retention_policy_years": self.retention_years,
        }

    # ------------------------------------------------------------------
    # Query pattern anomaly monitoring
    # ------------------------------------------------------------------

    def analyze_query_patterns(
        self,
        user_id: str,
        domains: List[str],
        sensitive_doc_ids: Optional[List[str]] = None,
        window_seconds: int = 60,
        rate_threshold: int = 20,
        sensitive_repeat_threshold: int = 3,
        min_history_for_domain_anomaly: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Detect unusual query patterns and emit security events.

        Returns a list of detected anomalies for optional caller use.
        """
        anomalies: List[Dict[str, Any]] = []
        user_id = user_id or "anonymous"
        now_epoch = time.time()
        cutoff = now_epoch - window_seconds

        # 1) Rate spike detection.
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "SELECT timestamp FROM query_log WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        )
        rows = c.fetchall()
        conn.close()
        recent = 0
        for (ts,) in rows:
            try:
                if datetime.fromisoformat(ts).timestamp() >= cutoff:
                    recent += 1
            except Exception:
                continue
        if recent > rate_threshold:
            details = (
                f"user_id={user_id} recent_queries={recent} "
                f"window_seconds={window_seconds} threshold={rate_threshold}"
            )
            self.log_security_event("query_rate_spike", "warning", details)
            anomalies.append({"type": "query_rate_spike", "details": details})

        # 2) Repeated sensitive document access.
        for doc_id in (sensitive_doc_ids or []):
            key = f"{user_id}:{doc_id}"
            self._sensitive_access_counts[key] = self._sensitive_access_counts.get(key, 0) + 1
            count = self._sensitive_access_counts[key]
            if count >= sensitive_repeat_threshold:
                details = (
                    f"user_id={user_id} doc_id={doc_id} "
                    f"repeat_count={count} threshold={sensitive_repeat_threshold}"
                )
                self.log_security_event("repeated_sensitive_doc_access", "warning", details)
                anomalies.append({"type": "repeated_sensitive_doc_access", "details": details})

        # 3) Unusual domain access (domain drift per user profile).
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "SELECT domains FROM query_log WHERE user_id = ? ORDER BY id DESC LIMIT 100",
            (user_id,),
        )
        domain_rows = c.fetchall()
        conn.close()
        domain_counts: Dict[str, int] = {}
        history_total = 0
        for (dom_json,) in domain_rows:
            try:
                parsed = json.loads(dom_json or "[]")
            except Exception:
                parsed = []
            for d in parsed:
                domain_counts[d] = domain_counts.get(d, 0) + 1
                history_total += 1
        if history_total >= min_history_for_domain_anomaly and domain_counts:
            top_domains = {d for d, _ in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:2]}
            if domains and all(d not in top_domains for d in domains):
                details = (
                    f"user_id={user_id} current_domains={domains} "
                    f"usual_domains={sorted(top_domains)}"
                )
                self.log_security_event("unusual_domain_access_pattern", "info", details)
                anomalies.append({"type": "unusual_domain_access_pattern", "details": details})

        return anomalies

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        cosine = dot / (norm_a * norm_b)
        cosine = max(-1.0, min(1.0, cosine))
        return 1.0 - cosine

    def check_domain_drift(
        self,
        domain_centroids: Dict[str, Any],
        threshold: float = 0.15,
        update_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare current domain centroids to baseline and emit drift events.
        """
        current = {k: [float(x) for x in v.tolist()] if hasattr(v, "tolist") else [float(x) for x in v]
                   for k, v in domain_centroids.items()}
        if not current:
            return {"drift_detected": False, "mean_distance": 0.0, "domains_compared": 0}

        if not self._drift_baseline:
            self._drift_baseline = current
            self.log_metric(
                "domain_drift_baseline_initialized",
                1.0,
                {"num_domains": len(current)},
            )
            return {"drift_detected": False, "mean_distance": 0.0, "domains_compared": 0}

        common = sorted(set(self._drift_baseline.keys()) & set(current.keys()))
        if not common:
            if update_baseline:
                self._drift_baseline = current
            return {"drift_detected": False, "mean_distance": 0.0, "domains_compared": 0}

        distances = [
            self._cosine_distance(self._drift_baseline[d], current[d]) for d in common
        ]
        mean_distance = sum(distances) / len(distances)
        drift_detected = mean_distance > threshold

        self.log_metric(
            "domain_drift_mean_distance",
            mean_distance,
            {"domains_compared": len(common), "threshold": threshold},
        )
        if drift_detected:
            self.log_security_event(
                "domain_embedding_drift_detected",
                "warning",
                f"mean_distance={mean_distance:.4f} threshold={threshold} domains={len(common)}",
            )

        if update_baseline:
            self._drift_baseline = current

        return {
            "drift_detected": drift_detected,
            "mean_distance": round(mean_distance, 6),
            "domains_compared": len(common),
            "threshold": threshold,
        }
