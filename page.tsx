import React, { useState, useEffect, useRef, useCallback } from "react";

// ─── Types ───────────────────────────────────────────
type StatusType =
  | "DISCONNECTED"
  | "BLE_SCANNING"
  | "BLE_NOT_FOUND"
  | "BLE_CONNECTED"
  | "BLE_DISCONNECTED"
  | "RECORDING"
  | "SAVED"
  | "EMPTY"
  | "ERROR"
  | "PONG";

interface WsMessage {
  status: StatusType;
  label?: string;
  filename?: string;
  samples?: number;
  message?: string;
  ble_connected?: boolean;
  is_recording?: boolean;
  total_samples?: number;
}

interface LogEntry {
  id: number;
  time: string;
  text: string;
  type: "info" | "success" | "error" | "recording";
}

// ─── Constants ───────────────────────────────────────
const WS_URL = "ws://localhost:8000/ws";
const RECONNECT_DELAY = 3000;
const PING_INTERVAL = 5000;

// ─── Helpers ─────────────────────────────────────────
let logId = 0;
function mkLog(
  text: string,
  type: LogEntry["type"] = "info"
): LogEntry {
  return {
    id: logId++,
    time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    text,
    type,
  };
}

const statusConfig: Record<
  string,
  { label: string; color: string; pulse: boolean }
> = {
  DISCONNECTED: { label: "Disconnected", color: "#ef4444", pulse: false },
  BLE_SCANNING: { label: "Scanning…", color: "#f59e0b", pulse: true },
  BLE_NOT_FOUND: { label: "Pen Not Found", color: "#f59e0b", pulse: false },
  BLE_CONNECTED: { label: "BLE Connected", color: "#22c55e", pulse: false },
  BLE_DISCONNECTED: { label: "BLE Lost", color: "#ef4444", pulse: true },
  RECORDING: { label: "Recording", color: "#8b5cf6", pulse: true },
};

// ─── Component ───────────────────────────────────────
export default function SmartPenCollector() {
  const [wsReady, setWsReady] = useState(false);
  const [bleStatus, setBleStatus] = useState<string>("DISCONNECTED");
  const [isRecording, setIsRecording] = useState(false);
  const [label, setLabel] = useState("A");
  const [sampleCount, setSampleCount] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([
    mkLog("SmartPen Dataset Collector initialised"),
  ]);
  const [savedCount, setSavedCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const pushLog = useCallback(
    (text: string, type: LogEntry["type"] = "info") => {
      setLogs((prev) => [...prev.slice(-200), mkLog(text, type)]);
    },
    []
  );

  // ── WebSocket lifecycle ────────────────────────────
  const connect = useCallback(() => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    )
      return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsReady(true);
      pushLog("WebSocket connected to backend", "success");

      // Keep-alive ping
      pingTimer.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ command: "PING" }));
        }
      }, PING_INTERVAL);
    };

    ws.onmessage = (evt) => {
      try {
        const msg: WsMessage = JSON.parse(evt.data);
        handleServerMessage(msg);
      } catch {
        pushLog(`Unparseable message: ${evt.data}`, "error");
      }
    };

    ws.onclose = () => {
      setWsReady(false);
      setIsRecording(false);
      setBleStatus("DISCONNECTED");
      if (pingTimer.current) clearInterval(pingTimer.current);
      pushLog("WebSocket disconnected — retrying…", "error");

      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleServerMessage(msg: WsMessage) {
    switch (msg.status) {
      case "BLE_SCANNING":
        setBleStatus("BLE_SCANNING");
        pushLog("Scanning for SmartPen…");
        break;
      case "BLE_NOT_FOUND":
        setBleStatus("BLE_NOT_FOUND");
        pushLog("SmartPen not found — retrying", "error");
        break;
      case "BLE_CONNECTED":
        setBleStatus("BLE_CONNECTED");
        pushLog("BLE connected to SmartPen ✓", "success");
        break;
      case "BLE_DISCONNECTED":
        setBleStatus("BLE_DISCONNECTED");
        setIsRecording(false);
        pushLog("BLE connection lost", "error");
        break;
      case "RECORDING":
        setBleStatus("RECORDING");
        setIsRecording(true);
        if (msg.samples !== undefined) setSampleCount(msg.samples);
        if (msg.label) {
          pushLog(
            `Recording label "${msg.label}" — ${msg.samples ?? 0} samples`,
            "recording"
          );
        }
        break;
      case "SAVED":
        setBleStatus("BLE_CONNECTED");
        setIsRecording(false);
        setSavedCount((c) => c + 1);
        pushLog(
          `✅ Saved ${msg.filename}  (${msg.samples} samples, label="${msg.label}")`,
          "success"
        );
        break;
      case "EMPTY":
        setBleStatus("BLE_CONNECTED");
        setIsRecording(false);
        pushLog("⚠ Recording stopped — no samples captured", "error");
        break;
      case "ERROR":
        pushLog(`Server error: ${msg.message}`, "error");
        break;
      case "PONG":
        // Silent keep-alive
        break;
      default:
        pushLog(`Unknown status: ${msg.status}`);
    }
  }

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (pingTimer.current) clearInterval(pingTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Recording controls ─────────────────────────────
  const startRecording = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!label.trim()) {
      pushLog("Label is empty — please enter a label first", "error");
      return;
    }
    wsRef.current.send(
      JSON.stringify({ command: "START", label: label.trim() })
    );
    setSampleCount(0);
  }, [label, pushLog]);

  const stopRecording = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ command: "STOP" }));
  }, []);

  // Prevent context-menu on long-press (mobile)
  const preventContext = useCallback(
    (e: React.SyntheticEvent) => e.preventDefault(),
    []
  );

  // ── Derived UI state ───────────────────────────────
  const currentStatus =
    statusConfig[bleStatus] ?? statusConfig["DISCONNECTED"];
  const canRecord = wsReady && bleStatus === "BLE_CONNECTED" && !isRecording;
  const canRelease = wsReady && isRecording;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(145deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)",
        color: "#e0e0e0",
        fontFamily:
          "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "2rem 1rem",
      }}
    >
      {/* ─── Header ─────────────────────────────────── */}
      <header style={{ textAlign: "center", marginBottom: "2rem" }}>
        <h1
          style={{
            fontSize: "2.25rem",
            fontWeight: 800,
            background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            margin: 0,
            letterSpacing: "-0.02em",
          }}
        >
          SmartPen Dataset Collector
        </h1>
        <p style={{ color: "#94a3b8", marginTop: "0.5rem", fontSize: "0.95rem" }}>
          ESP32-C3 IMU · 6-Axis · BLE → WebSocket
        </p>
      </header>

      {/* ─── Main Card ──────────────────────────────── */}
      <div
        style={{
          width: "100%",
          maxWidth: "520px",
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: "1.25rem",
          backdropFilter: "blur(16px)",
          padding: "2rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
          boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
        }}
      >
        {/* Status Badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.6rem",
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              backgroundColor: currentStatus.color,
              display: "inline-block",
              boxShadow: `0 0 8px ${currentStatus.color}88`,
              animation: currentStatus.pulse
                ? "pulse-dot 1.4s ease-in-out infinite"
                : "none",
            }}
          />
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: currentStatus.color,
            }}
          >
            {currentStatus.label}
          </span>
          {isRecording && (
            <span
              style={{
                marginLeft: "0.75rem",
                fontSize: "0.8rem",
                color: "#c4b5fd",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {sampleCount} samples
            </span>
          )}
        </div>

        {/* Label Input */}
        <div>
          <label
            htmlFor="label-input"
            style={{
              display: "block",
              fontSize: "0.8rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "#94a3b8",
              marginBottom: "0.5rem",
            }}
          >
            Label (character / gesture)
          </label>
          <input
            id="label-input"
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={isRecording}
            maxLength={32}
            placeholder="e.g. A, B, circle…"
            style={{
              width: "100%",
              padding: "0.75rem 1rem",
              fontSize: "1.25rem",
              fontWeight: 700,
              textAlign: "center",
              letterSpacing: "0.15em",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "0.75rem",
              color: "#f1f5f9",
              outline: "none",
              transition: "border-color 0.2s, box-shadow 0.2s",
              boxSizing: "border-box",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "#8b5cf680";
              e.currentTarget.style.boxShadow = "0 0 0 3px #8b5cf630";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
              e.currentTarget.style.boxShadow = "none";
            }}
          />
        </div>

        {/* Record Button */}
        <button
          onMouseDown={canRecord ? startRecording : undefined}
          onMouseUp={canRelease ? stopRecording : undefined}
          onMouseLeave={canRelease ? stopRecording : undefined}
          onTouchStart={canRecord ? startRecording : undefined}
          onTouchEnd={canRelease ? stopRecording : undefined}
          onContextMenu={preventContext}
          disabled={!wsReady || bleStatus === "DISCONNECTED" || bleStatus === "BLE_SCANNING" || bleStatus === "BLE_NOT_FOUND" || bleStatus === "BLE_DISCONNECTED"}
          style={{
            position: "relative",
            padding: "1.25rem 2rem",
            fontSize: "1.1rem",
            fontWeight: 700,
            letterSpacing: "0.04em",
            border: "none",
            borderRadius: "1rem",
            cursor:
              canRecord || canRelease ? "pointer" : "not-allowed",
            color: "#fff",
            background: isRecording
              ? "linear-gradient(135deg, #7c3aed, #a855f7)"
              : canRecord
              ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
              : "rgba(255,255,255,0.08)",
            boxShadow: isRecording
              ? "0 0 24px #8b5cf666, inset 0 0 20px rgba(255,255,255,0.05)"
              : canRecord
              ? "0 4px 14px rgba(99,102,241,0.35)"
              : "none",
            transition: "all 0.2s ease",
            userSelect: "none",
            WebkitUserSelect: "none",
            touchAction: "manipulation",
            animation: isRecording ? "pulse-btn 1.2s ease-in-out infinite" : "none",
          }}
        >
          {isRecording
            ? `⏺  Recording "${label}" — Release to Stop`
            : canRecord
            ? "⏺  Press & Hold to Record"
            : "Waiting for BLE…"}
        </button>

        {/* Stats Row */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "2rem",
            fontSize: "0.8rem",
            color: "#64748b",
          }}
        >
          <span>
            Saved:{" "}
            <strong style={{ color: "#22c55e" }}>{savedCount}</strong>{" "}
            files
          </span>
          <span>
            WS:{" "}
            <strong style={{ color: wsReady ? "#22c55e" : "#ef4444" }}>
              {wsReady ? "OK" : "OFF"}
            </strong>
          </span>
        </div>
      </div>

      {/* ─── Log Terminal ───────────────────────────── */}
      <div
        style={{
          width: "100%",
          maxWidth: "520px",
          marginTop: "1.5rem",
          background: "rgba(0,0,0,0.35)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: "1rem",
          overflow: "hidden",
        }}
      >
        {/* Terminal header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            padding: "0.6rem 1rem",
            background: "rgba(255,255,255,0.03)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#ef4444",
            }}
          />
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#f59e0b",
            }}
          />
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#22c55e",
            }}
          />
          <span
            style={{
              marginLeft: "0.5rem",
              fontSize: "0.7rem",
              color: "#64748b",
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Event Log
          </span>
        </div>

        {/* Log body */}
        <div
          style={{
            height: "220px",
            overflowY: "auto",
            padding: "0.75rem 1rem",
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
            fontSize: "0.78rem",
            lineHeight: 1.7,
          }}
        >
          {logs.map((entry) => (
            <div key={entry.id} style={{ display: "flex", gap: "0.6rem" }}>
              <span style={{ color: "#475569", flexShrink: 0 }}>
                {entry.time}
              </span>
              <span
                style={{
                  color:
                    entry.type === "success"
                      ? "#4ade80"
                      : entry.type === "error"
                      ? "#f87171"
                      : entry.type === "recording"
                      ? "#c4b5fd"
                      : "#94a3b8",
                }}
              >
                {entry.text}
              </span>
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      </div>

      {/* ─── Animations (injected once) ─────────────── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.5; transform: scale(1.4); }
        }
        @keyframes pulse-btn {
          0%, 100% { box-shadow: 0 0 24px #8b5cf666, inset 0 0 20px rgba(255,255,255,0.05); }
          50%      { box-shadow: 0 0 40px #8b5cf699, inset 0 0 20px rgba(255,255,255,0.08); }
        }

        /* Scrollbar styling for the log */
        div::-webkit-scrollbar { width: 6px; }
        div::-webkit-scrollbar-track { background: transparent; }
        div::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.1);
          border-radius: 3px;
        }
        div::-webkit-scrollbar-thumb:hover {
          background: rgba(255,255,255,0.2);
        }

        /* Disable text selection on record button */
        button { -webkit-touch-callout: none; }

        input::placeholder { color: #475569; }
      `}</style>
    </div>
  );
}
