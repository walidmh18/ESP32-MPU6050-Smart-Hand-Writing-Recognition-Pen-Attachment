import React, { useState, useEffect, useRef, useCallback } from "react";

// ─── Types ───────────────────────────────────────────
type StatusType =
  | "DISCONNECTED"
  | "BLE_SCANNING"
  | "BLE_CONNECTING"
  | "BLE_NOT_FOUND"
  | "BLE_CONNECTED"
  | "BLE_DISCONNECTED"
  | "BLE_SCAN_RESULTS"
  | "RECORDING"
  | "SAVED"
  | "EMPTY"
  | "ERROR"
  | "LABEL_UPDATED"
  | "PONG";

interface DiscoveredDevice {
  name: string;
  address: string;
  rssi?: number;
}

interface WsMessage {
  status: StatusType;
  label?: string;
  folder?: string;
  id?: number;
  next_id?: number;
  next_file?: string;
  total_for_letter?: number;
  filename?: string;
  rel_path?: string;
  samples?: number;
  message?: string;
  ble_connected?: boolean;
  is_recording?: boolean;
  total_samples?: number;
  device_name?: string;
  device_address?: string;
  devices?: DiscoveredDevice[];
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
const QUICK_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const QUICK_DIGITS = "0123456789".split("");

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
  DISCONNECTED: { label: "Disconnected", color: "#94a3b8", pulse: false },
  BLE_SCANNING: { label: "Searching Pen…", color: "#f59e0b", pulse: true },
  BLE_CONNECTING: { label: "Connecting…", color: "#38bdf8", pulse: true },
  BLE_NOT_FOUND: { label: "Pen Not Found", color: "#f87171", pulse: false },
  BLE_CONNECTED: { label: "Connected", color: "#22c55e", pulse: false },
  BLE_DISCONNECTED: { label: "Disconnected", color: "#94a3b8", pulse: false },
  RECORDING: { label: "Recording", color: "#8b5cf6", pulse: true },
};

// ─── Component ───────────────────────────────────────
export default function SmartPenCollector() {
  const [wsReady, setWsReady] = useState(false);
  const [bleStatus, setBleStatus] = useState<string>("DISCONNECTED");
  const [isRecording, setIsRecording] = useState(false);
  const [label, setLabel] = useState("A");
  const labelRef = useRef(label);
  labelRef.current = label;

  const [sampleCount, setSampleCount] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([
    mkLog("SmartPen Dataset Collector initialised"),
  ]);
  const [savedCount, setSavedCount] = useState(0);

  // Folder & File Organization state
  const [folderInfo, setFolderInfo] = useState<{
    folder: string;
    nextId: number;
    nextFile: string;
    totalForLetter: number;
  }>({
    folder: "A",
    nextId: 1,
    nextFile: "A/dataset_A_1.csv",
    totalForLetter: 0,
  });

  // Manual BLE connection state
  const [targetDevice, setTargetDevice] = useState("SmartPen_ESP32C3");
  const [discoveredDevices, setDiscoveredDevices] = useState<DiscoveredDevice[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [showDevicePicker, setShowDevicePicker] = useState(false);
  const [connectedDevice, setConnectedDevice] = useState<{
    name: string;
    address: string;
  } | null>(null);

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

  const updateLabel = useCallback((newLabel: string) => {
    setLabel(newLabel);
    const trimmed = newLabel.trim();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && trimmed) {
      wsRef.current.send(JSON.stringify({ command: "SET_LABEL", label: trimmed }));
    }
  }, []);

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

      // Synchronize active letter immediately with backend
      const current = labelRef.current.trim() || "A";
      ws.send(JSON.stringify({ command: "SET_LABEL", label: current }));

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
      setConnectedDevice(null);
      setIsScanning(false);
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
    // If server provides folder & next ID info, keep UI updated
    if (msg.folder && msg.next_id !== undefined && msg.next_file) {
      setFolderInfo((prev) => ({
        folder: msg.folder!,
        nextId: msg.next_id!,
        nextFile: msg.next_file!,
        totalForLetter: msg.total_for_letter !== undefined ? msg.total_for_letter : prev.totalForLetter,
      }));
    }

    switch (msg.status) {
      case "LABEL_UPDATED":
        if (msg.message) {
          pushLog(msg.message, "info");
        }
        break;
      case "BLE_SCANNING":
        setBleStatus("BLE_SCANNING");
        pushLog(msg.message || "Scanning for SmartPen…");
        break;
      case "BLE_CONNECTING":
        setBleStatus("BLE_CONNECTING");
        pushLog(msg.message || "Connecting to SmartPen…");
        break;
      case "BLE_NOT_FOUND":
        setBleStatus("BLE_NOT_FOUND");
        setIsScanning(false);
        pushLog(msg.message || "SmartPen not found", "error");
        break;
      case "BLE_CONNECTED":
        setBleStatus("BLE_CONNECTED");
        setIsScanning(false);
        if (msg.device_name) {
          setConnectedDevice({
            name: msg.device_name,
            address: msg.device_address || "",
          });
        }
        pushLog(msg.message || "BLE connected to SmartPen ✓", "success");
        break;
      case "BLE_DISCONNECTED":
        setBleStatus("BLE_DISCONNECTED");
        setConnectedDevice(null);
        setIsRecording(false);
        setIsScanning(false);
        pushLog(msg.message || "BLE disconnected", "info");
        break;
      case "BLE_SCAN_RESULTS":
        setIsScanning(false);
        if (msg.devices) {
          setDiscoveredDevices(msg.devices);
          setShowDevicePicker(true);
          pushLog(
            msg.message || `Scan complete: ${msg.devices.length} devices found`,
            "info"
          );
        }
        break;
      case "RECORDING":
        setBleStatus("RECORDING");
        setIsRecording(true);
        if (msg.samples !== undefined) setSampleCount(msg.samples);
        if (msg.label) {
          pushLog(
            `Recording letter "${msg.label}" — ${msg.samples ?? 0} samples`,
            "recording"
          );
        }
        break;
      case "SAVED":
        setBleStatus("BLE_CONNECTED");
        setIsRecording(false);
        setSavedCount((c) => c + 1);
        if (msg.folder && msg.id !== undefined) {
          setFolderInfo((prev) => ({
            ...prev,
            folder: msg.folder!,
            nextId: msg.id! + 1,
            nextFile: `${msg.folder}/dataset_${msg.folder}_${msg.id! + 1}.csv`,
            totalForLetter: msg.total_for_letter !== undefined ? msg.total_for_letter : (prev.folder === msg.folder ? prev.totalForLetter + 1 : 1),
          }));
        }
        pushLog(
          `✅ Saved ${msg.rel_path || msg.filename} (${msg.samples} samples, ID: ${msg.id ?? "?"}, folder="${msg.folder || msg.label}")`,
          "success"
        );
        break;
      case "EMPTY":
        setBleStatus("BLE_CONNECTED");
        setIsRecording(false);
        pushLog("⚠ Recording stopped — no samples captured", "error");
        break;
      case "ERROR":
        setIsScanning(false);
        pushLog(`Server error: ${msg.message}`, "error");
        break;
      case "PONG":
        if (msg.ble_connected && msg.device_name) {
          setConnectedDevice({
            name: msg.device_name,
            address: msg.device_address || "",
          });
        }
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

  // ── Manual BLE Controls ─────────────────────────────
  const connectBle = useCallback(
    (customTarget?: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        pushLog("WebSocket not connected to backend", "error");
        return;
      }
      const t = (customTarget || targetDevice || "SmartPen_ESP32C3").trim();
      setBleStatus("BLE_SCANNING");
      pushLog(`Initiating manual connection to "${t}"…`, "info");
      wsRef.current.send(
        JSON.stringify({ command: "CONNECT_BLE", target: t })
      );
    },
    [targetDevice, pushLog]
  );

  const disconnectBle = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    pushLog("Initiating manual disconnect…", "info");
    wsRef.current.send(JSON.stringify({ command: "DISCONNECT_BLE" }));
  }, [pushLog]);

  const scanBle = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      pushLog("WebSocket not connected to backend", "error");
      return;
    }
    setIsScanning(true);
    setShowDevicePicker(true);
    pushLog("Scanning nearby BLE devices (4s)…", "info");
    wsRef.current.send(JSON.stringify({ command: "SCAN_BLE" }));
  }, [pushLog]);

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
  const isBleConnected = bleStatus === "BLE_CONNECTED";
  const isBleBusy =
    bleStatus === "BLE_SCANNING" || bleStatus === "BLE_CONNECTING" || isScanning;
  const canRecord = wsReady && isBleConnected && !isRecording;
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
          ESP32-C3 IMU · 6-Axis · Manual BLE Connect
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
          gap: "1.25rem",
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

        {/* ─── Manual BLE Connection Panel ───────────── */}
        <div
          style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "0.875rem",
            padding: "1rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
          }}
        >
          {isBleConnected ? (
            /* Connected state banner */
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "0.75rem",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: "rgba(34, 197, 94, 0.15)",
                    border: "1px solid rgba(34, 197, 94, 0.4)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1rem",
                  }}
                >
                  🖊
                </div>
                <div>
                  <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "#22c55e" }}>
                    {connectedDevice?.name || targetDevice}
                  </div>
                  {connectedDevice?.address && (
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
                      {connectedDevice.address}
                    </div>
                  )}
                </div>
              </div>

              <button
                type="button"
                onClick={disconnectBle}
                style={{
                  padding: "0.45rem 0.85rem",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "#f87171",
                  background: "rgba(239, 68, 68, 0.1)",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  borderRadius: "0.5rem",
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = "rgba(239, 68, 68, 0.2)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = "rgba(239, 68, 68, 0.1)";
                }}
              >
                Disconnect
              </button>
            </div>
          ) : (
            /* Disconnected state controls */
            <>
              <div>
                <label
                  htmlFor="target-device-input"
                  style={{
                    display: "block",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "#94a3b8",
                    marginBottom: "0.4rem",
                  }}
                >
                  BLE Target Device (Name or MAC)
                </label>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <input
                    id="target-device-input"
                    type="text"
                    value={targetDevice}
                    onChange={(e) => setTargetDevice(e.target.value)}
                    disabled={isBleBusy}
                    placeholder="e.g. SmartPen_ESP32C3 or MAC"
                    style={{
                      flex: 1,
                      padding: "0.5rem 0.75rem",
                      fontSize: "0.9rem",
                      fontWeight: 600,
                      background: "rgba(255,255,255,0.06)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: "0.5rem",
                      color: "#f1f5f9",
                      outline: "none",
                    }}
                  />
                  <button
                    type="button"
                    onClick={scanBle}
                    disabled={!wsReady || isBleBusy}
                    title="Scan for nearby Bluetooth devices"
                    style={{
                      padding: "0.5rem 0.85rem",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      color: "#cbd5e1",
                      background: "rgba(255,255,255,0.08)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: "0.5rem",
                      cursor: wsReady && !isBleBusy ? "pointer" : "not-allowed",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {isScanning ? "Scanning…" : "🔍 Scan"}
                  </button>
                </div>
              </div>

              {/* Connect Button */}
              <button
                type="button"
                onClick={() => connectBle()}
                disabled={!wsReady || isBleBusy}
                style={{
                  padding: "0.7rem 1rem",
                  fontSize: "0.95rem",
                  fontWeight: 700,
                  letterSpacing: "0.02em",
                  border: "none",
                  borderRadius: "0.6rem",
                  cursor: wsReady && !isBleBusy ? "pointer" : "not-allowed",
                  color: "#fff",
                  background: isBleBusy
                    ? "rgba(99,102,241,0.5)"
                    : "linear-gradient(135deg, #4f46e5, #06b6d4)",
                  boxShadow: isBleBusy
                    ? "none"
                    : "0 4px 12px rgba(79, 70, 229, 0.3)",
                  transition: "all 0.2s ease",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "0.5rem",
                }}
              >
                {bleStatus === "BLE_CONNECTING"
                  ? "⚡ Connecting to SmartPen…"
                  : bleStatus === "BLE_SCANNING"
                  ? "🔍 Searching for SmartPen…"
                  : "⚡ Connect SmartPen"}
              </button>

              {/* Discovered devices dropdown list */}
              {showDevicePicker && discoveredDevices.length > 0 && (
                <div
                  style={{
                    marginTop: "0.5rem",
                    padding: "0.5rem",
                    background: "rgba(0,0,0,0.3)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "0.5rem",
                    maxHeight: "150px",
                    overflowY: "auto",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.7rem",
                      fontWeight: 600,
                      color: "#94a3b8",
                      marginBottom: "0.3rem",
                      textTransform: "uppercase",
                    }}
                  >
                    Nearby Devices (Click to select & connect):
                  </div>
                  {discoveredDevices.map((dev) => (
                    <div
                      key={dev.address}
                      onClick={() => {
                        setTargetDevice(dev.name || dev.address);
                        connectBle(dev.address);
                      }}
                      style={{
                        padding: "0.4rem 0.6rem",
                        borderRadius: "0.375rem",
                        cursor: "pointer",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        fontSize: "0.8rem",
                        background:
                          dev.name.includes("SmartPen")
                            ? "rgba(139, 92, 246, 0.15)"
                            : "transparent",
                        border:
                          dev.name.includes("SmartPen")
                            ? "1px solid rgba(139, 92, 246, 0.3)"
                            : "none",
                        marginBottom: "0.25rem",
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "rgba(255,255,255,0.08)";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.backgroundColor = dev.name.includes("SmartPen")
                          ? "rgba(139, 92, 246, 0.15)"
                          : "transparent";
                      }}
                    >
                      <div>
                        <span style={{ fontWeight: 600, color: "#f8fafc" }}>
                          {dev.name}
                        </span>
                        <span
                          style={{
                            marginLeft: "0.5rem",
                            fontSize: "0.7rem",
                            color: "#64748b",
                          }}
                        >
                          {dev.address}
                        </span>
                      </div>
                      {dev.rssi !== undefined && (
                        <span style={{ fontSize: "0.7rem", color: "#38bdf8" }}>
                          {dev.rssi} dBm
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {/* Letter & Folder Selector Section */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <label
              htmlFor="label-input"
              style={{
                fontSize: "0.8rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                color: "#94a3b8",
              }}
            >
              Letter / Folder Selector
            </label>
            <span style={{ fontSize: "0.75rem", color: "#a78bfa", fontWeight: 700, fontFamily: "monospace" }}>
              datasets/{folderInfo.folder}/
            </span>
          </div>

          {/* Quick Alphabet Selector */}
          <div>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "0.25rem",
                maxHeight: "96px",
                overflowY: "auto",
                padding: "0.35rem",
                background: "rgba(0,0,0,0.3)",
                borderRadius: "0.6rem",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              {QUICK_LETTERS.map((char) => {
                const isSelected = label.toUpperCase() === char;
                return (
                  <button
                    key={char}
                    type="button"
                    onClick={() => updateLabel(char)}
                    disabled={isRecording}
                    style={{
                      width: "31px",
                      height: "31px",
                      fontSize: "0.85rem",
                      fontWeight: isSelected ? 800 : 600,
                      borderRadius: "0.375rem",
                      border: isSelected
                        ? "1px solid #8b5cf6"
                        : "1px solid rgba(255,255,255,0.08)",
                      background: isSelected
                        ? "linear-gradient(135deg, #7c3aed, #06b6d4)"
                        : "rgba(255,255,255,0.05)",
                      color: isSelected ? "#ffffff" : "#cbd5e1",
                      cursor: isRecording ? "not-allowed" : "pointer",
                      boxShadow: isSelected ? "0 0 10px rgba(139,92,246,0.5)" : "none",
                      transition: "all 0.15s ease",
                      padding: 0,
                    }}
                  >
                    {char}
                  </button>
                );
              })}
            </div>

            {/* Quick Digits row */}
            <div
              style={{
                display: "flex",
                gap: "0.25rem",
                marginTop: "0.35rem",
                overflowX: "auto",
              }}
            >
              {QUICK_DIGITS.map((digit) => {
                const isSelected = label === digit;
                return (
                  <button
                    key={digit}
                    type="button"
                    onClick={() => updateLabel(digit)}
                    disabled={isRecording}
                    style={{
                      flex: 1,
                      height: "26px",
                      fontSize: "0.8rem",
                      fontWeight: isSelected ? 800 : 600,
                      borderRadius: "0.375rem",
                      border: isSelected
                        ? "1px solid #38bdf8"
                        : "1px solid rgba(255,255,255,0.08)",
                      background: isSelected
                        ? "linear-gradient(135deg, #0284c7, #06b6d4)"
                        : "rgba(255,255,255,0.04)",
                      color: isSelected ? "#ffffff" : "#94a3b8",
                      cursor: isRecording ? "not-allowed" : "pointer",
                      transition: "all 0.15s ease",
                      padding: 0,
                    }}
                  >
                    {digit}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Custom text input */}
          <input
            id="label-input"
            type="text"
            value={label}
            onChange={(e) => updateLabel(e.target.value)}
            disabled={isRecording}
            maxLength={32}
            placeholder="e.g. A, B, circle…"
            style={{
              width: "100%",
              padding: "0.65rem 1rem",
              fontSize: "1.15rem",
              fontWeight: 700,
              textAlign: "center",
              letterSpacing: "0.15em",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "0.65rem",
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

          {/* Folder & Next File Info Card */}
          <div
            style={{
              background: "rgba(139, 92, 246, 0.08)",
              border: "1px solid rgba(139, 92, 246, 0.25)",
              borderRadius: "0.75rem",
              padding: "0.75rem 1rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.45rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                <span style={{ fontSize: "1rem" }}>📁</span>
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "#94a3b8",
                    fontWeight: 600,
                    textTransform: "uppercase",
                  }}
                >
                  Target Folder:
                </span>
                <span
                  style={{
                    fontSize: "0.85rem",
                    color: "#c4b5fd",
                    fontWeight: 700,
                    fontFamily: "monospace",
                  }}
                >
                  datasets/{folderInfo.folder}/
                </span>
              </div>
              <span
                style={{
                  fontSize: "0.7rem",
                  color: "#38bdf8",
                  background: "rgba(56, 189, 248, 0.12)",
                  padding: "0.2rem 0.5rem",
                  borderRadius: "9999px",
                  fontWeight: 600,
                }}
              >
                {folderInfo.totalForLetter} files saved
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
              <span style={{ fontSize: "1rem" }}>📄</span>
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "#94a3b8",
                  fontWeight: 600,
                  textTransform: "uppercase",
                }}
              >
                Next File:
              </span>
              <span
                style={{
                  fontSize: "0.92rem",
                  color: "#34d399",
                  fontWeight: 800,
                  fontFamily: "monospace",
                }}
              >
                dataset_{folderInfo.folder}_{folderInfo.nextId}.csv
              </span>
            </div>

            <div
              style={{
                fontSize: "0.72rem",
                color: "#94a3b8",
                borderTop: "1px solid rgba(255,255,255,0.06)",
                paddingTop: "0.4rem",
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
              }}
            >
              <span>🖊️</span>
              <span>
                <strong>Pen Button:</strong> Press physical pen button anytime to capture — saves as <strong>dataset_{folderInfo.folder}_{folderInfo.nextId}.csv</strong>!
              </span>
            </div>
          </div>
        </div>

        {/* Record Button */}
        <button
          onMouseDown={canRecord ? startRecording : undefined}
          onMouseUp={canRelease ? stopRecording : undefined}
          onMouseLeave={canRelease ? stopRecording : undefined}
          onTouchStart={canRecord ? startRecording : undefined}
          onTouchEnd={canRelease ? stopRecording : undefined}
          onContextMenu={preventContext}
          disabled={!canRecord && !canRelease}
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
            : !isBleConnected
            ? "Connect SmartPen to Record"
            : "Waiting…"}
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
