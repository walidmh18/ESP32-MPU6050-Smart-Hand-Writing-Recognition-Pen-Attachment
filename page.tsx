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
  | "MODE_UPDATED"
  | "PREDICTION_RESULT"
  | "PONG";

type AppMode = "COLLECT" | "PREDICT";

interface DiscoveredDevice {
  name: string;
  address: string;
  rssi?: number;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities?: Record<string, number>;
  samples?: number;
  capture_number?: number;
  timestamp?: string;
  source_file?: string;
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
  mode?: AppMode;
  predictor_ready?: boolean;
  prediction?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
  capture_number?: number;
  timestamp?: string;
  source_file?: string;
  last_prediction?: PredictionResult;
}

interface LogEntry {
  id: number;
  time: string;
  text: string;
  type: "info" | "success" | "error" | "recording" | "prediction";
}

// ─── Constants ───────────────────────────────────────
const WS_URL = "ws://localhost:8000/ws";
const RECONNECT_DELAY = 3000;
const PING_INTERVAL = 5000;
const QUICK_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const QUICK_DIGITS = "0123456789".split("");

// ─── Helpers ─────────────────────────────────────────
let logId = 0;
function mkLog(text: string, type: LogEntry["type"] = "info"): LogEntry {
  return {
    id: logId++,
    time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    text,
    type,
  };
}

const statusConfig: Record<string, { label: string; color: string; pulse: boolean }> = {
  DISCONNECTED: { label: "Disconnected", color: "#94a3b8", pulse: false },
  BLE_SCANNING: { label: "Searching Pen…", color: "#f59e0b", pulse: true },
  BLE_CONNECTING: { label: "Connecting…", color: "#38bdf8", pulse: true },
  BLE_NOT_FOUND: { label: "Pen Not Found", color: "#f87171", pulse: false },
  BLE_CONNECTED: { label: "Connected", color: "#22c55e", pulse: false },
  BLE_DISCONNECTED: { label: "Disconnected", color: "#94a3b8", pulse: false },
  RECORDING: { label: "Capturing…", color: "#ec4899", pulse: true },
};

export default function SmartPenCollector() {
  const [wsReady, setWsReady] = useState(false);
  const [bleStatus, setBleStatus] = useState<string>("DISCONNECTED");
  const [isRecording, setIsRecording] = useState(false);
  const [appMode, setAppMode] = useState<AppMode>("COLLECT");

  // Collection State
  const [label, setLabel] = useState("A");
  const labelRef = useRef(label);
  labelRef.current = label;

  const [sampleCount, setSampleCount] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([
    mkLog("SmartPen Studio initialised — full cockpit ready"),
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

  // Prediction State
  const [latestPrediction, setLatestPrediction] = useState<PredictionResult | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionResult[]>([]);
  const [predictorReady, setPredictorReady] = useState(true);

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

  const pushLog = useCallback((text: string, type: LogEntry["type"] = "info") => {
    setLogs((prev) => [...prev.slice(-200), mkLog(text, type)]);
  }, []);

  const updateLabel = useCallback((newLabel: string) => {
    setLabel(newLabel);
    const trimmed = newLabel.trim();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && trimmed) {
      wsRef.current.send(JSON.stringify({ command: "SET_LABEL", label: trimmed }));
    }
  }, []);

  const switchMode = useCallback((mode: AppMode) => {
    setAppMode(mode);
    setIsRecording(false);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "SET_MODE", mode }));
    }
    pushLog(
      mode === "PREDICT"
        ? "Switched to 🔮 Live Prediction Mode"
        : "Switched to 📊 Data Collection Mode",
      "info"
    );
  }, [pushLog]);

  const triggerTestPrediction = useCallback((targetClass: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      pushLog("WebSocket not connected to backend", "error");
      return;
    }
    pushLog(`🧪 Running model test on class '${targetClass}' sample…`, "info");
    wsRef.current.send(JSON.stringify({ command: "TEST_PREDICTION", class: targetClass }));
  }, [pushLog]);

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
      pushLog("WebSocket connected to backend ✓", "success");

      const current = labelRef.current.trim() || "A";
      ws.send(JSON.stringify({ command: "SET_LABEL", label: current }));

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
    if (msg.folder && msg.next_id !== undefined && msg.next_file) {
      setFolderInfo((prev) => ({
        folder: msg.folder!,
        nextId: msg.next_id!,
        nextFile: msg.next_file!,
        totalForLetter:
          msg.total_for_letter !== undefined ? msg.total_for_letter : prev.totalForLetter,
      }));
    }

    if (msg.mode && (msg.mode === "COLLECT" || msg.mode === "PREDICT")) {
      setAppMode(msg.mode);
    }
    if (msg.predictor_ready !== undefined) {
      setPredictorReady(msg.predictor_ready);
    }

    switch (msg.status) {
      case "MODE_UPDATED":
        if (msg.mode) setAppMode(msg.mode);
        if (msg.message) pushLog(msg.message, "info");
        break;

      case "PREDICTION_RESULT":
        setIsRecording(false);
        if (msg.prediction !== undefined) {
          const res: PredictionResult = {
            prediction: msg.prediction,
            confidence: msg.confidence ?? 1.0,
            probabilities: msg.probabilities,
            samples: msg.samples,
            capture_number: msg.capture_number,
            timestamp: msg.timestamp || new Date().toLocaleTimeString("en-GB", { hour12: false }),
            source_file: msg.source_file,
          };
          setLatestPrediction(res);
          setPredictionHistory((prev) => [res, ...prev.slice(0, 19)]);
          pushLog(
            `🔮 AI Recognized: "${msg.prediction}" (${((msg.confidence ?? 1) * 100).toFixed(1)}% match)`,
            "prediction"
          );
        }
        break;

      case "LABEL_UPDATED":
        if (msg.message) pushLog(msg.message, "info");
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
        if (msg.message) pushLog(msg.message, "recording");
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
            totalForLetter:
              msg.total_for_letter !== undefined
                ? msg.total_for_letter
                : prev.folder === msg.folder
                ? prev.totalForLetter + 1
                : 1,
          }));
        }
        pushLog(
          `✅ Saved ${msg.rel_path || msg.filename} (${msg.samples} samples, ID: ${msg.id ?? "?"})`,
          "success"
        );
        break;

      case "EMPTY":
        setBleStatus("BLE_CONNECTED");
        setIsRecording(false);
        pushLog("⚠ Capture idle — no samples received", "info");
        break;

      case "ERROR":
        setIsScanning(false);
        setIsRecording(false);
        pushLog(`Error: ${msg.message}`, "error");
        break;

      case "PONG":
        if (msg.mode) setAppMode(msg.mode);
        if (msg.last_prediction && !latestPrediction) {
          setLatestPrediction(msg.last_prediction);
        }
        if (msg.ble_connected && msg.device_name) {
          setConnectedDevice({
            name: msg.device_name,
            address: msg.device_address || "",
          });
        }
        break;

      default:
        pushLog(`Unknown message: ${msg.status}`);
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
      pushLog(`Initiating connection to "${t}"…`, "info");
      wsRef.current.send(JSON.stringify({ command: "CONNECT_BLE", target: t }));
    },
    [targetDevice, pushLog]
  );

  const disconnectBle = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    pushLog("Disconnecting SmartPen…", "info");
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

  // ── Capture & Recording controls ───────────────────
  const startCapture = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    if (appMode === "COLLECT") {
      if (!label.trim()) {
        pushLog("Label is empty — please enter a label first", "error");
        return;
      }
      wsRef.current.send(JSON.stringify({ command: "START", label: label.trim() }));
    } else {
      wsRef.current.send(JSON.stringify({ command: "START" }));
    }
    setSampleCount(0);
  }, [appMode, label, pushLog]);

  const stopCapture = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ command: "STOP" }));
  }, []);

  const handleActionButtonClick = useCallback(() => {
    if (isRecording) {
      stopCapture();
    } else {
      startCapture();
    }
  }, [isRecording, startCapture, stopCapture]);

  const preventContext = useCallback((e: React.SyntheticEvent) => e.preventDefault(), []);

  // ── Derived UI state ───────────────────────────────
  const currentStatus = statusConfig[bleStatus] ?? statusConfig["DISCONNECTED"];
  const isBleConnected = bleStatus === "BLE_CONNECTED";
  const isBleBusy = bleStatus === "BLE_SCANNING" || bleStatus === "BLE_CONNECTING" || isScanning;

  return (
    <div
      style={{
        height: "100vh",
        maxHeight: "100vh",
        width: "100vw",
        overflow: "hidden",
        background: "radial-gradient(ellipse at 50% 0%, #1a1740 0%, #0c0c16 60%, #06060a 100%)",
        color: "#e2e8f0",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        display: "flex",
        flexDirection: "column",
        boxSizing: "border-box",
      }}
    >
      {/* ─── Compact Single-Line Top Navigation Bar ──── */}
      <header
        style={{
          height: "50px",
          minHeight: "50px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 1.25rem",
          background: "rgba(15, 15, 26, 0.75)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          backdropFilter: "blur(12px)",
          flexShrink: 0,
        }}
      >
        {/* Brand / Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <div
            style={{
              width: "28px",
              height: "28px",
              borderRadius: "0.5rem",
              background: "linear-gradient(135deg, #8b5cf6, #ec4899)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "0.9rem",
              boxShadow: "0 0 12px rgba(139, 92, 246, 0.5)",
            }}
          >
            🖊
          </div>
          <span
            style={{
              fontWeight: 800,
              fontSize: "1.1rem",
              background: "linear-gradient(135deg, #f1f5f9, #c4b5fd)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: "-0.01em",
            }}
          >
            SmartPen Studio
          </span>
          <span
            style={{
              fontSize: "0.7rem",
              padding: "0.15rem 0.45rem",
              borderRadius: "9999px",
              background: "rgba(139, 92, 246, 0.15)",
              color: "#c084fc",
              fontWeight: 600,
              border: "1px solid rgba(139, 92, 246, 0.3)",
            }}
          >
            6-Axis IMU
          </span>
        </div>

        {/* Center Mode Segmented Switch */}
        <div
          style={{
            display: "flex",
            gap: "0.25rem",
            padding: "0.2rem",
            background: "rgba(0, 0, 0, 0.4)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "0.6rem",
          }}
        >
          <button
            type="button"
            onClick={() => switchMode("COLLECT")}
            style={{
              padding: "0.35rem 0.85rem",
              borderRadius: "0.45rem",
              border: "none",
              background:
                appMode === "COLLECT"
                  ? "linear-gradient(135deg, #7c3aed, #4f46e5)"
                  : "transparent",
              color: appMode === "COLLECT" ? "#ffffff" : "#94a3b8",
              fontWeight: appMode === "COLLECT" ? 700 : 500,
              fontSize: "0.82rem",
              cursor: "pointer",
              transition: "all 0.15s ease",
              boxShadow:
                appMode === "COLLECT" ? "0 2px 8px rgba(124, 58, 237, 0.4)" : "none",
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            <span>📁</span>
            <span>Data Collection</span>
          </button>

          <button
            type="button"
            onClick={() => switchMode("PREDICT")}
            style={{
              padding: "0.35rem 0.85rem",
              borderRadius: "0.45rem",
              border: "none",
              background:
                appMode === "PREDICT"
                  ? "linear-gradient(135deg, #db2777, #9333ea)"
                  : "transparent",
              color: appMode === "PREDICT" ? "#ffffff" : "#94a3b8",
              fontWeight: appMode === "PREDICT" ? 700 : 500,
              fontSize: "0.82rem",
              cursor: "pointer",
              transition: "all 0.15s ease",
              boxShadow:
                appMode === "PREDICT" ? "0 2px 8px rgba(219, 39, 119, 0.4)" : "none",
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            <span>🔮</span>
            <span>Live Prediction</span>
            <span
              style={{
                fontSize: "0.62rem",
                padding: "0.05rem 0.35rem",
                borderRadius: "9999px",
                background: "rgba(255, 255, 255, 0.25)",
                color: "#fff",
                fontWeight: 800,
              }}
            >
              AI
            </span>
          </button>
        </div>

        {/* Right Status Badge */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {isRecording && (
            <span
              style={{
                fontSize: "0.74rem",
                color: "#f472b6",
                fontWeight: 700,
                fontVariantNumeric: "tabular-nums",
                background: "rgba(244, 114, 182, 0.15)",
                padding: "0.15rem 0.5rem",
                borderRadius: "9999px",
              }}
            >
              ● {sampleCount}/500 pts
            </span>
          )}

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.45rem",
              background: "rgba(0, 0, 0, 0.35)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "0.25rem 0.65rem",
              borderRadius: "0.5rem",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: currentStatus.color,
                display: "inline-block",
                boxShadow: `0 0 8px ${currentStatus.color}`,
                animation: currentStatus.pulse ? "pulse-dot 1.3s ease-in-out infinite" : "none",
              }}
            />
            <span
              style={{
                fontSize: "0.76rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: currentStatus.color,
              }}
            >
              {currentStatus.label}
            </span>
          </div>

          <span
            style={{
              fontSize: "0.74rem",
              fontWeight: 600,
              color: wsReady ? "#4ade80" : "#f87171",
            }}
          >
            {wsReady ? "Online" : "Connecting…"}
          </span>
        </div>
      </header>

      {/* ─── 2-Column Cockpit Grid (Fits Screen, Zero Outer Scroll) ─── */}
      <main
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "1.15fr 0.85fr",
          gap: "0.85rem",
          padding: "0.75rem 1rem",
          boxSizing: "border-box",
          overflow: "hidden",
        }}
      >
        {/* ═══════════════════════════════════════════════════
            LEFT COLUMN: INTERACTIVE WORKSPACE (MODE SPECIFIC)
            ═══════════════════════════════════════════════════ */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
            minHeight: 0,
            overflowY: "auto",
            paddingRight: "0.2rem",
          }}
        >
          {/* ──── PREDICTION MODE VIEW ──── */}
          {appMode === "PREDICT" && (
            <>
              {/* Hero Prediction Display Card (Enlarged) */}
              <div
                style={{
                  background:
                    "radial-gradient(ellipse at 50% 25%, rgba(139, 92, 246, 0.16) 0%, rgba(15, 23, 42, 0.6) 100%)",
                  border: "1px solid rgba(139, 92, 246, 0.38)",
                  borderRadius: "1.15rem",
                  padding: "1.1rem 1.25rem",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.85rem",
                  position: "relative",
                  boxShadow: "0 8px 28px rgba(139, 92, 246, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
                    paddingBottom: "0.45rem",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                    <span style={{ fontSize: "0.9rem" }}>🧠</span>
                    <span
                      style={{
                        fontSize: "0.78rem",
                        fontWeight: 800,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        color: "#c084fc",
                      }}
                    >
                      Neural Air-Writing Prediction
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      color: predictorReady ? "#4ade80" : "#f87171",
                      fontWeight: 700,
                      background: predictorReady ? "rgba(34, 197, 94, 0.12)" : "rgba(239, 68, 68, 0.12)",
                      padding: "0.15rem 0.5rem",
                      borderRadius: "9999px",
                      border: predictorReady ? "1px solid rgba(34, 197, 94, 0.3)" : "1px solid rgba(239, 68, 68, 0.3)",
                    }}
                  >
                    {predictorReady ? "● 1D-CNN + LSTM Active" : "● Model Loading"}
                  </span>
                </div>

                {latestPrediction ? (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "1.35rem",
                      animation: "pop-in 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
                      padding: "0.2rem 0",
                    }}
                  >
                    {/* Enlarged Glowing Character Orb */}
                    <div
                      style={{
                        width: "100px",
                        height: "100px",
                        borderRadius: "50%",
                        background:
                          "linear-gradient(135deg, rgba(168, 85, 247, 0.35), rgba(56, 189, 248, 0.3))",
                        border: "2.5px solid #c084fc",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        boxShadow:
                          "0 0 35px rgba(168, 85, 247, 0.55), inset 0 0 15px rgba(255, 255, 255, 0.25)",
                        flexShrink: 0,
                      }}
                    >
                      <span
                        style={{
                          fontSize: "4.4rem",
                          fontWeight: 900,
                          lineHeight: 1,
                          background: "linear-gradient(135deg, #ffffff 40%, #e9d5ff 100%)",
                          WebkitBackgroundClip: "text",
                          WebkitTextFillColor: "transparent",
                          filter: "drop-shadow(0 2px 8px rgba(0, 0, 0, 0.5))",
                        }}
                      >
                        {latestPrediction.prediction}
                      </span>
                    </div>

                    {/* Stats details */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "0.4rem",
                        flex: 1,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.75rem",
                          flexWrap: "wrap",
                        }}
                      >
                        <span style={{ fontSize: "1.4rem", fontWeight: 900, color: "#fff", letterSpacing: "-0.01em" }}>
                          Digit &quot;{latestPrediction.prediction}&quot;
                        </span>
                        <span
                          style={{
                            padding: "0.25rem 0.75rem",
                            borderRadius: "9999px",
                            fontSize: "0.85rem",
                            fontWeight: 800,
                            background:
                              latestPrediction.confidence >= 0.85
                                ? "rgba(34, 197, 94, 0.22)"
                                : "rgba(245, 158, 11, 0.22)",
                            color:
                              latestPrediction.confidence >= 0.85 ? "#4ade80" : "#fbbf24",
                            border:
                              latestPrediction.confidence >= 0.85
                                ? "1px solid rgba(34, 197, 94, 0.45)"
                                : "1px solid rgba(245, 158, 11, 0.45)",
                          }}
                        >
                          {(latestPrediction.confidence * 100).toFixed(1)}% Match
                        </span>
                      </div>

                      <div style={{ fontSize: "0.78rem", color: "#94a3b8", lineHeight: 1.5 }}>
                        Captured at <strong style={{ color: "#cbd5e1" }}>{latestPrediction.timestamp}</strong> · Window: <strong style={{ color: "#cbd5e1" }}>500 pts @ 500 Hz (1.0s)</strong>
                        {latestPrediction.source_file && (
                          <div style={{ fontSize: "0.72rem", color: "#64748b", marginTop: "0.1rem" }}>
                            Source: {latestPrediction.source_file}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "1.1rem",
                      padding: "0.6rem 0.2rem",
                    }}
                  >
                    <div
                      style={{
                        width: "72px",
                        height: "72px",
                        borderRadius: "50%",
                        background: "rgba(139, 92, 246, 0.12)",
                        border: "1.5px dashed rgba(139, 92, 246, 0.5)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "2.1rem",
                        animation: "pulse-dot 2s infinite ease-in-out",
                        flexShrink: 0,
                      }}
                    >
                      ✍️
                    </div>
                    <div>
                      <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "#f1f5f9" }}>
                        Ready for Air-Writing Stroke
                      </div>
                      <div style={{ fontSize: "0.82rem", color: "#94a3b8", marginTop: "0.2rem", lineHeight: 1.45 }}>
                        Draw digit <strong>0</strong> or <strong>1</strong> in the air. Press the button on the pen anytime or click Capture below!
                      </div>
                    </div>
                  </div>
                )}

                {/* Softmax Probability Bars */}
                {latestPrediction?.probabilities && (
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.4rem",
                      background: "rgba(0, 0, 0, 0.35)",
                      border: "1px solid rgba(255, 255, 255, 0.06)",
                      borderRadius: "0.7rem",
                      padding: "0.65rem 0.85rem",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 800,
                        color: "#94a3b8",
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                      }}
                    >
                      Softmax Output Distribution
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                      {Object.entries(latestPrediction.probabilities).map(
                        ([clsName, prob]) => {
                          const isWinner = clsName === latestPrediction.prediction;
                          const pct = Math.round(prob * 1000) / 10;
                          return (
                            <div key={clsName}>
                              <div
                                style={{
                                  display: "flex",
                                  justifyContent: "space-between",
                                  fontSize: "0.78rem",
                                  marginBottom: "0.2rem",
                                }}
                              >
                                <span style={{ fontWeight: isWinner ? 800 : 600, color: isWinner ? "#c084fc" : "#cbd5e1" }}>
                                  Class &quot;{clsName}&quot; {isWinner && "✓"}
                                </span>
                                <span style={{ fontWeight: isWinner ? 800 : 600, color: isWinner ? "#4ade80" : "#94a3b8" }}>
                                  {pct}%
                                </span>
                              </div>
                              <div
                                style={{
                                  height: "7px",
                                  background: "rgba(255, 255, 255, 0.08)",
                                  borderRadius: "4px",
                                  overflow: "hidden",
                                }}
                              >
                                <div
                                  style={{
                                    width: `${Math.min(100, Math.max(0, pct))}%`,
                                    height: "100%",
                                    borderRadius: "4px",
                                    background: isWinner
                                      ? "linear-gradient(90deg, #8b5cf6, #38bdf8)"
                                      : "rgba(255, 255, 255, 0.2)",
                                    transition: "width 0.4s ease-out",
                                  }}
                                />
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons Row */}
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  type="button"
                  onClick={handleActionButtonClick}
                  disabled={!wsReady}
                  style={{
                    flex: 1,
                    padding: "0.85rem 1rem",
                    fontSize: "0.95rem",
                    fontWeight: 700,
                    letterSpacing: "0.02em",
                    border: "none",
                    borderRadius: "0.75rem",
                    cursor: wsReady ? "pointer" : "not-allowed",
                    color: "#fff",
                    background: isRecording
                      ? "linear-gradient(135deg, #db2777, #f43f5e)"
                      : "linear-gradient(135deg, #7c3aed, #db2777)",
                    boxShadow: isRecording
                      ? "0 0 20px rgba(244, 63, 94, 0.5)"
                      : "0 4px 14px rgba(124, 58, 237, 0.35)",
                    transition: "all 0.2s ease",
                    animation: isRecording ? "pulse-btn 1.2s ease-in-out infinite" : "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.45rem",
                  }}
                >
                  {isRecording ? (
                    <>
                      <span>🔮</span>
                      <span>Listening for Stroke… Draw Now! (Click to Cancel)</span>
                    </>
                  ) : (
                    <>
                      <span>⚡</span>
                      <span>Capture &amp; Predict Stroke</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => triggerTestPrediction("0")}
                  disabled={!wsReady}
                  title="Run inference test on a sample digit 0"
                  style={{
                    padding: "0 0.85rem",
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    borderRadius: "0.75rem",
                    border: "1px solid rgba(56, 189, 248, 0.35)",
                    background: "rgba(56, 189, 248, 0.08)",
                    color: "#38bdf8",
                    cursor: wsReady ? "pointer" : "not-allowed",
                  }}
                >
                  Test &quot;0&quot;
                </button>

                <button
                  type="button"
                  onClick={() => triggerTestPrediction("1")}
                  disabled={!wsReady}
                  title="Run inference test on a sample digit 1"
                  style={{
                    padding: "0 0.85rem",
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    borderRadius: "0.75rem",
                    border: "1px solid rgba(168, 85, 247, 0.35)",
                    background: "rgba(168, 85, 247, 0.08)",
                    color: "#c084fc",
                    cursor: wsReady ? "pointer" : "not-allowed",
                  }}
                >
                  Test &quot;1&quot;
                </button>
              </div>

              {/* Recent Predictions History (Small & Scrollable) */}
              <div
                style={{
                  background: "rgba(255, 255, 255, 0.02)",
                  border: "1px solid rgba(255, 255, 255, 0.06)",
                  borderRadius: "0.75rem",
                  padding: "0.45rem 0.75rem",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.3rem",
                  height: "90px",
                  maxHeight: "90px",
                  flexShrink: 0,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    fontSize: "0.68rem",
                    color: "#94a3b8",
                    fontWeight: 700,
                    textTransform: "uppercase",
                  }}
                >
                  <span>Recent Predictions History ({predictionHistory.length})</span>
                  {predictionHistory.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setPredictionHistory([])}
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "#64748b",
                        fontSize: "0.66rem",
                        cursor: "pointer",
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>

                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.2rem",
                    overflowY: "auto",
                    flex: 1,
                  }}
                >
                  {predictionHistory.length === 0 ? (
                    <div style={{ fontSize: "0.72rem", color: "#64748b", fontStyle: "italic", padding: "0.2rem 0" }}>
                      No previous predictions in this session.
                    </div>
                  ) : (
                    predictionHistory.map((item, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "0.22rem 0.5rem",
                          background: "rgba(0, 0, 0, 0.25)",
                          borderRadius: "0.3rem",
                          fontSize: "0.74rem",
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                          <span
                            style={{
                              fontWeight: 800,
                              color: "#fff",
                              background: "rgba(139, 92, 246, 0.35)",
                              padding: "0.06rem 0.4rem",
                              borderRadius: "0.25rem",
                              fontSize: "0.74rem",
                            }}
                          >
                            {item.prediction}
                          </span>
                          <span style={{ fontSize: "0.68rem", color: "#64748b" }}>
                            {item.timestamp}
                          </span>
                        </div>
                        <span
                          style={{
                            fontWeight: 700,
                            fontSize: "0.72rem",
                            color: item.confidence >= 0.85 ? "#4ade80" : "#fbbf24",
                          }}
                        >
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </>
          )}

          {/* ──── DATA COLLECTION MODE VIEW ──── */}
          {appMode === "COLLECT" && (
            <>
              {/* Target Label Configuration Card */}
              <div
                style={{
                  background: "rgba(255, 255, 255, 0.025)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: "0.875rem",
                  padding: "0.75rem 0.9rem",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.55rem",
                }}
              >
                {/* Active label row */}
                <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
                  <div style={{ flex: 1 }}>
                    <label
                      htmlFor="label-input"
                      style={{
                        display: "block",
                        fontSize: "0.7rem",
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        color: "#94a3b8",
                        marginBottom: "0.2rem",
                      }}
                    >
                      Active Target Label
                    </label>
                    <input
                      id="label-input"
                      type="text"
                      value={label}
                      onChange={(e) => updateLabel(e.target.value)}
                      disabled={isRecording}
                      maxLength={32}
                      placeholder="e.g. 0, 1, A…"
                      style={{
                        width: "100%",
                        padding: "0.45rem 0.75rem",
                        fontSize: "1rem",
                        fontWeight: 700,
                        background: "rgba(255, 255, 255, 0.06)",
                        border: "1px solid rgba(255, 255, 255, 0.12)",
                        borderRadius: "0.5rem",
                        color: "#f1f5f9",
                        outline: "none",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>

                  <div
                    style={{
                      background: "rgba(139, 92, 246, 0.1)",
                      border: "1px solid rgba(139, 92, 246, 0.25)",
                      borderRadius: "0.5rem",
                      padding: "0.4rem 0.65rem",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-end",
                    }}
                  >
                    <span style={{ fontSize: "0.68rem", color: "#94a3b8", textTransform: "uppercase" }}>
                      Target Directory
                    </span>
                    <span style={{ fontSize: "0.82rem", color: "#c4b5fd", fontWeight: 700, fontFamily: "monospace" }}>
                      datasets/{folderInfo.folder}/
                    </span>
                  </div>
                </div>

                {/* Quick Alphabet Selector */}
                <div>
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "0.2rem",
                      padding: "0.3rem",
                      background: "rgba(0,0,0,0.3)",
                      borderRadius: "0.5rem",
                      border: "1px solid rgba(255,255,255,0.05)",
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
                            width: "27px",
                            height: "27px",
                            fontSize: "0.78rem",
                            fontWeight: isSelected ? 800 : 600,
                            borderRadius: "0.3rem",
                            border: isSelected
                              ? "1px solid #8b5cf6"
                              : "1px solid rgba(255,255,255,0.06)",
                            background: isSelected
                              ? "linear-gradient(135deg, #7c3aed, #06b6d4)"
                              : "rgba(255,255,255,0.04)",
                            color: isSelected ? "#ffffff" : "#cbd5e1",
                            cursor: isRecording ? "not-allowed" : "pointer",
                            padding: 0,
                          }}
                        >
                          {char}
                        </button>
                      );
                    })}
                  </div>

                  {/* Quick Digits row */}
                  <div style={{ display: "flex", gap: "0.2rem", marginTop: "0.25rem" }}>
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
                            height: "25px",
                            fontSize: "0.78rem",
                            fontWeight: isSelected ? 800 : 600,
                            borderRadius: "0.3rem",
                            border: isSelected
                              ? "1px solid #38bdf8"
                              : "1px solid rgba(255,255,255,0.06)",
                            background: isSelected
                              ? "linear-gradient(135deg, #0284c7, #06b6d4)"
                              : "rgba(255,255,255,0.03)",
                            color: isSelected ? "#ffffff" : "#94a3b8",
                            cursor: isRecording ? "not-allowed" : "pointer",
                            padding: 0,
                          }}
                        >
                          {digit}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Next file preview banner */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    background: "rgba(0, 0, 0, 0.2)",
                    borderRadius: "0.45rem",
                    padding: "0.35rem 0.65rem",
                    fontSize: "0.76rem",
                  }}
                >
                  <span style={{ color: "#94a3b8" }}>
                    Next File: <strong style={{ color: "#34d399", fontFamily: "monospace" }}>dataset_{folderInfo.folder}_{folderInfo.nextId}.csv</strong>
                  </span>
                  <span style={{ color: "#38bdf8", fontWeight: 700 }}>
                    {folderInfo.totalForLetter} files saved
                  </span>
                </div>
              </div>

              {/* Big Record Button */}
              <button
                onMouseDown={wsReady && !isRecording ? startCapture : undefined}
                onMouseUp={wsReady && isRecording ? stopCapture : undefined}
                onMouseLeave={wsReady && isRecording ? stopCapture : undefined}
                onTouchStart={wsReady && !isRecording ? startCapture : undefined}
                onTouchEnd={wsReady && isRecording ? stopCapture : undefined}
                onContextMenu={preventContext}
                disabled={!wsReady}
                style={{
                  padding: "1rem 1.25rem",
                  fontSize: "1rem",
                  fontWeight: 700,
                  letterSpacing: "0.02em",
                  border: "none",
                  borderRadius: "0.85rem",
                  cursor: wsReady ? "pointer" : "not-allowed",
                  color: "#fff",
                  background: isRecording
                    ? "linear-gradient(135deg, #7c3aed, #a855f7)"
                    : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                  boxShadow: isRecording
                    ? "0 0 20px #8b5cf666"
                    : "0 4px 12px rgba(99,102,241,0.35)",
                  transition: "all 0.15s ease",
                  animation: isRecording ? "pulse-btn 1.2s ease-in-out infinite" : "none",
                }}
              >
                {isRecording
                  ? `⏺  Recording "${label}" — Release or Press Pen Button`
                  : `⏺  Press & Hold (or Click Pen Button) to Record "${label}"`}
              </button>

              {/* Session saved counter */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "0.4rem 0.85rem",
                  background: "rgba(255, 255, 255, 0.02)",
                  borderRadius: "0.5rem",
                  fontSize: "0.76rem",
                  color: "#94a3b8",
                }}
              >
                <span>Session Captures: <strong style={{ color: "#22c55e" }}>{savedCount} files</strong></span>
                <span>Pen Button: <strong style={{ color: "#38bdf8" }}>Active</strong></span>
              </div>
            </>
          )}
        </div>

        {/* ═══════════════════════════════════════════════════
            RIGHT COLUMN: HARDWARE, TELEMETRY & SYSTEM LOG
            ═══════════════════════════════════════════════════ */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.65rem",
            minHeight: 0,
            overflow: "hidden",
          }}
        >
          {/* BLE Hardware Card */}
          <div
            style={{
              background: "rgba(255, 255, 255, 0.025)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: "0.75rem",
              padding: "0.65rem 0.85rem",
              flexShrink: 0,
            }}
          >
            {isBleConnected ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "1.1rem" }}>🖊</span>
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#22c55e" }}>
                      {connectedDevice?.name || targetDevice}
                    </div>
                    {connectedDevice?.address && (
                      <div style={{ fontSize: "0.7rem", color: "#64748b" }}>
                        {connectedDevice.address}
                      </div>
                    )}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={disconnectBle}
                  style={{
                    padding: "0.25rem 0.6rem",
                    fontSize: "0.74rem",
                    fontWeight: 600,
                    color: "#f87171",
                    background: "rgba(239, 68, 68, 0.1)",
                    border: "1px solid rgba(239, 68, 68, 0.25)",
                    borderRadius: "0.4rem",
                    cursor: "pointer",
                  }}
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", gap: "0.4rem" }}>
                <input
                  type="text"
                  value={targetDevice}
                  onChange={(e) => setTargetDevice(e.target.value)}
                  disabled={isBleBusy}
                  placeholder="SmartPen_ESP32C3"
                  style={{
                    flex: 1,
                    padding: "0.4rem 0.6rem",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: "0.45rem",
                    color: "#f1f5f9",
                    outline: "none",
                  }}
                />
                <button
                  type="button"
                  onClick={scanBle}
                  disabled={!wsReady || isBleBusy}
                  style={{
                    padding: "0.4rem 0.65rem",
                    fontSize: "0.76rem",
                    fontWeight: 600,
                    color: "#cbd5e1",
                    background: "rgba(255, 255, 255, 0.08)",
                    border: "1px solid rgba(255, 255, 255, 0.12)",
                    borderRadius: "0.45rem",
                    cursor: wsReady && !isBleBusy ? "pointer" : "not-allowed",
                  }}
                >
                  Scan
                </button>
                <button
                  type="button"
                  onClick={() => connectBle()}
                  disabled={!wsReady || isBleBusy}
                  style={{
                    padding: "0.4rem 0.75rem",
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    border: "none",
                    borderRadius: "0.45rem",
                    cursor: wsReady && !isBleBusy ? "pointer" : "not-allowed",
                    color: "#fff",
                    background: "linear-gradient(135deg, #4f46e5, #06b6d4)",
                  }}
                >
                  Connect
                </button>
              </div>
            )}

            {/* Dropdown list if scanning discovered devices */}
            {showDevicePicker && discoveredDevices.length > 0 && !isBleConnected && (
              <div
                style={{
                  marginTop: "0.35rem",
                  padding: "0.35rem",
                  background: "rgba(0, 0, 0, 0.5)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: "0.4rem",
                  maxHeight: "90px",
                  overflowY: "auto",
                }}
              >
                {discoveredDevices.map((dev) => (
                  <div
                    key={dev.address}
                    onClick={() => {
                      setTargetDevice(dev.name || dev.address);
                      connectBle(dev.address);
                    }}
                    style={{
                      padding: "0.25rem 0.45rem",
                      borderRadius: "0.3rem",
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "0.74rem",
                    }}
                  >
                    <span>{dev.name}</span>
                    <span style={{ color: "#64748b" }}>{dev.address}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Telemetry / Hardware Info Card */}
          <div
            style={{
              background: "rgba(255, 255, 255, 0.02)",
              border: "1px solid rgba(255, 255, 255, 0.06)",
              borderRadius: "0.75rem",
              padding: "0.55rem 0.85rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.35rem",
              flexShrink: 0,
            }}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.4rem", textAlign: "center" }}>
              <div style={{ background: "rgba(0,0,0,0.2)", padding: "0.3rem", borderRadius: "0.4rem" }}>
                <div style={{ fontSize: "0.64rem", color: "#94a3b8" }}>SAMPLING</div>
                <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#38bdf8" }}>500 Hz (2ms)</div>
              </div>
              <div style={{ background: "rgba(0,0,0,0.2)", padding: "0.3rem", borderRadius: "0.4rem" }}>
                <div style={{ fontSize: "0.64rem", color: "#94a3b8" }}>WINDOW</div>
                <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#c084fc" }}>500 samples</div>
              </div>
              <div style={{ background: "rgba(0,0,0,0.2)", padding: "0.3rem", borderRadius: "0.4rem" }}>
                <div style={{ fontSize: "0.64rem", color: "#94a3b8" }}>IMU AXES</div>
                <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#34d399" }}>6-Axis Motion</div>
              </div>
            </div>

            <div style={{ fontSize: "0.7rem", color: "#94a3b8", display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span>💡</span>
              <span><strong>Hardware Trigger:</strong> Press the button on the pen anytime to capture!</span>
            </div>
          </div>

          {/* System Event Log Terminal (Fills Remaining Screen Height) */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid rgba(255, 255, 255, 0.06)",
              borderRadius: "0.75rem",
              overflow: "hidden",
            }}
          >
            {/* Terminal Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0.4rem 0.75rem",
                background: "rgba(255, 255, 255, 0.03)",
                borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
                flexShrink: 0,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#ef4444" }} />
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#f59e0b" }} />
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#22c55e" }} />
                <span
                  style={{
                    marginLeft: "0.35rem",
                    fontSize: "0.68rem",
                    color: "#64748b",
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                  }}
                >
                  Live System Events
                </span>
              </div>

              <button
                type="button"
                onClick={() => setLogs([])}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#64748b",
                  fontSize: "0.66rem",
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
            </div>

            {/* Scrollable Log Body */}
            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflowY: "auto",
                padding: "0.5rem 0.75rem",
                fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
                fontSize: "0.73rem",
                lineHeight: 1.55,
              }}
            >
              {logs.map((entry) => (
                <div key={entry.id} style={{ display: "flex", gap: "0.45rem" }}>
                  <span style={{ color: "#475569", flexShrink: 0 }}>{entry.time}</span>
                  <span
                    style={{
                      color:
                        entry.type === "success"
                          ? "#4ade80"
                          : entry.type === "error"
                          ? "#f87171"
                          : entry.type === "prediction"
                          ? "#f472b6"
                          : entry.type === "recording"
                          ? "#c4b5fd"
                          : "#94a3b8",
                      wordBreak: "break-all",
                    }}
                  >
                    {entry.text}
                  </span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </main>

      {/* ─── Keyframe Animations ────────────────────── */}
      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.5; transform: scale(1.35); }
        }
        @keyframes pulse-btn {
          0%, 100% { box-shadow: 0 0 16px rgba(244, 63, 94, 0.4); }
          50%      { box-shadow: 0 0 28px rgba(244, 63, 94, 0.7); }
        }
        @keyframes pop-in {
          0% { transform: scale(0.9); opacity: 0; }
          100% { transform: scale(1); opacity: 1; }
        }

        /* Scrollbars inside contained areas only */
        div::-webkit-scrollbar { width: 4px; height: 4px; }
        div::-webkit-scrollbar-track { background: transparent; }
        div::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.12);
          border-radius: 2px;
        }
        div::-webkit-scrollbar-thumb:hover {
          background: rgba(255,255,255,0.25);
        }

        button { -webkit-touch-callout: none; }
        input::placeholder { color: #475569; }
      `}</style>
    </div>
  );
}
