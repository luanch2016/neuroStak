// src/components/ConsoleLog.jsx
import React from "react";

export default function ConsoleLog({ logs }) {
  return (
    <div className="console-panel">
      <strong>🧾 Console</strong>
      {Array.isArray(logs) &&
        logs.slice(-10).map((msg, i) => (
          <div key={i}>• {typeof msg === "string" ? msg : JSON.stringify(msg)}</div>
        ))}
    </div>
  );
}