import React, { useState } from "react";

export default function PromptPanel({ onGenerateFromPrompt, disabled }) {
  const [prompt, setPrompt] = useState(
    "simple cnn: conv32 -> relu -> maxpool -> conv64 -> relu -> maxpool -> flatten -> dense128 -> relu -> dense10"
  );

  const submit = () => {
    if (!prompt.trim()) return;
    onGenerateFromPrompt(prompt.trim());
  };

  return (
    <div className="prompt-panel">
      <label className="prompt-label">Build from prompt</label>
      <textarea
        className="prompt-input"
        rows={3}
        placeholder="e.g. 2 conv blocks then maxpool, then flatten and dense 128, relu, dense 10"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={disabled}
      />
      <div className="prompt-actions">
        <button className="btn blue" onClick={submit} disabled={disabled}>
          Generate from Prompt
        </button>
      </div>
    </div>
  );
}