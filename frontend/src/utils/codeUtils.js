export function generatePyTorchCode(nodes, edges, exportPy, logMessage, modelName) {
    const layerMap = {
      Conv2d: "nn.Conv2d",
      ConvTranspose2d: "nn.ConvTranspose2d",
      DepthwiseConv2d: "nn.Conv2d", // PyTorch doesn't have a dedicated class, use groups=in_channels
      SeparableConv2d: "nn.Conv2d", // Approximation using Depthwise + Pointwise
      AdaptiveAvgPool2d: "nn.AdaptiveAvgPool2d",
      MaxPool2d: "nn.MaxPool2d",
      AvgPool2d: "nn.AvgPool2d",
      GlobalAvgPool2d: "nn.AdaptiveAvgPool2d", // Approximate with kernel_size=input size
      Upsample: "nn.Upsample",
      BatchNorm: "nn.BatchNorm2d",
      GroupNorm: "nn.GroupNorm",
      InstanceNorm2d: "nn.InstanceNorm2d",
      LayerNorm: "nn.LayerNorm",
      ReLU: "nn.ReLU",
      LeakyReLU: "nn.LeakyReLU",
      PReLU: "nn.PReLU",
      ELU: "nn.ELU",
      GELU: "nn.GELU",
      Tanh: "nn.Tanh",
      Sigmoid: "nn.Sigmoid",
      Softmax: "nn.Softmax",
      Dropout: "nn.Dropout",
      Dropout2d: "nn.Dropout2d",
      Flatten: "nn.Flatten",
      Skip: "# Skip marker",
      Linear: "nn.Linear",
      Concat: "# Concat marker",
      Reshape: "# Reshape marker",
      ResidualBlock: "# Residual block marker",
      SqueezeExcitation: "# SqueezeExcitation marker",
      // Add recurrent layers
      LSTM: "nn.LSTM",
      GRU: "nn.GRU",
      RNN: "nn.RNN",
      Embedding: "nn.Embedding",
      BiLSTM: "nn.LSTM",  // requires bidirectional=True
      StackedLSTM: "nn.LSTM", // requires num_layers=2+
      LSTMCell: "nn.LSTMCell",
      Attention: "# Attention placeholder",
      PackPaddedSequence: "# PackPaddedSequence utility",
      PadPackedSequence: "# PadPackedSequence utility",
      TimeDistributed: "# TimeDistributed wrapper",
    };
  
    const sorted = [...nodes].sort((a, b) => a.position.y - b.position.y);
    const lines = [
      "import torch",
      "import torch.nn as nn",
      "",
      `class ${modelName || "GeneratedModel"}(nn.Module):`,
      "    def __init__(self):",
      "        super().__init__()",
      "        self.model = nn.Sequential(",
    ];
  
    sorted.forEach((node) => {
      const def = layerMap[node.data?.layerType];
      if (!def) return;
      if (node.data?.layerType === "ResidualBlock") {
        lines.push("            nn.Sequential(");
        lines.push("                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),");
        lines.push("                nn.BatchNorm2d(out_channels),");
        lines.push("                nn.ReLU(),");
        lines.push("            ),");
        return;
      }
      if (def.startsWith("#")) return;

      const args = Object.entries(node.data?.params || {})
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");

      if (node.data?.layerType === "BiLSTM") {
        lines.push(`            ${def}(${args}, bidirectional=True),`);
      } else if (node.data?.layerType === "StackedLSTM") {
        lines.push(`            ${def}(${args}, num_layers=2),`);
      } else {
        lines.push(`            ${def}(${args}),`);
      }
    });

    lines.push("        )", "", "    def forward(self, x):", "        out = x");

    sorted.forEach((node, idx) => {
      const def = layerMap[node.data?.layerType];
      if (!def) return;

      const layerName = `self.model[${idx}]`;

      if (def.startsWith("#")) {
        if (node.data?.layerType === "ResidualBlock") {
          lines.push("        identity = out");
          lines.push(`        out = self.model[${idx}](out)`);
          lines.push("        out += identity");
          lines.push("        out = nn.ReLU()(out)");
        } else {
          lines.push(`        # ${node.data?.layerType} logic needs to be implemented`);
        }
      } else if (node.data?.layerType === "BiLSTM") {
        lines.push(`        out, _ = ${layerName}(out)  # BiLSTM`);
      } else if (node.data?.layerType === "StackedLSTM") {
        lines.push(`        out, _ = ${layerName}(out)  # Stacked LSTM`);
      } else if (["LSTM", "GRU", "RNN"].includes(node.data?.layerType)) {
        lines.push(`        out, _ = ${layerName}(out)`);
      } else if (node.data?.layerType === "LSTMCell") {
        lines.push(`        h_t, c_t = ${layerName}(out)  # LSTMCell usage example`);
      } else {
        lines.push(`        out = ${layerName}(out)`);
      }
    });

    lines.push("        return out");
  
    const code = lines.join("\n");
  
    if (exportPy) {
      const blob = new Blob([code], { type: "text/x-python" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${modelName || "model"}.py`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      logMessage("⬇️ Exported model.py");
    } else {
      console.log("🧠 Generated Code:\n", code);
      logMessage("✅ PyTorch code printed to console");
    }
  }
  
export function saveModel(nodes, edges, logMessage, modelName) {
  try {
    if (!Array.isArray(nodes) || !Array.isArray(edges)) {
      if (typeof logMessage === "function") {
        logMessage("❌ Failed to save model: Invalid nodes or edges");
      }
      return;
    }

    const cleanNodes = nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: { x: node.position.x, y: node.position.y },
      data: {
        layerType: node.data?.layerType,
        label: node.data?.label,
        params: JSON.parse(JSON.stringify(node.data?.params || {})),
      },
    }));

    const cleanEdges = edges.map(({ id, source, target, label, style }) => ({
      id,
      source,
      target,
      label,
      style,
    }));

    const data = { nodes: cleanNodes, edges: cleanEdges };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${modelName || "neural_network_model"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    if (typeof logMessage === "function") {
      logMessage("💾 Model saved");
    }
  } catch (error) {
    if (typeof logMessage === "function") {
      logMessage(`❌ Failed to save model: ${error.message}`);
    }
    console.error("Save model error:", error);
  }
}
  
  export function loadModel(event, setNodes, setEdges, logMessage) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const { nodes, edges } = JSON.parse(e.target.result);
        setNodes(nodes);
        setEdges(edges);
        logMessage("📂 Loaded model from file");
      } catch (err) {
        logMessage("❌ Failed to load model: invalid file");
      }
    };
    reader.readAsText(file);
  }