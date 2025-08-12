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

export function generateTensorFlowCode(nodes, edges, exportPy, logMessage, modelName) {
  // --- Helpers ---
  const tfLayerMap = {
    // Convolutions
    Conv2d: "layers.Conv2D",
    ConvTranspose2d: "layers.Conv2DTranspose",
    DepthwiseConv2d: "layers.DepthwiseConv2D",
    SeparableConv2d: "layers.SeparableConv2D",

    // Pooling / resize
    MaxPool2d: "layers.MaxPooling2D",
    AvgPool2d: "layers.AveragePooling2D",
    GlobalAvgPool2d: "layers.GlobalAveragePooling2D",
    AdaptiveAvgPool2d: "layers.GlobalAveragePooling2D", // approximation
    Upsample: "layers.UpSampling2D",

    // Normalization
    BatchNorm: "layers.BatchNormalization",
    LayerNorm: "layers.LayerNormalization",
    GroupNorm: "tfa.layers.GroupNormalization", // requires tensorflow-addons
    InstanceNorm2d: "tfa.layers.InstanceNormalization", // requires tensorflow-addons

    // Activations
    ReLU: "layers.ReLU",
    LeakyReLU: "layers.LeakyReLU",
    PReLU: "layers.PReLU",
    ELU: "layers.ELU",
    GELU: "layers.Activation", // use Activation('gelu')
    Tanh: "layers.Activation",
    Sigmoid: "layers.Activation",
    Softmax: "layers.Activation",

    // Regularization
    Dropout: "layers.Dropout",
    Dropout2d: "layers.SpatialDropout2D",

    // Linear / utility
    Flatten: "layers.Flatten",
    Linear: "layers.Dense",
    Concat: "# Concat marker",
    Reshape: "layers.Reshape",
    Skip: "# Skip marker",
    ResidualBlock: "# Residual block marker",
    SqueezeExcitation: "# SE block marker",

    // Sequence / RNN
    LSTM: "layers.LSTM",
    GRU: "layers.GRU",
    RNN: "layers.SimpleRNN",
    Embedding: "layers.Embedding",
    BiLSTM: "layers.Bidirectional", // wrapper
    StackedLSTM: "layers.LSTM",
    LSTMCell: "layers.RNN", // with layers.LSTMCell
    Attention: "# Attention placeholder",
    PackPaddedSequence: "# PackPaddedSequence utility",
    PadPackedSequence: "# PadPackedSequence utility",
    TimeDistributed: "# TimeDistributed wrapper",
  };

  const tfParams = (type, params = {}) => {
    const p = { ...params };

    // 2D conv / transpose conv / depthwise / separable
    if (["Conv2d", "ConvTranspose2d", "DepthwiseConv2d", "SeparableConv2d"].includes(type)) {
      if (p.out_channels != null) { p.filters = p.out_channels; delete p.out_channels; }
      if (p.kernel_size != null) { p.kernel_size = p.kernel_size; }
      if (p.stride != null) { p.strides = p.stride; delete p.stride; }
      if (p.padding != null) { p.padding = Number(p.padding) > 0 ? 'same' : 'valid'; }
      if (p.in_channels != null) { delete p.in_channels; }
    }

    if (["MaxPool2d", "AvgPool2d"].includes(type)) {
      if (p.kernel_size != null) { p.pool_size = p.kernel_size; delete p.kernel_size; }
      if (p.stride != null) { p.strides = p.stride; delete p.stride; }
      if (p.padding != null) { p.padding = Number(p.padding) > 0 ? 'same' : 'valid'; }
    }

    if (type === "Upsample") {
      if (p.scale_factor != null) { p.size = p.scale_factor; delete p.scale_factor; }
    }

    if (type === "Linear") {
      if (p.out_features != null) { p.units = p.out_features; delete p.out_features; }
      if (p.in_features != null) { delete p.in_features; }
    }

    if (["Softmax", "Sigmoid", "Tanh"].includes(type)) {
      p.activation = type.toLowerCase();
    }

    return p;
  };

  // Parse an input shape string like "[3, 224, 224]" to Keras channels_last (224, 224, 3)
  const parseInputShape = (shapeStr) => {
    try {
      const arr = JSON.parse(shapeStr.replace(/(\w+)/g, (m) => (/^\d+$/.test(m) ? m : `"${m}"`)));
      if (Array.isArray(arr) && arr.length === 3) {
        const [c, h, w] = arr.map(Number);
        if (!Number.isNaN(c) && !Number.isNaN(h) && !Number.isNaN(w)) {
          return `${h}, ${w}, ${c}`; // channels_last
        }
      }
    } catch (_) {}
    return null;
  };

  const sorted = [...nodes].sort((a, b) => a.position.y - b.position.y);

  // Build TensorFlow Keras Functional model code
  const lines = [
    "import tensorflow as tf",
    "from tensorflow import keras",
    "from tensorflow.keras import layers, models",
    "try:\n    import tensorflow_addons as tfa\nexcept Exception:\n    tfa = None  # Optional: used for Group/Instance Norm",
    "",
    "def build_model():",
  ];

  // Find Input layer if present
  const inputNode = sorted.find(n => (n.data?.layerType || '').toLowerCase().includes('input'));
  let inputShapeExpr = null;
  if (inputNode) {
    const s = inputNode.data?.params?.shape;
    if (typeof s === 'string') {
      inputShapeExpr = parseInputShape(s);
    }
  }
  if (!inputShapeExpr) {
    // Fallback to a safe default
    inputShapeExpr = "224, 224, 3";
  }

  lines.push(`    inputs = keras.Input(shape=(${inputShapeExpr}))`);
  lines.push("    x = inputs");

  // Build the body
  sorted.forEach((node) => {
    const type = node.data?.layerType;
    if (!type) return;
    if ((type || '').toLowerCase().includes('input')) return; // already handled

    const def = tfLayerMap[type];
    if (!def) return;

    // Special markers
    if (def.startsWith('#')) {
      if (type === 'ResidualBlock') {
        lines.push("    # Residual Block (basic): Conv-BN-ReLU -> Conv-BN -> Add -> ReLU");
        lines.push("    identity = x");
        lines.push("    x = layers.Conv2D(filters=identity.shape[-1] if identity.shape.rank is not None else 64, kernel_size=3, padding='same')(x)");
        lines.push("    x = layers.BatchNormalization()(x)");
        lines.push("    x = layers.ReLU()(x)");
        lines.push("    x = layers.Conv2D(filters=identity.shape[-1] if identity.shape.rank is not None else 64, kernel_size=3, padding='same')(x)");
        lines.push("    x = layers.BatchNormalization()(x)");
        lines.push("    x = layers.Add()([x, identity])");
        lines.push("    x = layers.ReLU()(x)");
      } else if (type === 'Concat') {
        lines.push("    # TODO: Concat requires multi-input graph wiring (Functional API)");
      } else if (type === 'Skip') {
        lines.push("    # TODO: Skip connection requires graph reference to earlier tensor");
      } else if (type === 'SqueezeExcitation') {
        lines.push("    # TODO: Squeeze-and-Excitation block placeholder");
      } else if (type === 'Attention') {
        lines.push("    # TODO: Attention block placeholder");
      }
      return;
    }

    // Regular layers
    const params = tfParams(type, node.data?.params || {});

    // Special cases where the layer expression differs
    if (type === 'BiLSTM') {
      const innerArgs = Object.entries(params).map(([k,v]) => `${k}=${v}`).join(', ');
      lines.push(`    x = layers.Bidirectional(layers.LSTM(${innerArgs}))(x)`);
      return;
    }
    if (type === 'GELU') {
      lines.push("    x = layers.Activation('gelu')(x)");
      return;
    }
    if (type === 'LSTMCell') {
      const units = params.units || params.out_features || 128;
      lines.push(`    x = layers.RNN(layers.LSTMCell(${units}))(x)`);
      return;
    }
    if (type === 'TimeDistributed') {
      lines.push("    # TODO: TimeDistributed wrapper requires inner layer");
      return;
    }

    const args = Object.entries(params).map(([k,v]) => `${k}=${v}`).join(', ');
    lines.push(`    x = ${def}(${args})(x)`);
  });

  lines.push("    model = keras.Model(inputs=inputs, outputs=x, name='" + (modelName || 'GeneratedModelTF') + "')");
  lines.push("    return model");
  lines.push("\nif __name__ == '__main__':");
  lines.push("    model = build_model()");
  lines.push("    model.summary()");

  const code = lines.join('\n');

  if (exportPy) {
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${modelName || 'model_tf'}.py`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    if (typeof logMessage === 'function') logMessage('⬇️ Exported TensorFlow (Keras Functional) model.py');
  } else {
    console.log('🧠 Generated TensorFlow (Keras Functional) Code:\n', code);
    if (typeof logMessage === 'function') logMessage('✅ TensorFlow (Keras) code printed to console');
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