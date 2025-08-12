// Simple, extendable rules that turn a prompt into nodes/edges.
let idCounter = 1;
const getId = () => `node_${idCounter++}`;

const tokenMap = {
  // Convolutions
  conv: (spec) => {
    // conv32, conv64k3s2p1, conv128 -> parse out_channels and optional k/s/p
    const m = spec.match(/^conv(\d+)(?:k(\d+))?(?:s(\d+))?(?:p(\d+))?$/i);
    const out = m ? parseInt(m[1], 10) : 32;
    const k = m?.[2] ? parseInt(m[2], 10) : 3;
    const s = m?.[3] ? parseInt(m[3], 10) : 1;
    const p = m?.[4] ? parseInt(m[4], 10) : 0;
    return {
      layerType: "Conv2d",
      params: { in_channels: "__auto__", out_channels: out, kernel_size: k, stride: s, padding: p }
    };
  },
  relu: () => ({ layerType: "ReLU", params: {} }),
  gelu: () => ({ layerType: "GELU", params: {} }),
  elu: () => ({ layerType: "ELU", params: {} }),
  leakyrelu: () => ({ layerType: "LeakyReLU", params: { negative_slope: 0.01 } }),
  bn: () => ({ layerType: "BatchNorm", params: { num_features: "__auto__" } }),
  batchnorm: () => ({ layerType: "BatchNorm", params: { num_features: "__auto__" } }),
  maxpool: (spec) => {
    // maxpool, maxpool2, maxpool3s2
    const m = spec.match(/^maxpool(?:(\d+))?(?:s(\d+))?$/i);
    const k = m?.[1] ? parseInt(m[1], 10) : 2;
    const s = m?.[2] ? parseInt(m[2], 10) : k;
    return { layerType: "MaxPool2d", params: { kernel_size: k, stride: s, padding: 0 } };
  },
  avgpool: () => ({ layerType: "AvgPool2d", params: { kernel_size: 2, stride: 2, padding: 0 } }),
  dropout: (spec) => {
    const m = spec.match(/^dropout(?:([\d.]+))?$/i);
    const p = m?.[1] ? parseFloat(m[1]) : 0.5;
    return { layerType: "Dropout", params: { p } };
  },
  flatten: () => ({ layerType: "Flatten", params: {} }),
  dense: (spec) => {
    const m = spec.match(/^dense(\d+)$/i);
    const out = m ? parseInt(m[1], 10) : 128;
    return { layerType: "Linear", params: { in_features: "__auto__", out_features: out } };
  },
  linear: (spec) => {
    const m = spec.match(/^linear(\d+)$/i);
    const out = m ? parseInt(m[1], 10) : 128;
    return { layerType: "Linear", params: { in_features: "__auto__", out_features: out } };
  },
  // RNNs
  lstm: (spec) => {
    const m = spec.match(/^lstm(?:(\d+))?$/i);
    const units = m?.[1] ? parseInt(m[1], 10) : 128;
    return { layerType: "LSTM", params: { input_size: "__auto__", hidden_size: units, batch_first: true } };
  },
  bilstm: (spec) => {
    const m = spec.match(/^bilstm(?:(\d+))?$/i);
    const units = m?.[1] ? parseInt(m[1], 10) : 128;
    return { layerType: "BiLSTM", params: { input_size: "__auto__", hidden_size: units, batch_first: true } };
  },
  gru: (spec) => {
    const m = spec.match(/^gru(?:(\d+))?$/i);
    const units = m?.[1] ? parseInt(m[1], 10) : 128;
    return { layerType: "GRU", params: { input_size: "__auto__", hidden_size: units, batch_first: true } };
  },
  // blocks
  res: () => ({ layerType: "ResidualBlock", params: { in_channels: "__auto__", out_channels: "__auto__", num_blocks: 2 } }),
  skip: () => ({ layerType: "Skip", params: {} }),
};

function parseTokens(text) {
  // normalize: split by -> , > , | or whitespace lines
  const raw = text
    .toLowerCase()
    .replace(/\s*->\s*|\s*>\s*|\s*\|\s*/g, " ")
    .split(/[\s,]+/)
    .filter(Boolean);

  const layers = [];
  for (const tok of raw) {
    // find first rule that matches prefix
    const key = Object.keys(tokenMap).find((k) => tok.startsWith(k));
    if (!key) continue;
    layers.push(tokenMap[key](tok));
  }
  return layers;
}

export function promptToGraph(prompt, opts = {}) {
  idCounter = 1;
  const nodes = [];
  const edges = [];
  const spacingY = opts.spacingY || 90;
  const startX = opts.startX || 100;
  const startY = opts.startY || 50;

  // add input
  nodes.push({
    id: "input",
    type: "input",
    position: { x: startX, y: startY },
    data: { layerType: "Input Layer", label: "Input Layer", params: opts.input || {} },
  });

  let prevId = "input";
  let y = startY + spacingY;

  const parts = parseTokens(prompt);
  parts.forEach((layerSpec, i) => {
    const id = getId();
    nodes.push({
      id,
      type: "default",
      position: { x: startX, y },
      data: {
        layerType: layerSpec.layerType,
        label: `${layerSpec.layerType} Layer`,
        params: layerSpec.params || {},
      },
    });
    edges.push({
      id: `e-${prevId}-${id}`,
      source: prevId,
      target: id,
      label: "",
      style: { stroke: "#4b5563", strokeWidth: 2 },
    });
    prevId = id;
    y += spacingY;
  });

  return { nodes, edges };
}