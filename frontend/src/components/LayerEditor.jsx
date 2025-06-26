import React, { useEffect, useState } from "react";

const defaultParamsByLayer = {
  AdaptiveAvgPool2d: { output_size: 1 },
  Attention: { query_dim: 64, key_dim: 64, value_dim: 64 },
  AvgPool2d: { kernel_size: 2 },
  BatchNorm: { num_features: 32 },
  BiLSTM: { input_size: 128, hidden_size: 64, num_layers: 1, bidirectional: true },
  Concat: {},
  Conv2d: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 0 },
  ConvTranspose2d: { in_channels: 32, out_channels: 16, kernel_size: 2, stride: 2 },
  DepthwiseConv2d: { in_channels: 32, kernel_size: 3 },
  DropOut: { p: 0.5 },
  Dropout2d: { p: 0.5 },
  ELU: {},
  Flatten: {},
  GELU: {},
  GlobalAvgPool2d: {},
  GroupNorm: { num_groups: 4, num_channels: 32 },
  Input: { shape: "[1, 28, 28]" },
  InstanceNorm2d: { num_features: 32 },
  LayerNorm: { normalized_shape: "[32]" },
  LeakyReLU: { negative_slope: 0.01 },
  Linear: { in_features: 128, out_features: 64 },
  LSTMCell: { input_size: 128, hidden_size: 64 },
  MaxPool2d: { kernel_size: 2 },
  PackPaddedSequence: { batch_first: true },
  PadPackedSequence: { batch_first: true },
  PReLU: {},
  ReLU: {},
  ResidualBlock: {},
  Reshape: { shape: "[64]" },
  SeparableConv2d: { in_channels: 32, out_channels: 64, kernel_size: 3 },
  Sigmoid: {},
  Skip: {},
  Softmax: { dim: 1 },
  SqueezeExcitation: { channels: 64, reduction: 16 },
  StackedLSTM: { input_size: 128, hidden_size: 64, num_layers: 2 },
  Tanh: {},
  TimeDistributed: { module: "Linear" },
  Upsample: { scale_factor: 2 },
};

export default function LayerEditor({ selectedNode, updateNode }) {
  const [params, setParams] = useState({});

  useEffect(() => {
    if (selectedNode) {
      const layerType = selectedNode.data?.layerType || "Linear";
      const currentParams = selectedNode.data?.params || {};
      const defaultParams = defaultParamsByLayer[layerType] || {};
      const merged = { ...defaultParams, ...currentParams };
      setParams(merged);
    }
  }, [selectedNode]);

  useEffect(() => {
    console.log("Editing Node:", selectedNode?.id);
  }, [selectedNode]);

  const handleChange = (key, value) => {
    const updatedParams = { ...params, [key]: value };
    setParams(updatedParams);
    const updatedNode = {
      ...selectedNode,
      data: {
        ...selectedNode.data,
        params: updatedParams,
      },
    };
    updateNode(updatedNode);
  };

  if (!selectedNode) return null;

  const layerType = selectedNode.data?.layerType || "Layer";

  return (
    <div style={{ padding: "1rem", border: "1px solid #ccc", backgroundColor: "#fff", borderRadius: "8px" }}>
      <h3>⚙️ {layerType} Parameters</h3>
      {Object.entries(params).map(([key, val]) => (
        <div key={key} style={{ marginBottom: "0.5rem" }}>
          <label style={{ display: "block", fontSize: "0.9rem", marginBottom: "0.2rem" }}>{key}</label>
          <input
            type="text"
            value={val}
            onChange={(e) => handleChange(key, e.target.value)}
            style={{
              width: "100%",
              padding: "0.4rem",
              borderRadius: "4px",
              border: "1px solid #ccc",
              fontSize: "0.9rem",
            }}
          />
        </div>
      ))}
    </div>
  );
}