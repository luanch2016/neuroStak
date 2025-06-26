function computeOutputShape(node) {
  const type = node.data?.layerType;
  const params = node.data?.params || {};
  const inputShapeStr = node.data?.inputShape || node.data?.outputShape || "[1, 28, 28]";
  let shape;

  try {
    shape = JSON.parse(inputShapeStr.replace(/'/g, '"'));
  } catch {
    return null;
  }

  switch (type) {
    case "Conv2d": {
      const [c, h, w] = shape;
      const out_channels = parseInt(params.out_channels);
      const kernel = parseInt(params.kernel_size || 3);
      const stride = parseInt(params.stride || 1);
      const pad = parseInt(params.padding || 0);
      const outH = Math.floor((h + 2 * pad - kernel) / stride + 1);
      const outW = Math.floor((w + 2 * pad - kernel) / stride + 1);
      return [out_channels, outH, outW];
    }
    case "MaxPool2d": {
      const [c, h, w] = shape;
      const kernel = parseInt(params.kernel_size || 2);
      const stride = parseInt(params.stride || kernel);
      const outH = Math.floor((h - kernel) / stride + 1);
      const outW = Math.floor((w - kernel) / stride + 1);
      return [c, outH, outW];
    }
    case "Flatten":
      return [shape.reduce((a, b) => a * b, 1)];
    case "Linear":
      return [parseInt(params.out_features)];
    case "BatchNorm":
    case "ReLU":
    case "DropOut":
    case "LeakyReLU":
    case "PReLU":
    case "sigmoid":
    case "tanh":
      return shape;
    default:
      return shape;
  }
}

export default function validateConnections(nodes = [], edges = []) {
  const nodeMap = new Map();
  nodes.forEach((node) => nodeMap.set(node.id, { ...node }));

  // Step 1: compute output shapes for all nodes
  const outputShapes = {};
  for (const node of nodes) {
    const inputShapeStr = node.data?.inputShape || node.data?.outputShape || "[1, 28, 28]";
    node.data.inputShape = inputShapeStr;
    const outShape = computeOutputShape(node);
    node.data.outputShape = JSON.stringify(outShape);
    node.data.inputLabel = "IN: " + inputShapeStr;
    node.data.outputLabel = "OUT: " + JSON.stringify(outShape);
    outputShapes[node.id] = outShape;
  }

  // Step 2: validate edges and propagate input shapes to target nodes
  const updatedEdges = edges.map((edge) => {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);

    let isValid = true;
    let reason = "";

    if (!sourceNode || !targetNode) {
      isValid = false;
      reason = "Missing source or target node";
    } else {
      const sourceParams = sourceNode.data?.params || {};
      const targetParams = targetNode.data?.params || {};
      const sourceType = sourceNode.data?.layerType;
      const targetType = targetNode.data?.layerType;

      // Specific validations
      if (sourceType === "Input" && targetType === "Conv2d") {
        const shapeStr = sourceParams.shape || "[1, 28, 28]";
        try {
          const shape = JSON.parse(shapeStr.replace(/'/g, '"'));
          const inputChannels = shape[0];
          const convIn = parseInt(targetParams.in_channels);
          if (inputChannels !== convIn) {
            isValid = false;
            reason = `Conv2d in_channels (${convIn}) ≠ Input channels (${inputChannels})`;
          }
        } catch {
          isValid = false;
          reason = "Invalid input shape format";
        }
      }

      if (sourceType === "Conv2d" && targetType === "Conv2d") {
        const out = parseInt(sourceParams.out_channels);
        const incoming = parseInt(targetParams.in_channels);
        if (out !== incoming) {
          isValid = false;
          reason = `Conv2d in_channels (${incoming}) ≠ previous out_channels (${out})`;
        }
      }

      if (sourceType === "Conv2d" && targetType === "Linear") {
        isValid = false;
        reason = "Add a Flatten layer between Conv2d and Linear";
      }

      // Propagate shape if valid
      if (isValid) {
        const newInputShape = outputShapes[edge.source];
        if (newInputShape) {
          targetNode.data = {
            ...targetNode.data,
            inputShape: JSON.stringify(newInputShape),
          };
          targetNode.data.outputShape = JSON.stringify(computeOutputShape(targetNode));
        }
      }
    }

    return {
      ...edge,
      animated: !isValid,
      label: !isValid ? "invalid" : "",
      style: {
        stroke: isValid ? "#4b5563" : "red",
        strokeDasharray: isValid ? "" : "5,5",
        strokeWidth: 2,
      },
      data: {
        ...edge.data,
        warning: isValid ? null : reason,
      },
    };
  });

  const warnings = updatedEdges
    .filter((e) => e.data?.warning)
    .map((e) => e.data.warning);

  return { updatedEdges, warnings };
}