import React, { useState } from "react";
import { Handle, Position } from "react-flow-renderer";

const CustomNode = ({ id, data, selected }) => {
  const [hovered, setHovered] = useState(false);

  const addNode = (direction) => {
    if (data?.onAdd) {
      data.onAdd(id, direction);
    }
  };

  const directions = [
    { name: "top", style: { top: -10, left: "50%", transform: "translateX(-50%)" } },
    { name: "bottom", style: { bottom: -10, left: "50%", transform: "translateX(-50%)" } },
    { name: "left", style: { left: -10, top: "50%", transform: "translateY(-50%)" } },
    { name: "right", style: { right: -10, top: "50%", transform: "translateY(-50%)" } },
  ];

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={(e) => {
        e.stopPropagation();
      }}
      style={{
        padding: 10,
        border: selected ? "2px solid #3b82f6" : "1px solid #ccc",
        borderRadius: 8,
        background: "white",
        position: "relative",
        minWidth: 120,
        textAlign: "center",
        cursor: "move",
      }}
    >
      <div>{data.label || data.layerType || "Layer"}</div>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />

      {hovered &&
        directions.map((dir) => (
          <button
            key={dir.name}
            onClick={(e) => {
              e.stopPropagation();
              addNode(dir.name);
            }}
            style={{
              position: "absolute",
              ...dir.style,
              backgroundColor: "#f59e0b",
              borderRadius: "50%",
              border: "none",
              color: "#fff",
              width: 20,
              height: 20,
              fontSize: 12,
              cursor: "pointer",
              zIndex: 10,
            }}
            title={`Add ${dir.name}`}
          >
            +
          </button>
        ))}
    </div>
  );
};

export default CustomNode;
