import { MarkerType } from "react-flow-renderer";

export const styleEdges = (edges) => {
  return edges.map((edge) => ({
    ...edge,
    animated: edge.label === "skip",
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: edge.label === "skip" ? "#f59e0b" : (edge.style?.stroke === "#f00" ? "#f00" : "#4b5563"),
    },
    style: edge.label === "skip"
      ? { stroke: "#f59e0b", strokeDasharray: "5,5", strokeWidth: 2 }
      : edge.style || { stroke: "#4b5563", strokeWidth: 2 },
  }));
};