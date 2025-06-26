import { addEdge } from "react-flow-renderer";
import validateConnections from "./validateConnections";

let id = 1;
export const getId = () => `node_${id++}`;

export const logMessage = (msg, setLogs) => {
  setLogs((prev) => [...(Array.isArray(prev) ? prev : []), msg]);
};

export const validateAndUpdateEdges = (nodesList, edgesList, setEdges, setLogs) => {
  const { updatedEdges, warnings } = validateConnections(nodesList || [], edgesList || []);
  setEdges(updatedEdges);
  if (setLogs) warnings.forEach((w) => logMessage(w, setLogs));
  return { updatedEdges, warnings };
};

export const onConnectHandler = (params, nodes, edges, setEdges, validateEdges) => {
  const rawEdges = addEdge(params, edges);
  validateEdges(nodes, rawEdges);
};

export const onDropHandler = (
  event,
  reactFlowWrapper,
  reactFlowInstance,
  nodes,
  edges,
  setNodes,
  validateEdges,
  log,
  getId
) => {
  event.preventDefault();
  const type = event.dataTransfer.getData("application/reactflow");
  if (!type) return;

  const bounds = reactFlowWrapper.current.getBoundingClientRect();
  const position = reactFlowInstance?.project
    ? reactFlowInstance.project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      })
    : {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };

  const newNode = {
    id: getId(),
    type: "default",
    position,
    data: {
      label: `${type} Layer`,
      layerType: type,
      params: {},
    },
  };

  const newNodes = [...nodes, newNode];
  setNodes(newNodes);
  validateEdges(newNodes, edges);
  log(`➕ Added ${type} layer`);
};

export const updateNodeHandler = (
  updated,
  nodes,
  setNodes,
  edges,
  setEdges,
  setSelectedNode,
  log
) => {
  const newNodes = nodes.map((n) => (n.id === updated.id ? updated : n));
  setNodes(newNodes);

  const { updatedEdges, warnings } = validateConnections(newNodes, edges || []);
  setEdges(updatedEdges);

  if (log) {
    log(`✅ Updated parameters for ${updated.data?.layerType}`);
    warnings.forEach(log);
  }

  setSelectedNode(updated);
};

export const onDragOverHandler = (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
};