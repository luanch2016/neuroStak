import React, { useCallback, useRef, useState, useEffect } from "react";
import { toPng, toSvg } from "html-to-image";
import jsPDF from "jspdf";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer";

import LayerNavbar from "./components/LayerNavbar";
import LayerEditor from "./components/LayerEditor";
import ButtonPanel from "./components/ButtonPanel";
import ConsoleLog from "./components/ConsoleLog";
import ModelLibraryPanel from "./components/ModelLibraryPanel";

import {
  onConnectHandler,
  onDropHandler,
  updateNodeHandler,
  validateAndUpdateEdges,
  logMessage,
  onDragOverHandler,
} from "./utils/nodeUtils";
import { generatePyTorchCode, generateTensorFlowCode, saveModel, loadModel } from "./utils/codeUtils";
import initialNodes from "./flowConfig";
import "@fontsource/work-sans/400.css";


import "./App.css";

let id = 1;
const getId = () => `node_${id++}`;

export default function App() {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [logs, setLogs] = useState(["🧠 Neural Network Builder Ready"]);
  const [modelName, setModelName] = useState("GeneratedModel");
  const [history, setHistory] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [showExportOptions, setShowExportOptions] = useState(false);

  const log = (msg) => logMessage(msg, setLogs);

  const validateEdges = (nodesList, edgesList) => {
    const { updatedEdges, warnings } = validateAndUpdateEdges(nodesList || [], edgesList || [], setEdges, setLogs);
    setEdges(updatedEdges);
    warnings.forEach(log);
  };

  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: {
          ...n.data,
          onClick: () => setSelectedNode(n),
        },
      }))
    );
  }, []);

  const onConnect = useCallback(
    (params) => onConnectHandler(params, nodes, edges, setEdges, validateEdges),
    [nodes, edges]
  );

  const onDrop = useCallback(
    (event) => {
      setHistory((h) => [...h, { nodes, edges }]);
      onDropHandler(event, reactFlowWrapper, reactFlowInstance, nodes, edges, setNodes, validateEdges, log, getId);
    },
    [nodes, edges, reactFlowInstance]
  );

  const updateNode = (updated) => {
    setHistory((h) => [...h, { nodes, edges }]);
    updateNodeHandler(updated, nodes, setNodes, edges, setEdges, setSelectedNode, log);
  };

  const handleModelLoad = (model) => {
    setHistory((h) => [...h, { nodes, edges }]);
    const enrichedNodes = (model.nodes || []).map((n) => ({
      ...n,
      data: {
        ...n.data,
        onClick: () => setSelectedNode(n),
      },
    }));
    setNodes(enrichedNodes);
    setEdges(model.edges || []);
    setSelectedNode(null);
    setLogs((prev) => [
      ...prev,
      `📦 Loaded model from template: ${model.name || "Unnamed"}`,
    ]);
  };

  const handleUndo = () => {
    if (history.length > 0) {
      const last = history[history.length - 1];
      setRedoStack((r) => [...r, { nodes, edges }]);
      setNodes(last.nodes);
      setEdges(last.edges);
      setHistory((h) => h.slice(0, -1));
      setLogs((prev) => [...prev, "↩️ Undo action"]);
    }
  };

  const handleRedo = () => {
    if (redoStack.length > 0) {
      const next = redoStack[redoStack.length - 1];
      setHistory((h) => [...h, { nodes, edges }]);
      setNodes(next.nodes);
      setEdges(next.edges);
      setRedoStack((r) => r.slice(0, -1));
      setLogs((prev) => [...prev, "↪️ Redo action"]);
    }
  };

  const handleExportSVG = () => {
    if (!reactFlowWrapper.current || !reactFlowInstance) return;

    reactFlowInstance.fitView({ padding: 0.1 });

    setTimeout(() => {
      toSvg(reactFlowWrapper.current.querySelector('.react-flow__viewport'))
        .then((dataUrl) => {
          const link = document.createElement("a");
          link.download = `${modelName}.svg`;
          link.href = dataUrl;
          link.click();
          log(`🖼️ Exported SVG: ${modelName}.svg`);
        })
        .catch((error) => {
          console.error("SVG Export Error:", error);
          log("❌ SVG export failed.");
        });
    }, 500);
  };

  const handleExportPDF = () => {
    if (!reactFlowWrapper.current || !reactFlowInstance) return;

    reactFlowInstance.fitView({ padding: 0.1 });

    setTimeout(() => {
      toPng(reactFlowWrapper.current.querySelector('.react-flow__viewport'))
        .then((dataUrl) => {
          const pdf = new jsPDF("landscape", "pt", "a4");
          const imgProps = pdf.getImageProperties(dataUrl);
          const pdfWidth = pdf.internal.pageSize.getWidth();
          const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
          pdf.addImage(dataUrl, "PNG", 0, 0, pdfWidth, pdfHeight);
          pdf.save(`${modelName}.pdf`);
          log(`📄 Exported PDF: ${modelName}.pdf`);
        })
        .catch((error) => {
          console.error("PDF Export Error:", error);
          log("❌ PDF export failed.");
        });
    }, 500);
  };

  return (
    <div className="app-container">
      <div className="navbar">
      <img src="/src/assets/neuroStak.png" alt="Logo" className="navbar-logo" />
        <LayerNavbar/>
      </div>
      <div className="main-content">
        <div className="flow-wrapper" ref={reactFlowWrapper}>
          <div className="model-controls">
            <div className="model-header">
              <label htmlFor="model-name">📝 Model Name:</label>
              <input
                id="model-name"
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="model-name-input"
              />
            </div>
            <div className="button-bar">
              <div className="export-dropdown">
                <button
                  className="btn gray"
                  onClick={() => setShowExportOptions((prev) => !prev)}
                >
                  Tools ▼
                </button>

                {showExportOptions && (
                  <div className="export-options">
                    <p>Export</p>
                    <button
                      className="btn gray"
                      onClick={() =>
                        generatePyTorchCode(nodes, edges, true, log, modelName)
                      }
                    >
                      Pytorch
                    </button>
                    <button
                      className="btn gray"
                      onClick={() =>
                        generateTensorFlowCode(nodes, edges, true, log, modelName)
                      }
                    >
                      TensorFlow
                    </button>
                    <button className="btn gray" onClick={handleExportSVG}>
                      .SVG
                    </button>
                    <button className="btn gray" onClick={handleExportPDF}>
                      .PDF
                    </button>
                    <button
                      className="btn gray"
                      onClick={() => {
                        const strippedNodes = nodes.map(({ data, ...rest }) => ({
                          ...rest,
                          data: {
                            ...data,
                            label:
                              (typeof data.label === "string" && data.label) ||
                              (data?.label?.props?.children?.[1]?.props?.children ??
                                data?.layerType ??
                                "Layer"),
                          },
                        }));
                        saveModel(strippedNodes, edges, log, modelName);
                      }}
                    >
                      .JSON
                    </button>
                    <p>Other</p>
                    <button className="btn gray" onClick={() => document.getElementById('file-input').click()}>
                      Load JSON
                    </button>
                    <input
                      id="file-input"
                      type="file"
                      accept=".json"
                      style={{ display: "none" }}
                      onChange={(e) => loadModel(e, setNodes, setEdges, log, setModelName)}
                    />
                    <button className="btn gray" onClick={() => {
                      setNodes([]);
                      setEdges([]);
                      setSelectedNode(null);
                      log("🧼 Canvas cleared.");
                    }}>
                      Clear
                    </button>
                    <button className="btn gray" onClick={handleUndo}>Undo</button>
                    <button className="btn gray" onClick={handleRedo}>Redo</button>
                  </div>
                )}
              </div>
            </div>
          </div>
          <ReactFlow
            nodes={nodes.map((node) => {
              const { params = {}, label: originalLabel, layerType } = node.data || {};
              const inText = params.in_channels ? `in: ${params.in_channels}` : "";
              const outText = params.out_channels ? `out: ${params.out_channels}` : "";
              const labelLines = [];
              
              if (inText) labelLines.push(inText);
              labelLines.push(originalLabel || `${layerType} Layer`);
              if (outText) labelLines.push(outText);

              return {
                ...node,
                data: {
                  ...node.data,
                  label: (
                    <div style={{ textAlign: "center" }}>
                      {inText && <div style={{ fontSize: "0.75rem", color: "#666" }}>{inText}</div>}
                      <div style={{ fontWeight: "bold" }}>{originalLabel || `${layerType} Layer`}</div>
                      {outText && <div style={{ fontSize: "0.75rem", color: "#666" }}>{outText}</div>}
                    </div>
                  ),
                },
              };
            })}
            edges={edges.map((edge) => ({
              ...edge,
              animated: edge.label === "skip",
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: edge.label === "skip" ? "#f59e0b" : edge.style?.stroke === "#f00" ? "#f00" : "#4b5563",
              },
              style: edge.label === "skip"
                ? { stroke: "#f59e0b", strokeDasharray: "5,5", strokeWidth: 2 }
                : edge.style || { stroke: "#4b5563", strokeWidth: 2 },
            }))}
            onInit={setReactFlowInstance}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOverHandler}
            onNodeClick={(e, n) => setSelectedNode(n)}
            fitView
          >
            <MiniMap />
            <Controls />
            <Background />
          </ReactFlow>
        </div>

        <div className="side-panel">
          <div className="editor-panel">
            {selectedNode && <LayerEditor selectedNode={selectedNode} updateNode={updateNode} />}
          </div>
          <ModelLibraryPanel onLoadTemplate={handleModelLoad} />
          <ConsoleLog logs={logs} />
        </div>
      </div>
    </div>
  );
}