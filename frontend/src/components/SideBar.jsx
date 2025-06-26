import React from "react";

const layers = 
[
    "Conv2d", 
    "ReLU", 
    "LeakyReLU", 
    "PReLu",
    "sigmoid",
    "tanh",
    "MaxPool2d",
    "BatchNorm",
    "Skip",
    "DropOut", 
    "Flatten", 
    "Linear"

];

export default function Sidebar() {
  return (
    <aside className="w-60 bg-gray-100 p-4 border-r">
      <h2 className="text-xl font-bold mb-2">Layer Blocks</h2>
      {layers.map((layer) => (
        <div
          key={layer}
          className="p-2 mb-2 bg-white rounded shadow hover:bg-gray-200 cursor-pointer"
          draggable
          onDragStart={(e) => {
            e.dataTransfer.setData("application/reactflow", layer);
            e.dataTransfer.effectAllowed = "move";
          }}
        >
          {layer}
        </div>
      ))}
    </aside>
  );
}