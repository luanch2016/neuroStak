import React from "react";
import "../App.css"; // if styles are here

export default function ButtonPanel({ onGenerate, onExport, onExportTF, onExportSVG, onExportPDF, onSave, onLoad, onClear, onUndo, onRedo }) {
    return (
      <div className="button-panel">
        {/*<button className="btn blue" onClick={onGenerate}>
          Generate Code
        </button>*/}
        <div className="dropdown">
          <button className="btn gray">Export Options ▼</button>
          <div className="dropdown-menu">
            <button className="btn gray" onClick={onExport}>Pytorch</button>
            <button className="btn gray" onClick={onExportTF}>Tensorflow</button>
            <button className="btn gray" onClick={onExportSVG}>Export SVG</button>
            <button className="btn gray" onClick={onExportPDF}>Export PDF</button>
            <button className="btn gray" onClick={onSave}>Save Model</button>
          </div>
        </div>
        <label className="btn orange">
          Load Model
          <input type="file" accept=".json" onChange={onLoad} style={{ display: "none" }} />
        </label>
        <button className="btn gray" onClick={onClear}>
          Clear
        </button>
        <button className="btn gray" onClick={onUndo}>
          ↩️
        </button>
        <button className="btn gray" onClick={onRedo}>
          ↪️
        </button>
      </div>
    );
  }