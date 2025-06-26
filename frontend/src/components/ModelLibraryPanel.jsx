import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";

export default function ModelLibraryPanel({ onLoadTemplate }) {
  const [templates, setTemplates] = useState([]);

  useEffect(() => {
    import("../utils/modelLibrary.json")
      .then((module) => {
        setTemplates(module.default || module);
      })
      .catch((err) => {
        console.error("Failed to load model templates:", err);
      });
  }, []);

  const handleModelLoad = (templateName) => {
    const model = templates.find((t) => t.name === templateName);
    if (model) {
      onLoadTemplate(model);
    } else {
      console.warn("Model not found:", templateName);
    }
  };

  return (
    <div className="model-library-panel">
      <h4>📚 Model Templates</h4>
      {templates.map((template) => (
        <button key={template.name} onClick={() => handleModelLoad(template.name)}>
          {template.name}
        </button>
      ))}
    </div>
  );
}

ModelLibraryPanel.propTypes = {
  onLoadTemplate: PropTypes.func.isRequired,
};