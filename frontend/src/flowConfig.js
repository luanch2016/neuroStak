export default [
    {
      id: "input",
      type: "input",
      position: { x: 100, y: 50 },
      data: {
        label: "Input Layer",
        layerType: "Input Layer",  // ✅ define layerType
        params: {},          // ✅ safe default
      },
    }
  ];