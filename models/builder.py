import torch.nn as nn

block_map = {
    "Conv2d": nn.Conv2d,
    "ReLU": nn.ReLU,
    "MaxPool2d": nn.MaxPool2d,
    "Flatten": nn.Flatten,
    "Linear": nn.Linear
}

def build_model(net_def):
    layers = []
    for layer_def in net_def.layers_sequence:
        layer_type = block_map[layer_def.block_id]
        layer = layer_type(**layer_def.params)
        layers.append(layer)
    return nn.Sequential(*layers)

def generate_code(net_def):
    code_lines = ["import torch.nn as nn", "", "class GeneratedNet(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    code_lines.append("        self.model = nn.Sequential(")
    for layer in net_def.layers_sequence:
        params = ", ".join([f"{k}={v}" for k, v in layer.params.items()])
        code_lines.append(f"            nn.{layer.block_id}({params}),")
    code_lines.append("        )")
    code_lines.append("")
    code_lines.append("    def forward(self, x):")
    code_lines.append("        return self.model(x)")
    return "\n".join(code_lines)