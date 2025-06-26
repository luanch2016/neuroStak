import React, { useState } from "react";
import "../App.css";
import Logo from "../assets/neuroStak.png";
import Conv2D from "../assets/Conv2D.svg";
import BatchNorm from "../assets/BatchNorm.svg";
import Dropout from "../assets/Dropout.svg";
import Linear from "../assets/Linear.svg";
import MaxPool2D from "../assets/MaxPool2D.svg";
import PReLU from "../assets/PReLU.svg";
import Skip from "../assets/skip_layer.svg";
import Tanh from "../assets/tanh.svg";
import Flatten from "../assets/Flatten.svg";
import ReLu from "../assets/ReLU.svg";
import LeakyReLU from "../assets/LeakyReLU.svg";
import Sigmoid from "../assets/sigmoid.svg";
import ELU from "../assets/ELU.svg";
import GELU from "../assets/GELU.svg";
import Softmax from "../assets/Softmax.svg";
import AdaptiveAvgPool2d from "../assets/AdaptiveAvgPool2d.svg";
import AvgPool2d from "../assets/AvgPool2d.svg";
import Concat from "../assets/Concat.svg";
import ConvTranspose2d from "../assets/ConvTranspose2d.svg";
import DepthwiseConv2d from "../assets/DepthwiseConv2d.svg";
import Dropout2d from "../assets/Dropout2d.svg";
import GlobalAvgPool2d from "../assets/GlobalAvgPool2d.svg";
import GroupNorm from "../assets/GroupNorm.svg";
import InstanceNorm2d from "../assets/InstanceNorm2d.svg";
import LayerNorm from "../assets/LayerNorm.svg";
import Reshape from "../assets/Reshape.svg";
import Upsample from "../assets/Upsample.svg";
import ResidualBlock from "../assets/ResidualBlock.svg";
import SeparableConv2d from "../assets/SeparableConv2d.svg";
import SqueezeExcitation from "../assets/SqueezeExcitation.svg";
import Embedding from "../assets/Embedding.svg";
import OneHotEncoding from "../assets/OneHotEncoding.svg";
import RNN from "../assets/RNN.svg";
import GRU from "../assets/GRU.svg";
import LSTM from "../assets/LSTM.svg";
import BiLSTM from "../assets/BiLSTM.svg";
import StackedLSTM from "../assets/StackedLSTM.svg";
import LSTMCell from "../assets/LSTMCell.svg";
import Attention from "../assets/Attention.svg";
import PackPadded from "../assets/PackPaddedSequence.svg";
import PadPacked from "../assets/PadPackedSequence.svg";
import TimeDistributed from "../assets/TimeDistributed.svg";

const layerGroups = {
  Convolution: [
    { name: "Conv2d", icon: Conv2D },
    { name: "ConvTranspose2d", icon: ConvTranspose2d },
    { name: "DepthwiseConv2d", icon: DepthwiseConv2d },
    { name: "SeparableConv2d", icon: SeparableConv2d },
    { name: "AdaptiveAvgPool2d", icon: AdaptiveAvgPool2d },

],
  NLP: [
    { name: "Embedding", icon: Embedding },
    { name: "OneHotEncoding", icon: OneHotEncoding },
  ],
  Sequence: [
    { name: "RNN", icon: RNN },
    { name: "GRU", icon: GRU },
    { name: "LSTM", icon: LSTM },
    { name: "BiLSTM", icon: BiLSTM },
    { name: "StackedLSTM", icon: StackedLSTM },
    { name: "LSTMCell", icon: LSTMCell },
    { name: "Attention", icon: Attention },
    { name: "PackPaddedSequence", icon: PackPadded },
    { name: "PadPackedSequence", icon: PadPacked },
    { name: "TimeDistributed", icon: TimeDistributed },
  ],
  Pooling: [
    { name: "MaxPool2d", icon: MaxPool2D },
    { name: "AvgPool2d", icon: AvgPool2d },
    { name: "GlobalAvgPool2d", icon: GlobalAvgPool2d },
    { name: "AdaptiveAvgPool2d", icon: AdaptiveAvgPool2d },
    { name: "Upsample", icon: Upsample },

],
  Normalization: [
    { name: "BatchNorm", icon: BatchNorm },
    { name: "GroupNorm", icon: GroupNorm },
    { name: "InstanceNorm2d", icon: InstanceNorm2d },
    { name: "LayerNorm", icon: LayerNorm },],
  Activation: [
    { name: "ReLU", icon: ReLu },
    { name: "LeakyReLU", icon: LeakyReLU },
    { name: "PReLU", icon: PReLU },
    { name: "ELU", icon: ELU },
    { name: "GELU", icon: GELU },
    { name: "Tanh", icon: Tanh },
    { name: "Sigmoid", icon: Sigmoid },
    { name: "Softmax", icon: Softmax },
  ],
  Regularization: [
    { name: "Dropout", icon: Dropout },
    { name: "Dropout2d", icon: Dropout2d },

  ],
  Linear: [
    { name: "Flatten", icon: Flatten },
    { name: "Skip", icon: Skip },
    { name: "Linear", icon: Linear },
  ],
  Advanced: [
    { name: "Concat", icon: Concat },
    { name: "Reshape", icon: Reshape },
    { name: "ResidualBlock", icon: ResidualBlock },
    { name: "SqueezeExcitation", icon: SqueezeExcitation },
    ],
};

export default function LayerNavbar() {
  const [openGroups, setOpenGroups] = useState({});

  const toggleGroup = (group) => {
    setOpenGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const onDragStart = (event, layerType) => {
    event.dataTransfer.setData("application/reactflow", layerType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
      
    <div className="layer-navbar-list">

      {Object.entries(layerGroups).map(([category, layers]) => (
        <div key={category} className="layer-group">
          <div
            className="layer-group-toggle"
            onClick={() => toggleGroup(category)}
          >
            {category}
          </div>
          {openGroups[category] && (
            <div className="layer-group-content">
              {layers.map((layer) => (
                <div
                  key={layer.name}
                  className="layer-navbar-item"
                  draggable
                  onDragStart={(event) => onDragStart(event, layer.name)}
                >
                  <img src={layer.icon} alt={layer.name} />
                  <span>{layer.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}