# 🚀 YOLOTools

YOLO深度学习工具集，提供环境检测和命令生成功能。

## 📚 相关文档

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

## 📁 文件说明

| 文件 | 功能 |
|------|------|
| `envCheck.py` | 环境检测脚本 - 检测PyTorch、CUDA、OpenCV等必需库是否安装 |
| `yoloCmdGen.py` | YOLO命令生成器 - 交互式生成训练/推理/验证/导出命令 |
| `codeRun.ipynb` | Jupyter Notebook - 实验代码 |
| `videos/` | 视频文件目录 |

## 🛠️ 使用方法

### ✅ 环境检测

```bash
python envCheck.py
```

检测项：PyTorch、TorchVision、Matplotlib、NumPy、OpenCV、Pillow、Ultralytics、GPU/CUDA状态

### ⚡ YOLO命令生成器

```bash
python yoloCmdGen.py
```

功能：
- 🎯 **训练** - 生成 `yolo train` 命令，支持YOLOv5/v8/v9/v10/v11
- 🔍 **推理** - 生成 `yolo predict` 命令，支持图片/视频/摄像头
- 📊 **验证** - 生成 `yolo val` 命令
- 📦 **导出** - 生成 `yolo export` 命令，支持ONNX/TensorRT等格式
