# cat vs dog small model

- A ultra-lightweight cat and dog image classification model
- with a size of only **274KB** 
- and CPU inference speed of **~7ms per image**.

## Features

- **Ultra-lightweight**: The model is only 274KB and can be embedded in any device
- **High-speed inference**: 7ms per image on CPU, can process 140 images per second
- **High accuracy**: 100% for standard poses, 50% for difficult poses
- **Cross-platform**: Supports ONNX format, can run on browsers and mobile devices
- **Easy deployment**: No GPU required, no complex environment needed

## Model Information

| Metric | Value |
|------|------|
| Model Size | 274 KB |
| Number of Parameters | 70K |
| Input Size | 32×32×3 |
| Output | Cat / Dog (Binary Classification) |
| Inference Speed | ~7ms (CPU) |
| Memory Usage | < 1MB |

## Environment library

- Onnxruntimes
- Pillow
- Numpy

## Instructions for use

```bash
python3 main.py [Folder path]
