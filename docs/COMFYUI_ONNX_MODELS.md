# ComfyUI ONNX 模型清单

插件默认复用 `scripts/demo/demo_soma_onnx.py`。**节点首次运行时会自动下载下表文件（`comfyui_gem_x/models.py`）**，本页用于核对或手动下载。

## 完整图像特征模式：image_features=sam3d

| 文件 | 保存位置，相对 ComfyUI models/gem_x | 约大小 |
|---|---|---:|
| `vitpose.onnx` + `vitpose.onnx.data` | `onnx/` | 3.40 GB |
| `gem_denoiser.onnx` + `gem_denoiser.onnx.data` | `onnx/` | 201 MB |
| `sam3db_backbone.onnx` + `sam3db_backbone.onnx.data` | `onnx/` | 3.36 GB |
| `gem_soma.ckpt`（仅 GEM PyTorch 回退需要） | `pretrained/` | 542 MB |
| `sam3d_body.ckpt` + `model_config.yaml` | `checkpoints/sam-3d-body-dinov3/` | 2.11 GB |
| `mhr_model.pt` | `mhr_data/` | 696 MB |
| `scale_mean.pth` + `scale_comps.pth` | `soma_data/` | 约 10 KB |

ONNX 成功加载时约 **9.8 GB**；加上可选 GEM PyTorch 回退权重约 **10.3 GB**，不包括 SOMA 资产、检测器及 Python 依赖。ONNX 文件在 [onnx 子目录](https://huggingface.co/nvidia/GEM-X/tree/main/onnx)，其余文件在仓库根目录。`.onnx` 和同名 `.onnx.data` 必须同时保留在同一目录，不要改名。

`sam3db_backbone.onnx` 只导出了骨干网络，现有代码仍需加载 `sam3d_body.ckpt` 和 MHR 模型来运行 PyTorch 解码器。插件已将 ONNX 路径改为直接实例化原来的固定统计 EnDecoder，不再为解码加载完整 GEM 和 `gem_soma.ckpt`；只有 GEM PyTorch 回退才需要该 checkpoint。无需另外下载 `vitpose.pth`，除非要使用其 PyTorch 回退。

## 关键点模式：image_features=keypoints_only

下载以下文件，共约 **4.7 GB**（加上可选 GEM checkpoint 约 5.3 GB），不包括下面的通用资产：

- `vitpose.onnx` + `vitpose.onnx.data`，放入 `onnx/`。
- `gem_denoiser_no_imgfeat.onnx` + `gem_denoiser_no_imgfeat.onnx.data`，放入 `onnx/`。
- 可选 `gem_soma.ckpt`，放入 `pretrained/`，仅 GEM PyTorch 回退需要。
- `scale_mean.pth` + `scale_comps.pth`，放入 `soma_data/`。

这个模式跳过 SAM3D 特征，不必下载 SAM3D backbone、SAM3D checkpoint 或作为 SAM3D 输入的 `mhr_data/mhr_model.pt`。SOMA 身体资产用于网格渲染及 GEM PyTorch 回退；数据导出的 ONNX 解码路径不再实例化身体模型。缺失图像特征会改变模型条件，效果需实际对比，不能当成与完整版等价。

## 两种模式都需要

- **YOLOX-X HumanArt 人物检测模型**：现有 `gem/utils/yolox_detector.py` 会下载 [OpenMMLab 的模型压缩包](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip)，并提取 ONNX。ComfyUI 插件将检测器保存在 `models/gem_x/yolox/`。独立 demo 原来的缓存位置保持不变。
- **SOMA 身体资产**：按 `third_party/soma/README.md` 获取真实 LFS 文件，将所需的资产目录内容放入 ComfyUI 的 `models/gem_x/soma_assets/`。GEM 初始化、身体计算及网格预览都会使用它们。不同 SOMA 身份模型的全部资产不必一并下载，也不需要为此另外下载 SMPL/SMPL-X。

注意版本：本仓库锁定的 SOMA 代码为 `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`。
2026-10-15 的服务器实测发现，ModelScope/HuggingFace 最新 `SOMA_neutral.npz` 已修改字段结构，不能与此代码直接混用。
以下两份文件应取自锁定版本的 [SOMA 资产目录](https://github.com/NVlabs/SOMA-X/tree/e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5/assets)，下载 Git LFS 的真实内容：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `SOMA_neutral.npz` | 27488638 | `515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f` |
| `correctives_model.pt` | 95437256 | `3b0f82928dc25c952e0a23af59dbc9c02854ccb4c4c1c988f70bbffd7c284636` |

此外，当前 `low_lod=True, identity_model_type=mhr` 还需 `MHR/mhr_model_lod6.pt`、`MHR/base_body_lod6.obj`、`MHR/SOMA_wrap_lod1.obj`。这三份文件经核对与镜像中的大小一致。

这些不是额外训练数据；无需下载训练集，也无需为 ONNX 模式下载 TensorRT `.engine`。先使用基础 ONNX；FP16/INT8/CoreML 转换产物是后续优化项。

插件直接使用 ComfyUI 的 `folder_paths.models_dir/gem_x`，无需链接 GEM-X 的 `inputs`。
