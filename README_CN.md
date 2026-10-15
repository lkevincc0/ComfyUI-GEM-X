# ComfyUI-GEM-X

[English](README.md) | 中文

在 ComfyUI 里用 NVIDIA [GEM-X](https://github.com/NVlabs/GEM-X) 从单目视频做全身动作捕捉（SOMA 人体，77 关节），并把动作重定向到 Mixamo / Unreal / Unity 的 FBX 角色，在 ComfyUI 自带的 3D 预览里直接播放。

视频 → 人物跟踪 → ViTPose 77 点 → SAM-3D-Body 特征 → GEM → SOMA 动作 → FBX 角色动画。

![GEM-X 工作流](docs/images/workflow.png)

*这张图里嵌入了工作流，直接拖进 ComfyUI 就能打开；同样的工作流也在 [example_workflows/](example_workflows)。*

## 安装

需要 NVIDIA GPU + CUDA，用 ComfyUI 自己的 Python 运行下面的命令。

**1. 把本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lkevincc0/ComfyUI-GEM-X.git
```

**2. 安装依赖：**

```bash
cd ComfyUI-GEM-X
pip install -r requirements.txt
python install.py
```

`install.py` 会创建插件自己的推理环境 `.venv`（复用 ComfyUI 已装的 PyTorch），安装推理依赖 `requirements-worker.txt`，并把自带的三个角色（含贴图）解压到 `ComfyUI/models/gem_x/fbx/`。用 ComfyUI-Manager 通过 Git URL 安装时会自动执行这一步。

**3. 重启 ComfyUI。** 节点出现在 `GEM-X` 分类下；示例工作流在 `example_workflows/`，可直接拖进 ComfyUI 打开。

可选环境变量：`GEM_X_PYTHON` 指定已有的推理 Python 环境（同时设 `GEM_X_SKIP_VENV=1` 跳过创建 `.venv`）；`COMFYUI_MODELS_DIR` 指定非默认的模型目录。

## 模型

模型在首次使用时自动下载到 `ComfyUI/models/gem_x/`，只下载所选模式需要的文件。先从 HuggingFace 下载，失败自动换 ModelScope 镜像，支持断点续传。

| 模型 | 用途 | 大小 | 需要的模式 |
|---|---|---:|---|
| YOLOX-X HumanArt | 人物检测 | 370 MB | 全部 |
| SOMA 身体资产 + scale 数据 | 人体模型、重定向 | 130 MB | 全部（含重定向节点） |
| ViTPose（ONNX） | 77 点 2D 关键点 | 3.4 GB | `onnx` |
| GEM denoiser（ONNX） | 动作回归 | 200 MB | `onnx` + `sam3d` |
| SAM-3D-Body backbone（ONNX） | 图像特征 | 3.4 GB | `onnx` + `sam3d` |
| SAM-3D-Body checkpoint + MHR | 图像特征解码 | 2.8 GB | `sam3d`、`pytorch` |
| GEM denoiser no-imgfeat（ONNX） | 动作回归（无图像特征） | 1.3 GB | `onnx` + `keypoints_only` |
| GEM checkpoint + ViTPose（PyTorch） | PyTorch 后端 | 3.9 GB | `pytorch` |

默认模式（`backend=onnx`、`image_features=sam3d`）共约 9.8 GB。也可以提前下载：

```bash
python -m comfyui_gem_x.models /path/to/ComfyUI/models/gem_x          # 默认模式
python -m comfyui_gem_x.models /path/to/ComfyUI/models/gem_x --all    # 全部模式
```

完整文件清单和手动下载地址见 [docs/COMFYUI_ONNX_MODELS.md](docs/COMFYUI_ONNX_MODELS.md)。

## 节点

| 节点 | 作用 |
|---|---|
| **GEM-X Video to SOMA Motion** | 视频 → SOMA 动作。输出 `motion`、`motion.npz`、关键点预览视频、结果清单；节点内可播放预览、下载结果。 |
| **GEM-X Retarget to FBX Character** | 把 `motion` 重定向到 `models/gem_x/fbx/` 里的角色，输出 `FILE_3D_FBX`、FBX 路径和镜头参数；接内置 **Preview 3D (Advanced)** 播放。 |
| **GEM-X Frames to Video** | 把 `IMAGE` 帧批次编码成临时 MP4，接视频生成等节点的输出。 |
| **GEM-X Video Path** | 直接填视频路径（绝对路径或相对 `input/`）。 |

示例工作流在 [example_workflows/](example_workflows)：Load Video → Trim Video（选片段）→ 动作捕捉 → 三个角色各自重定向并预览。

### 推理参数

- `backend`：默认 `onnx`；`pytorch` 才能开 `postprocess`（接触/平移后处理）。
- `image_features`：`sam3d` 用图像特征；`keypoints_only` 跳过 SAM3D，下载更少，但效果需自行对比。
- `preview`：`none` / `keypoints` / `mesh`（mesh 需要 Open3D/EGL）。
- 只处理跟踪到的一个主要人物；相机按静态处理；保留输入帧率，可变帧率视频请先转成恒定帧率。

### 重定向

- 按骨骼名自动识别：Mixamo（`mixamorig:` 前缀可有可无）、Unreal UE4/UE5/UEFN 人形、Unity Robot Kyle。Y-up/Z-up、T-pose/A-pose 自动对齐。
- 映射躯干、头、手臂、腿和脚趾；手指、扭转骨、IK 骨保持角色原姿态。根位移按腿长缩放；`root_motion` 关闭时原地播放；`yaw_offset` 调整朝向。
- `material=auto`：在 FBX 旁边（含子文件夹）按文件名找贴图，找到就嵌入输出 FBX（TGA 等转成 PNG）；一张都找不到时用纯橙色；也可选 `original` / `red` / `orange` / `gray` / `blue`。
- 自带角色：Mixamo X Bot、UEFN 人偶（无贴图，默认橙色）、Unity Robot Kyle（带颜色贴图）。放入新的 `.fbx` 到 `models/gem_x/fbx/`（可建子文件夹放贴图）即可在下拉框中选择。
- 导出后会回读动画自检（旋转误差 > 1° 报错），这只保证关键帧写入正确，动作好坏需看预览。

## 输出格式

`motion.npz` 可用 `numpy.load(path, allow_pickle=False)` 读取，包含 `body_params_incam/<参数>`、`body_params_global/<参数>`、`K_fullimg`、`fps`、`frame_count`。平移单位为米，姿态为 SOMA 轴角表示。`hpe_results.pt` 保留上游原始预测。

## 测试

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖工作进程调用、异常、取消、API 和 WebSocket 转发、视频范围请求、ZIP 内容以及 BVH 旋转回读；模型精度和 GPU 执行需另外进行端到端验证。

## 致谢与许可

基于 NVIDIA [GEM-X](https://github.com/NVlabs/GEM-X)，内置 [SOMA-X](https://github.com/NVlabs/SOMA-X) 和 [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body)（`third_party/`，含少量兼容性修复）。重定向思路参考 [ComfyUI-Kimodo](https://github.com/jtydhr88/ComfyUI-Kimodo)。

源码遵循 [Apache 2.0](LICENSE)，第三方组件见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md) 及 `third_party/` 下各自的 LICENSE。模型权重遵循 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)。`assets/characters.zip` 中的角色分别来自 Adobe Mixamo、Epic Games（UEFN）和 Unity，各自适用原平台的使用条款。

```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: A GENeralist Model for Human MOtion},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
