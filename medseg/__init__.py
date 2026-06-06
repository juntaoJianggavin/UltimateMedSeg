"""MedSeg: 模块化 2D 医学图像分割框架。
MedSeg: Modular 2D Medical Image Segmentation Framework.

目录结构 / Package structure:
    medseg/
    ├── models/          # 逻辑分组入口 / Logical group entry
    │   (实际代码 / actual code in:)
    │   ├── encoders/        编码器 / Encoders
    │   ├── decoders/        解码器 / Decoders
    │   ├── bottlenecks/     瓶颈层 / Bottleneck layers
    │   ├── skip_connections/ 跳跃连接 / Skip connections
    │   ├── networks/        完整网络 / Complete architectures
    │   ├── text_unet/       文本引导模型 / Text-guided models
    │   └── losses/          损失函数 / Loss functions
    ├── training/        # 逻辑分组入口 / Logical group entry
    │   (实际代码 / actual code in:)
    │   ├── semi/            半监督 / Semi-supervised
    │   ├── domain_adaptation/ 域适应 / Domain adaptation
    │   ├── distillation/    知识蒸馏 / Knowledge distillation
    │   └── weakly_supervised/ 弱监督 / Weakly supervised
    ├── inference/       推理 / Inference (mllm (含 medisee), ensemble, tta)
    ├── datasets/        数据加载 / Data loading
    ├── utils/           工具 / Utilities
    └── kernels/         CUDA kernels
"""

from .model_builder import build_model, SegmentationModel

# 注册所有组件 / Register all components
# --- 模型组件 / Model components (medseg.models.*) ---
from .models import encoders
from .models import decoders
from .models import bottlenecks
from .models import skip_connections

# --- 损失函数 / Loss functions ---
from . import losses

# --- 训练范式 / Training paradigms (medseg.training.*) ---
from .training import semi
from .training import domain_adaptation
from .training import distillation
from .training import weakly_supervised
from .models import text_unet

__version__ = "0.1.0"
