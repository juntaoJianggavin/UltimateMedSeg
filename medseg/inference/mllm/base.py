"""MLLM Grounder 抽象基类与数据结构。

# Reference: https://github.com/QwenLM/Qwen2-VL
# Reference: https://github.com/OpenGVLab/InternVL
# Reference: https://github.com/IDEA-Research/GroundingDINO
# Paper: https://arxiv.org/abs/2409.12191  (Qwen2-VL technical report)
# Paper: https://arxiv.org/abs/2303.05499  (Grounding DINO, ECCV 2024)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class BBox:
    """归一化到 [0, 1] 的 bbox。"""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ""

    def to_pixel(self, w: int, h: int) -> Tuple[int, int, int, int]:
        return (
            int(self.x1 * w),
            int(self.y1 * h),
            int(self.x2 * w),
            int(self.y2 * h),
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class DetectionResult:
    """单张图像的 grounding 结果（按类别组织）。"""
    image_shape: Tuple[int, int]                       # (H, W)
    boxes_by_class: Dict[str, List[BBox]] = field(default_factory=dict)
    raw_response: str = ""                              # MLLM 原始输出文本

    def flatten(self) -> List[BBox]:
        out = []
        for boxes in self.boxes_by_class.values():
            out.extend(boxes)
        return out

    def num_boxes(self) -> int:
        return sum(len(v) for v in self.boxes_by_class.values())


class MLLMGrounder:
    """所有 MLLM grounder 的抽象基类。

    子类必须实现 _load_model() 和 detect()。

    Strict policy: 如果模型权重 / 依赖缺失，子类的 _load_model() 应该 raise，
    不允许自动切换到 self.mock_mode = True 用 mock 输出代替真实推理。
    mock_mode 仍是公开字段，调用方若需要做装配/单元测试可以在外部显式设置
    (instance.mock_mode = True)，但不会再有任何路径在加载失败时自动触发。
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: str = "Locate the {class_name} in this image. Output bbox.",
        max_new_tokens: int = 256,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.extra_kwargs = kwargs

        self.model = None
        self.processor = None
        self.mock_mode = False
        self._load_model()

    # ------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------
    def _load_model(self) -> None:
        raise NotImplementedError

    def _detect_single_class(
        self,
        image: np.ndarray,
        class_name: str,
    ) -> List[BBox]:
        """对一张图、一个类别名做 grounding，返回若干 bbox。"""
        raise NotImplementedError

    # ------------------------------------------------------------
    # 通用入口（按类别串行调用 _detect_single_class）
    # ------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        class_names: List[str],
    ) -> DetectionResult:
        """对一张图，依次 ground 多个类别。

        Args:
            image: (H, W, 3) uint8/float RGB 图像
            class_names: 要定位的类别名列表

        Returns:
            DetectionResult
        """
        h, w = image.shape[:2]
        result = DetectionResult(image_shape=(h, w))
        for name in class_names:
            boxes = self._detect_single_class(image, name)
            result.boxes_by_class[name] = boxes
        return result

    # ------------------------------------------------------------
    # Mock fallback：未安装依赖/无权重时使用
    # ------------------------------------------------------------
    def _mock_detect_single_class(
        self,
        image: np.ndarray,
        class_name: str,
    ) -> List[BBox]:
        """生成一个居中的 mock bbox（用于 pipeline 装配验证）。"""
        # 简单 hash 让不同类别得到不同 bbox 中心
        seed = abs(hash(class_name)) % 100 / 100.0
        cx = 0.3 + 0.4 * seed
        cy = 0.3 + 0.4 * (1.0 - seed)
        half = 0.12
        return [
            BBox(
                x1=max(0.0, cx - half),
                y1=max(0.0, cy - half),
                x2=min(1.0, cx + half),
                y2=min(1.0, cy + half),
                score=0.5,
                label=class_name,
            )
        ]
