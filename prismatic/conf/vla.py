"""
vla.py

Draccus 数据类定义，用于 VLAConfig 对象，包含各种已注册的子类，每个子类对应一个 VLA 实验和模型配置。
一个给定的 VLA 模型（`policy`）配置以下属性：
    - 数据混合（例如 Bridge、OXE_MAGIC_SOUP 等）
    - 来自 Prismatic 注册表的基础 VLM（例如 `prism-dinosiglip+7b`）
    - VLA 模型架构/参数（例如冻结视觉编码器、最后一层微调）
    - 训练/优化超参数
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # 唯一的 VLA 策略 ID，完全指定配置变体
    base_vlm: Union[str, Path]                      # 基础 VLM 作为 ID/运行目录路径（例如 `prism-dinosiglip+7b`）
    freeze_vision_backbone: bool                    # 冻结视觉骨干网络参数（类似于预训练）
    freeze_llm_backbone: bool                       # 冻结 LLM 骨干网络参数
    unfreeze_last_llm_layer: bool                   # 解冻 LLM 的最后一层（仅在 LLM 被冻结时生效）

    # 数据混合参数
    data_mix: str                                   # Open-X Embodiment 数据集 =>> 唯一混合 ID（例如 `bridge`）
    shuffle_buffer_size: int                        # 混洗缓冲区大小（Bridge 为 100K，OXE 为 1M）

    # 优化参数
    epochs: int                                     # 运行的轮数（如果未指定 `max_steps`）
    max_steps: Optional[int]                        # [可选] 最大梯度步数（覆盖 `epochs`）

    expected_world_size: int                        # 预期的 GPU 数量 =>> 允许我们根据硬件条件控制训练
    global_batch_size: int                          # 全局批次大小（在进程/世界大小之间分配）
    per_device_batch_size: int                      # 每个设备的批次大小（每个进程/单个 GPU）
                                                    #   =>> 累积步数自动计算

    learning_rate: float                            # 峰值学习率（`lr_scheduler_type` 设置预热/衰减）
    weight_decay: float                             # AdamW 优化器的权重衰减
    max_grad_norm: float                            # 最大梯度范数（用于全局梯度裁剪）
    lr_scheduler_type: str                          # 学习率调度器（通常为："constant" | "linear-warmup+cosine-decay"）
    warmup_ratio: float                             # 预热步数比例（用于预热学习率调度器）

    train_strategy: str                             # 训练策略（默认 "fsdp-full-shard"）

    # 启用梯度/激活检查点（针对 LLM 骨干网络）
    enable_gradient_checkpointing: bool = True      # 在训练期间启用梯度/激活检查点

    # 通过 Torch Native AMP（`autocast`）进行混合精度训练
    enable_mixed_precision_training: bool = True    # 启用传统的 BF16 混合精度
    reduce_in_full_precision: bool = True           # 在 FP32 全精度下累积/减少 All-Gather 梯度

    # fmt: on


# === OpenVLA 训练配置 ===


# = [8 GPU] 快速迭代 =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# = [8 GPU] SigLIP 224px 冻结视觉骨干网络 + Bridge =
@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True


# = [8 GPU] 快速迭代 =>> DINO-SigLIP 224px + Bridge =
@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    data_mix: str = "bridge"


# = [64 GPU] SigLIP 224px + OXE Magic Soup =
@dataclass
class Exp_SigLIP_224px_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "oxe_magic_soup"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# = [64 GPU] DINO-SigLIP 224px + OXE Magic Soup++ =
@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-oxe-magic-soup-plus"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # 注意 =>> 我们采用两阶段训练，在重新采样前，在包含 DROID 的混合数据上训练 70% 的训练时间！
    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# === OpenVLA 微调配置 ===


# = [8 GPU] SigLIP 224px + T-DROID =
@dataclass
class Exp_SigLIP_224px_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_TDROID_PourCornInPot(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_pour_corn_in_pot"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_pour_corn_in_pot"


# = [8 GPU] SigLIP 224px + T-DROID -- 部分微调 =
@dataclass
class Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-last_layer+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-sandwich+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


# === [8 GPU] SigLIP 224px + FrankaWipe ===
@dataclass
class Exp_SigLIP_224px_Droid_Wipe(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-droid_wipe"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid_wipe"


# === 定义 VLA 注册表枚举用于参考和验证 ===
@unique
class VLARegistry(Enum):
    # 完整性检查配置 =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge

    # SigLIP 冻结骨干网络实验
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge

    # [OpenVLA v0.1 7B] SigLIP 224px + OXE Magic Soup
    SIGLIP_224PX_MX_OXE_MAGIC_SOUP = Exp_SigLIP_224px_OXE_Magic_Soup

    # [OpenVLA 7B] DINO + SigLIP 224px + OXE Magic Soup++
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus

    # === TDROID 微调配置 ===
    SIGLIP_224PX_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_TDROID_CarrotInBowl
    SIGLIP_224PX_MX_TDROID_POUR_CORN_IN_POT = Exp_SigLIP_224px_TDROID_PourCornInPot

    SIGLIP_224PX_ICY_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl
    SIGLIP_224PX_LASTLAYER_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl
    SIGLIP_224PX_SANDWICH_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl

    # === DROID 微调配置 ===
    SIGLIP_224PX_MX_DROID_WIPE = Exp_SigLIP_224px_Droid_Wipe

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# 在 Choice Registry 中注册 VLA
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
