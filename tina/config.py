from dataclasses import dataclass, field
from typing import Literal


# check ./recipes/MODEL_NAME/PT_METHOD/model_DATASET.yaml
@dataclass
class ModelPTConfig:
    # //*******Model post-training configs*******//
    model_post_train_type: Literal["grpo", "sft"] = field(default="grpo") # 指定后训练使用的训练方法
    model_post_train_dataset_name: str = field(default="curated_deepscaler") # 指定后训练使用的数据集
    model_post_train_dataset_config: str | None = field(default=None) # 数据集的额外配置信息

    rl_post_train_reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    rl_post_train_reward_weights: list[float] = field(default_factory=lambda: [2.0, 1.0])
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000) # 限制用于余弦相似度计算的文本最大长度
    repetition_n_grams: int = field(default=3) # 用于计算重复内容的n-gram大小
    repetition_max_penalty: float = field(default=-1.0) # 对重复内容的惩罚值
