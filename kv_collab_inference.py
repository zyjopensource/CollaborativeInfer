from typing import Union

from mindspeed_llm import megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.legacy.model import GPTModel
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.infer_base import task_factory
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM

import os
import torch
from contextlib import nullcontext, contextmanager
from transformers import AutoTokenizer
import logging
from torch import distributed as dist
import statistics

def model_provider(pre_process=True, post_process=True) -> Union[GPTModelInfer, GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModelInfer, GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.sequence_parallel and args.use_kv_cache:
        raise AssertionError('Use_kv_cache can not be true in sequence_parallel mode.')

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = GPTModel(
            config,
            parallel_output=True if args.sequence_parallel else False,
            pre_process=pre_process,
            post_process=post_process
        )

    return model

@contextmanager
def override_args_temporarily(**overrides):
    """
    临时覆写 Megatron 的全局 args（get_args() 返回的单例），
    with 作用域内生效，退出后自动还原。
    """
    args = get_args()
    backup = {}
    for k, v in overrides.items():
        backup[k] = getattr(args, k, None)
        setattr(args, k, v)
    try:
        yield
    finally:
        for k, v in backup.items():
            setattr(args, k, v)

def main():
    initialize_megatron(args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    # ====== 先加载小模型：沿用命令行里的参数 ======
    small = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    collab_enable = os.getenv("COLLAB_ENABLE", "0") == "1"
    large_ckpt = os.getenv("LARGE_CHECKPOINT", "")

    if collab_enable and large_ckpt:
        # ====== 加载大模型：进入临时覆写作用域 ======
        # 专属参数
        large_overrides = {
            "num_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "ffn_hidden_size": 12288,
            "max_position_embeddings": 4096,
            "untie_embeddings_and_output_weights": True,
        }
        # 如果 spec/tokenizer/vocab 等也不同，可在此继续覆写：
        # large_overrides.update({
        #     "padded_vocab_size": 151936,
        #     "group_query_attention": True,
        #     "num_query_groups": 40,
        # })

        # 覆写 ckpt 路径 + 超参，构建完立即还原
        _old_load = args.load
        args.load = large_ckpt
        with override_args_temporarily(**large_overrides):
            large = MegatronModuleForCausalLM.from_pretrained(
                model_provider=model_provider,
                pretrained_model_name_or_path=args.load
            )
        args.load = _old_load

        # ====== 包装为协作模型（保持 .generate 接口不变） ======
        model = CollaborativeCausalLM(
            small_model=small,
            large_model=large,
            args=args
        )
    else:
        model = small

    task_factory(args, model)

class SimpleKVState:
    """
    轻量版 InferenceParams：补齐 Megatron 推理路径需要访问的属性
    """
    def __init__(self, max_batch_size: int, max_sequence_length: int, device: torch.device):
        # —— 必需字段 ——（命名对齐 Megatron）
        self.max_batch_size = int(max_batch_size)
        self.max_sequence_length = int(max_sequence_length)

        # 当前这一批（batch=1）
        self.batch_size = 1
        self.batch_size_offset = 0  # 多批并行时的起始下标；我们固定 0

        # 序列写入偏移（增量步从 seen 处写）
        self.sequence_len_offset = 0
        # 兼容某些版本可能访问的别名
        self.sequence_length_offset = 0

        # 每个样本当前上下文长度：IntTensor[batch]，增量时 Megatron 会用它推算位置
        self.lengths_per_sample = torch.zeros((self.batch_size,), dtype=torch.int32, device=device)

        # KV 全局字典：由各层在首次前向时按需填充/复用
        self.key_value_memory_dict = {}

        # 兼容性字段（部分分支可能读取）
        self.is_generation_step = True
        self.fused_ft_kernel = False

class KVCacheHelper:
    def __init__(self, module, max_seq: int):
        self.module = module
        self._core = CollaborativeCausalLM._get_core(module)
        # 取模型真实设备
        self.device = next(self._core.parameters()).device

        # 构造 KV 状态（带正确的 device）
        self.kv = SimpleKVState(max_batch_size=1, max_sequence_length=int(max_seq), device=self.device)

        self.seen = 0
        self._last_logits = None

    @staticmethod
    def _build_incremental_mask(offset: int, q_len: int, k_len: int, device):
        """
        构造布尔因果掩码（True=屏蔽），形状 [1,1,q_len,k_len]
        对于第 i 个 query（其绝对位置为 offset+i），允许看到 key 位置 <= offset+i
        也就是 mask[col > offset+i] = True
        """
        arange_k = torch.arange(k_len, device=device).view(1, 1, 1, k_len)          # [1,1,1,k_len]
        allowed  = offset + torch.arange(q_len, device=device).view(1, 1, q_len, 1) # [1,1,q_len,1]
        mask = arange_k > allowed   # True 表示未来位，需要屏蔽
        return mask

    @staticmethod
    def _build_causal_mask(seq_len, device):
        return torch.triu(torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def _update_lengths(self):
        # 将“当前上下文长度”写回到 inference_params
        # 注意：dtype 为 int32，shape = [batch]
        self.kv.lengths_per_sample = torch.tensor([self.seen], dtype=torch.int32, device=self.device)
        self.kv.batch_size = 1
        self.kv.batch_size_offset = 0
        self.kv.sequence_len_offset = int(self.seen)
        self.kv.sequence_length_offset = int(self.seen)  # 兼容别名

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        B, T = input_ids.shape
        assert B == 1

        # 首帧写入从 0 开始
        self.seen = 0
        self.kv.sequence_len_offset = 0
        self.kv.sequence_length_offset = 0
        self.kv.batch_size = 1
        self.kv.batch_size_offset = 0
        self.kv.lengths_per_sample = torch.tensor([T], dtype=torch.int32, device=self.device)

        pos = torch.arange(0, T, device=self.device, dtype=torch.long).unsqueeze(0)
        mask = self._build_causal_mask(T, self.device)

        outputs = self._core(input_ids, pos, mask, inference_params=self.kv)
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        self.seen = T
        self._last_logits = logits[:, -1, :].float()
        return self._last_logits

    def append_one(self, new_token_id: int) -> torch.Tensor:
        ids = torch.tensor([[new_token_id]], device=self.device, dtype=torch.long)

        # 本步写入位置/偏移
        self.kv.sequence_len_offset = int(self.seen)
        self.kv.sequence_length_offset = int(self.seen)
        self.kv.batch_size = 1
        self.kv.batch_size_offset = 0

        # 位置 id：当前写入的是绝对位置 self.seen
        pos = torch.tensor([[self.seen]], device=self.device, dtype=torch.long)

        # 注意：此时 K_len = seen + 1，Q_len = 1
        q_len = 1
        k_len = self.seen + 1
        attn_mask = self._build_incremental_mask(self.seen, q_len, k_len, self.device)

        outputs = self._core(ids, pos, attn_mask, inference_params=self.kv)
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        self.seen += 1
        self.kv.lengths_per_sample = torch.tensor([self.seen], dtype=torch.int32, device=self.device)

        self._last_logits = logits[:, -1, :].float()
        return self._last_logits

    def append_chunk(self, new_token_ids: list[int]) -> torch.Tensor:
        if not new_token_ids:
            return self._last_logits

        N = len(new_token_ids)
        ids = torch.tensor([new_token_ids], device=self.device, dtype=torch.long)

        self.kv.sequence_len_offset = int(self.seen)
        self.kv.sequence_length_offset = int(self.seen)
        self.kv.batch_size = 1
        self.kv.batch_size_offset = 0

        pos = torch.arange(self.seen, self.seen + N, device=self.device, dtype=torch.long).unsqueeze(0)

        # 此次增量：Q_len = N，K_len = seen + N
        q_len = N
        k_len = self.seen + N
        attn_mask = self._build_incremental_mask(self.seen, q_len, k_len, self.device)

        outputs = self._core(ids, pos, attn_mask, inference_params=self.kv)
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        self.seen += N
        self.kv.lengths_per_sample = torch.tensor([self.seen], dtype=torch.int32, device=self.device)

        self._last_logits = logits[:, -1, :].float()
        return self._last_logits

class CollaborativeCausalLM:
    """
    词元级跨模型协作：
      - 每步使用小模型S做前向，计算最后一位logits -> gating判定
      - 若触发，用大模型L在同一前缀上前向并产生本token
    """
    def __init__(self, small_model, large_model, args):
        self.S = small_model.eval()
        self.L = large_model.eval()
        self.args = args

        # 阈值超参从环境变量读取
        self.p_thr = float(os.getenv("COLLAB_P_THR", "0.5"))
        self.h_thr = float(os.getenv("COLLAB_H_THR", "3.5"))
        self.margin = float(os.getenv("COLLAB_MARGIN", "0.06"))
        self.min_L_steps = max(0, int(os.getenv("COLLAB_MIN_L_STEPS", "2")))
        self.max_new = int(os.getenv("COLLAB_MAX_NEW_TOKENS", str(getattr(args, "max_new_tokens", 256))))
        self.mark_large = os.getenv("COLLAB_MARK_LARGE", "0") == "1"
        self.l_warmup = max(0, int(os.getenv("COLLAB_L_WARMUP_TOKENS", "0")))

        # 预算模式与阈值
        self.budget_enable = os.getenv("COLLAB_BUDGET_ENABLE", "0") == "1"
        self.tau = float(os.getenv("COLLAB_LARGE_FRAC_MAX", "0.25"))
        self.tau = max(0.0, min(1.0, self.tau))  # clip 到 [0,1]

        # lambda及其更新超参（乘法+积分）
        self.lambda_price = float(os.getenv("COLLAB_LAMBDA_INIT", "0.6"))
        self.lr  = float(os.getenv("COLLAB_LR", "0.08"))   # 比例（乘法）步长
        self.ki  = float(os.getenv("COLLAB_KI", "0.03"))   # 积分项权重
        self._e_acc = 0.0  # 误差积分

        # 冷却计数：触发后继续用L多少步，防止抖动
        self.use_L_cooldown = 0

        # 精度环境
        self._autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if getattr(args, "bf16", False)
            else (torch.autocast("cuda", dtype=torch.float16) if getattr(args, "fp16", False)
                  else nullcontext())
        )

        # tokenizer：优先用模型自带，否则用 --tokenizer-name-or-path / TOKENIZER_PATH
        tok = getattr(self.S, "tokenizer", None)
        if tok is None:
            tok_path = getattr(self.args, "tokenizer_name_or_path", None) or os.getenv("TOKENIZER_PATH", None)
            assert tok_path is not None, "找不到tokenizer路径：请设置 --tokenizer-name-or-path 或 TOKENIZER_PATH"
            use_fast = not getattr(self.args, "tokenizer_not_use_fast", False)
            tok = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)
        self.tokenizer = tok
        self.bos_id = getattr(self.tokenizer, "bos_token_id", None)

        # KV 管理器占位
        self._kv_S = None
        self._kv_L = None

        # L 的“未追平”尾部 token 暂存，用于 catch-up
        self._l_tail = []
        self._catchup_stride = max(0, int(os.getenv("COLLAB_CATCHUP_STRIDE", "0")))

        # 统计小/大模型产出的token数量
        self.s_count = 0
        self.l_count = 0
        self.last_stats = {"small": 0, "large": 0}

        # 输出可视化：对 large 产出的连续片段加 []
        self.mark_large = os.getenv("COLLAB_MARK_LARGE", "0") == "1"
        # 记录生成token的分段：(start_idx, end_idx_exclusive, is_large)
        self._spans = []

        # gating 指标采集：每步记录小模型 s_logits 的 p1、H、margin
        self._p1_list = []
        self._H_list = []
        self._margin_list = []
        self.last_gating_stats = {}   # 生成结束后存放整段统计

    def _log_collab_stats(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            self.last_stats = {"small": self.s_count, "large": self.l_count}
            portion = self.l_count / (self.s_count + self.l_count)
            logging.info(f"[Collaborative] tokens by small={self.s_count}, large={self.l_count}, portion={portion}")

    # 当步允许的累计大模型用量（配额轨道 A_t）
    def _allowance(self, step):
        # step 从 1 开始
        return int(self.tau * step)

    # 用 top-1 概率构造收益分数
    def _score_from_p1(self, p1: float) -> float:
        return 1.0 - float(p1)

    # lambda的在线更新（乘法 + 加性(积分)）
    def _update_lambda(self, l_count: int, allowance: int):
        # e_t = 实际累计 - 当步允许累计；>0 表示超前，用多了
        e = float(l_count - allowance)
        self._e_acc += e

        # 乘法比例调节
        if e > 0:
            self.lambda_price *= (1.0 + self.lr)
        elif e < 0:
            self.lambda_price *= max(0.0, (1.0 - self.lr))

        # 加性积分项
        self.lambda_price += self.ki * self._e_acc

        # 数值稳定：裁剪
        lambda_min, lambda_max = 0.4, 1.0
        self.lambda_price = float(min(max(self.lambda_price, lambda_min), lambda_max))

    # ---- 计算单步 gating 指标（基于小模型的 logits） ----
    @staticmethod
    def _metrics_from_logits(logits):
        probs = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values[0]
        p1 = float(top2[0].item())
        p2 = float(top2[1].item())
        margin = p1 - p2
        H = float(-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).item())
        return p1, H, margin

    # ---- 记录一帧指标 ----
    def _record_gating(self, p1, H, margin):
        self._p1_list.append(p1)
        self._H_list.append(H)
        self._margin_list.append(margin)

    # ---- 汇总统计值 ----
    @staticmethod
    def _summarize_array(xs):
        n = len(xs)
        if n == 0:
            return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
        # mean/median/std 用 statistics；std 用总体标准差（pstdev），避免 n==1 报错
        return {
            "n": n,
            "min": min(xs),
            "max": max(xs),
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "std": statistics.pstdev(xs) if n > 1 else 0.0,
        }

    # ---- 在结尾打印 gating 统计 ----
    def _log_gating_stats(self):
        p1_stats = self._summarize_array(self._p1_list)
        H_stats = self._summarize_array(self._H_list)
        m_stats = self._summarize_array(self._margin_list)
        self.last_gating_stats = {"p1": p1_stats, "H": H_stats, "margin": m_stats}

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            def fmt(s):  # 简洁格式化
                if s["n"] == 0: return "n=0"
                return (f"n={s['n']} min={s['min']:.6f} median={s['median']:.6f} "
                        f"mean={s['mean']:.6f} max={s['max']:.6f} std={s['std']:.6f}")
            #logging.info("[Collaborative][gating] p1   stats: %s", fmt(p1_stats))
            #logging.info("[Collaborative][gating] H    stats: %s", fmt(H_stats))
            #logging.info("[Collaborative][gating] p1-p2 stats: %s", fmt(m_stats))

    # ---- 更新分段：在 append 新token之后调用 ----
    def _update_spans(self, is_large: bool, out_tokens_len: int):
        """ out_tokens_len 是当前生成序列长度（追加完新token之后的长度） """
        idx = out_tokens_len - 1  # 刚刚追加的新token索引
        if not self._spans:
            self._spans.append([idx, idx + 1, is_large])
            return
        s, e, last_large = self._spans[-1]
        # 若与上一个片段同来源且连续，则扩展 end；否则新起一段
        if last_large == is_large and e == idx:
            self._spans[-1][1] = idx + 1
        else:
            self._spans.append([idx, idx + 1, is_large])

    # ---- 清空所有统计/分段（每次生成开始时调用） ----
    def _reset_stats_all(self):
        self.s_count = 0
        self.l_count = 0
        self._p1_list.clear()
        self._H_list.clear()
        self._margin_list.clear()
        self.last_stats = {"small": 0, "large": 0}
        self.last_gating_stats = {}
        self._spans.clear()

    # ---- 渲染带 [] 的文本（按分段分别decode并拼接） ----
    def _render_annotated(self, out_tokens: list[int]) -> str:
        if not self._spans:
            return self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        parts = []
        for s, e, is_large in self._spans:
            seg_ids = out_tokens[s:e]
            seg_text = self.tokenizer.decode(seg_ids, skip_special_tokens=True)
            if is_large:
                parts.append("[" + seg_text + "]")
            else:
                parts.append(seg_text)
        return "".join(parts)

    # --------- 工具：拿到底层可调用的 core 模型（不同封装名的兜底） ----------
    @staticmethod
    def _get_core(module):
        for name in ["model", "module", "model_module", "gpt", "core"]:
            core = getattr(module, name, None)
            if core is not None:
                return core
        # 最后兜底：直接用本体
        return module

    # --------- 工具：构建因果mask与position ids ----------
    @staticmethod
    def _build_mask_pos(seq_len, device):
        # 大seq会占显存；此处为了接口兼容性采用标准[1,1,seq,seq]下三角mask
        attn = torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool).tril_()
        pos = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0)
        return attn, pos

    # --------- 前向：整序列前向，取最后一位logits ----------
    @torch.no_grad()
    def _forward_last_logits(self, module, input_ids):
        """
        显式提供 position_ids 与 布尔因果 attention_mask：
          - position_ids: [1, seq]，0..seq-1
          - attention_mask: [1,1,seq,seq]，上三角(未来位)=True -> 被屏蔽；下三角/对角=False
        满足栈里 fused_softmax 对 mask 的 dtype/语义要求。
        返回最后一位 logits [1, V]（用 fp32 以稳住 softmax 数值）
        """
        core = self._get_core(module)
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # 1) position_ids
        position_ids = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq]

        # 2) 布尔因果掩码：True 表示屏蔽（未来位）；False 表示可见（历史+自身）
        attention_mask = torch.triu(
            torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1
        )  # 上三角 True

        # 3) 前向（栈要求位置参数形态）
        outputs = core(input_ids, position_ids, attention_mask)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits[:, -1, :].float()  # 采样前用 fp32 更稳

    # --------- gating：基于top1概率/熵/top1-top2间隔 ----------
    @staticmethod
    def _entropy(probs):
        return -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).item()

    def _need_large(self, logits):
        probs = torch.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values[0]
        p1, p2 = top2[0].item(), top2[1].item()
        H = self._entropy(probs)
        cond = (p1 < self.p_thr)
        # 更完整的条件：
        #cond = (p1 < self.p_thr) or (H > self.h_thr) or ((p1 - p2) < self.margin)
        return cond

    # --------- 采样/贪心 ----------
    @staticmethod
    def _sample_from_logits(logits, temperature=1.0, do_sample=False, top_k=50, top_p=1.0):
        logits = logits.float() / max(temperature, 1e-6)
        #logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if not do_sample:
            return torch.argmax(probs, dim=-1).item()

        # top-k
        if top_k and top_k > 0:
            kth = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
            mask = probs < kth.values[..., -1].unsqueeze(-1)
            probs = probs.masked_fill(mask, 0.0)

        # top-p
        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= top_p
            # 至少保留第一个
            keep[..., 0] = True
            filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_local = torch.multinomial(filtered, num_samples=1).item()
            return sorted_idx[0, next_local].item()

        return torch.multinomial(probs, num_samples=1).item()

    # --------- 生成：无yield外壳，按需返回字符串或迭代器 ---------
    @torch.no_grad()
    def generate(self, prompts, do_sample=False, top_k=50, top_p=1.0, temperature=1.0,
                 max_new_tokens=None, stream=False, broadcast=False, **kwargs):
        if isinstance(prompts, list):
            assert len(prompts) == 1, "当前实现仅支持batch=1"
            prompt = prompts[0]
        else:
            prompt = prompts

        if stream:
            # 返回一个可迭代对象（生成器），供 chat 模式逐步消费
            return self._stream_generate(prompt, do_sample, top_k, top_p, temperature, max_new_tokens)
        else:
            # 非流式：直接跑完并返回字符串
            return self._nonstream_generate(prompt, do_sample, top_k, top_p, temperature, max_new_tokens)

    # --------- 非流式实现：返回最终字符串（无yield） ---------
    def _nonstream_generate(self, prompt, do_sample, top_k, top_p, temperature, max_new_tokens):
        # 设备
        device = torch.cuda.current_device()

        # ---- 编码 + BOS ----
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.bos_id is not None and (len(ids) == 0 or ids[0] != self.bos_id):
            ids = [self.bos_id] + ids
        input_ids = torch.tensor([ids], dtype=torch.long).cuda(non_blocking=True)

        # 统计计数 + gating 指标归零
        self._reset_stats_all()
        tgt_len = max_new_tokens or self.max_new
        out_tokens = []

        warmup_rem = min(self.l_warmup, tgt_len)  # L 的强制预热步数
        total_budget = int(self.tau * tgt_len) if self.budget_enable else None

        # ==== 为 S/L 建立 KV 管理器 ====
        max_pos = int(getattr(self.args, "max_position_embeddings", 32768))
        ctx_max = min(max_pos, input_ids.shape[1] + tgt_len + 2)

        self._kv_S = KVCacheHelper(self.S, ctx_max)
        s_logits = self._kv_S.prefill(input_ids)       # S：整段 prefill

        self._kv_L = KVCacheHelper(self.L, ctx_max)
        l_prefill_done = False
        if warmup_rem > 0 or self._catchup_stride > 0:
            # 若需要预热或定期追平，先把 L prefill，避免第一次触发时全量成本
            self._kv_L.prefill(input_ids)
            l_prefill_done = True

        with self._autocast:
            step = 0
            while step < tgt_len:
                # === 预算参数准备 ===
                t = step + 1
                if self.budget_enable:
                    allow = self._allowance(t)
                    B_rem = total_budget - self.l_count
                    quota_ok = (self.l_count < allow) and (B_rem > 0)
                    can_start_L = (B_rem >= max(1, self.min_L_steps))

                # === 看是否处于 L 的强制预热期 ===
                force_L_now = (warmup_rem > 0)

                # === 基于 S 的 gating 指标 ===
                p1, H, margin = self._metrics_from_logits(s_logits)
                self._record_gating(p1, H, margin)
                score = self._score_from_p1(p1)  # 仅 1-p1

                # === 决策是否使用 L ===
                if force_L_now:
                    use_L = True
                elif self.budget_enable:
                    # 预算模式
                    if self.use_L_cooldown > 0 and (total_budget - self.l_count) > 0:
                        self.use_L_cooldown = min(self.use_L_cooldown, total_budget - self.l_count)
                        use_L = True
                    else:
                        use_L = quota_ok and can_start_L and (score >= self.lambda_price)
                else:
                    # 无预算
                    gating_use_L = self._need_large(s_logits)
                    use_L = True if self.use_L_cooldown > 0 else gating_use_L

                if use_L:
                    if os.getenv("COLLAB_DEBUG_SYNC", "1") == "1":
                        ctx_len = input_ids.shape[1] + len(out_tokens)
                        S_seen = self._kv_S.seen
                        L_seen = self._kv_L.seen if self._kv_L else -1
                        need   = max(0, ctx_len - L_seen)

                        # 1) 小模型必须始终追上完整上下文
                        assert S_seen == ctx_len, f"S desync: S_seen={S_seen}, ctx={ctx_len}"

                        # 2) 大模型不允许超过上下文；若 need==0 则必须相等
                        assert L_seen <= ctx_len, f"L overshoot: L_seen={L_seen}, ctx={ctx_len}"
                        if need == 0:
                            assert L_seen == ctx_len, f"L should be caught up: L_seen={L_seen}, ctx={ctx_len}"

                        logging.info(f"[L sync] before P={len(out_tokens)}, L_seen={L_seen}, need={need}, tail={self._l_tail[-4:]}")
                    # ---- 确保 L 的 KV 已追平到当前上下文 ----
                    # 当前上下文长度 = 提示词 + 已输出 out_tokens
                    cur_ctx_len = input_ids.shape[1] + len(out_tokens)
                    if not l_prefill_done:
                        self._kv_L.prefill(input_ids)
                        l_prefill_done = True
                    if self._kv_L.seen < cur_ctx_len:
                        # 需要把 [已生成但 L 未见过] 的尾部补进去
                        tail_needed = out_tokens[self._kv_L.seen - input_ids.shape[1]:]
                        self._kv_L.append_chunk(tail_needed)
                        self._l_tail.clear()  # 已经追平

                    # 用 L 产生当前 token
                    l_logits_cur = self._kv_L._last_logits  # 追平或 prefill 后，_last_logits 就是当前位置的分布
                    next_id = self._sample_from_logits(l_logits_cur, temperature, do_sample, top_k, top_p)

                    # 前进一步：把 next_id 追加进 L/S 的 KV
                    self._kv_L.append_one(next_id)
                    s_logits = self._kv_S.append_one(next_id)  # 保证下一步 S 的 logits 就绪

                    out_tokens.append(next_id)
                    self.l_count += 1
                    self._update_spans(is_large=True, out_tokens_len=len(out_tokens))

                    # 冷却与预算跟进
                    if self.use_L_cooldown > 0:
                        self.use_L_cooldown -= 1
                    else:
                        if self.min_L_steps > 1:
                            # 不要把冷却步数超过剩余预算
                            if self.budget_enable:
                                B_rem_after = max(0, (total_budget - self.l_count))
                                self.use_L_cooldown = min(self.min_L_steps - 1, B_rem_after)
                            else:
                                self.use_L_cooldown = self.min_L_steps - 1
                        else:
                            self.use_L_cooldown = 0

                    if warmup_rem > 0:
                        warmup_rem -= 1

                else:
                    # ---- 用 S 产生当前 token ----
                    next_id = self._sample_from_logits(s_logits, temperature, do_sample, top_k, top_p)

                    s_logits = self._kv_S.append_one(next_id)  # 前进一步，s_logits 更新为“下一步”的分布
                    out_tokens.append(next_id)
                    self.s_count += 1
                    self._update_spans(is_large=False, out_tokens_len=len(out_tokens))

                    # L 的追平策略：攒一段 tail，再批量 append（减少触发成本）
                    self._l_tail.append(next_id)
                    if self._catchup_stride > 0 and len(self._l_tail) >= self._catchup_stride:
                        if not l_prefill_done:
                            self._kv_L.prefill(input_ids)
                            l_prefill_done = True
                        self._kv_L.append_chunk(self._l_tail)
                        self._l_tail.clear()

                # 预算更新（仅预算模式）
                if self.budget_enable:
                    self._update_lambda(self.l_count, self._allowance(t))

                step += 1

        # 结尾日志输出
        self._log_collab_stats()
        self._log_gating_stats()
        if self.mark_large:
            return self._render_annotated(out_tokens)
        else:
            return self.tokenizer.decode(out_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    main()
