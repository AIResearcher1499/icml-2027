"""Generation backends: dummy (tests) and HuggingFace (GPU)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from accesstrap.data import ProbeItem
from accesstrap.entropy import entropy_from_logits
from accesstrap.prompts import MATH_SYSTEM, QA_SYSTEM, math_user, qa_user


CONDITIONS = ("A", "B", "C")


@dataclass
class Sample:
    item_id: str
    condition: str
    sample_idx: int
    text: str
    tokens: list[str]
    entropies: list[float]

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "condition": self.condition,
            "sample_idx": self.sample_idx,
            "text": self.text,
            "tokens": self.tokens,
            "entropies": self.entropies,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> Sample:
        return cls(
            item_id=rec["item_id"],
            condition=rec["condition"],
            sample_idx=int(rec["sample_idx"]),
            text=rec["text"],
            tokens=list(rec["tokens"]),
            entropies=[float(x) for x in rec["entropies"]],
        )


def access_block(item: ProbeItem, condition: str) -> str | None:
    if condition == "A":
        return None
    if condition == "B":
        return item.gold_access
    if condition == "C":
        return item.distractor_access
    raise ValueError(condition)


def build_messages(item: ProbeItem, condition: str) -> list[dict[str, str]]:
    block = access_block(item, condition)
    if item.split == "math":
        system, user = MATH_SYSTEM, math_user(item.question, block)
    else:
        system, user = QA_SYSTEM, qa_user(item.question, block)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def dummy_sample(item: ProbeItem, condition: str, sample_idx: int) -> Sample:
    """Deterministic fake traces with known correctness and entropy pattern.

    Seeded so tests can assert Venn/entropy directions without a GPU.
    """
    h = int(hashlib.sha256(f"{item.item_id}:{condition}:{sample_idx}".encode()).hexdigest(), 16)
    # Math dummy-0 gold is 72. Make A often correct, B less often, C rarely.
    correct_mod = {"A": 2, "B": 3, "C": 5}[condition]
    if item.split == "math":
        if h % correct_mod == 0:
            text = f"Let's add them. Therefore the total is #### {item.gold}"
        else:
            text = f"Wait, I will guess. Therefore the total is #### {int(item.gold) + 1}"
    else:
        if h % correct_mod == 0:
            text = f"Because of the evidence, #### {item.gold}"
        else:
            text = "However that is unclear. #### Unknown"
    # Fake token stream with connectives. Gold (B) gets lower entropy on 'therefore'.
    tokens = ["Wait", ",", "therefore", "the", "answer"]
    base = {"A": 1.2, "B": 0.6, "C": 1.4}[condition]
    ents = [0.3, 0.1, base + (h % 7) * 0.01, 0.2, 0.2]
    return Sample(item.item_id, condition, sample_idx, text, tokens, ents)


def _attn_implementation(torch) -> str:
    """Never request flash_attention_2. It hard-fails on pre-Ampere (Turing T4/2080/RTX 6000).

    Ampere+ (A6000 is sm_86) uses PyTorch SDPA. Older GPUs use eager.
    """
    if not torch.cuda.is_available():
        return "eager"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"cuda device={name} capability={major}.{minor}", flush=True)
    if major < 8:
        torch.backends.cuda.enable_flash_sdp(False)
        return "eager"
    return "sdpa"


class HFGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        attn_implementation = _attn_implementation(torch)
        print(f"loading {model_name} attn={attn_implementation} dtype={dtype}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        if device_map is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def sample(self, item: ProbeItem, condition: str, sample_idx: int, seed: int) -> Sample:
        torch = self.torch
        messages = build_messages(item, condition)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_seed = seed + 1009 * sample_idx + 17 * (ord(condition) - 64)
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        tokens: list[str] = []
        ents: list[float] = []
        for step, step_scores in enumerate(out.scores):
            logits = step_scores[0].float().cpu().tolist()
            ents.append(entropy_from_logits(logits))
            tok_id = int(gen_ids[step])
            tokens.append(self.tokenizer.decode([tok_id], skip_special_tokens=False))
        return Sample(item.item_id, condition, sample_idx, text, tokens, ents)
