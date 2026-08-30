"""HuggingFace generation for F / P / N. Same checkpoint, think vs base template."""

from __future__ import annotations

from weighttraces.cot import prefix_ids
from weighttraces.dummy import ProbeItem, Sample

BASE_SUFFIX = "Solve step by step. Put the final numeric answer after ####."
MAX_F = 2048
MAX_ANSWER = 64


def _attn_implementation(torch) -> str:
    """Never request flash_attention_2. Ampere+ uses SDPA; older GPUs use eager."""
    if not torch.cuda.is_available():
        return "eager"
    major, _minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"cuda device={name} capability={major}.{_minor}", flush=True)
    if major < 8:
        torch.backends.cuda.enable_flash_sdp(False)
        return "eager"
    return "sdpa"


def user_content(item: ProbeItem, arm: str) -> str:
    if arm == "think":
        return item.question
    if arm == "base":
        return f"{item.question}\n\n{BASE_SUFFIX}"
    raise ValueError(arm)


class HFGenerator:
    def __init__(self, model_name: str, temperature: float = 0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        attn = _attn_implementation(torch)
        print(f"loading {model_name} attn={attn} dtype={dtype}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn,
        )
        if device_map is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._close_think = self._encode("</think>\n")
        self._open_close_think = self._encode("<think>\n</think>\n")
        self._hash = self._encode("#### ")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_ids(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        for sp in (self.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"):
            if sp:
                text = text.replace(sp, "")
        return text

    def prompt_f(self, item: ProbeItem, arm: str) -> str:
        messages = [{"role": "user", "content": user_content(item, arm)}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(arm == "think"),
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _prompt_ids(self, item: ProbeItem, arm: str) -> list[int]:
        return self.tokenizer(self.prompt_f(item, arm), add_special_tokens=False)["input_ids"]

    def _find_subseq(self, hay: list[int], needle: list[int]) -> int | None:
        if not needle or len(hay) < len(needle):
            return None
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i : i + n] == needle:
                return i
        return None

    def extract_cot_ids(self, gen_ids: list[int], arm: str) -> list[int]:
        if arm == "think":
            close = self._encode("</think>")
            open_ = self._encode("<think>")
            end = self._find_subseq(gen_ids, close)
            start = self._find_subseq(gen_ids, open_)
            if end is not None:
                if start is not None and start < end:
                    return gen_ids[start + len(open_) : end]
                return gen_ids[:end]
            if start is not None:
                return gen_ids[start + len(open_) :]
            return list(gen_ids)
        h = self._encode("####")
        idx = self._find_subseq(gen_ids, h)
        if idx is None:
            return list(gen_ids)
        return gen_ids[:idx]

    def _generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        seed: int,
        sample_idx: int,
        arm: str,
        cond: str,
    ) -> list[int]:
        torch = self.torch
        arm_i = 0 if arm == "base" else 1
        cond_i = {"F": 0, "P": 1, "N": 2}[cond]
        gen_seed = seed + 1009 * sample_idx + 17 * arm_i + 101 * cond_i
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, input_ids.shape[1] :].tolist()
        return [int(x) for x in gen]

    def sample_F(self, item: ProbeItem, arm: str, sample_idx: int, seed: int) -> Sample:
        prompt_ids = self._prompt_ids(item, arm)
        gen = self._generate(prompt_ids, MAX_F, seed, sample_idx, arm, "F")
        cot_ids = self.extract_cot_ids(gen, arm)
        return Sample(
            item_id=item.item_id,
            model=arm,
            condition="F",
            sample_idx=sample_idx,
            text=self.decode_ids(gen),
            n_cot_tokens=len(cot_ids),
            cot_text=self.decode_ids(cot_ids),
            cot_ids=cot_ids,
        )

    def _n_suffix_ids(self, item: ProbeItem, arm: str) -> list[int]:
        if arm != "think":
            return list(self._hash)
        prompt = self.prompt_f(item, arm)
        tail = prompt.rstrip()
        if tail.endswith("<think>") or prompt.endswith("<think>\n"):
            return list(self._close_think)
        return list(self._open_close_think)

    def sample_N(self, item: ProbeItem, arm: str, sample_idx: int, seed: int) -> Sample:
        prompt_ids = self._prompt_ids(item, arm)
        suffix = self._n_suffix_ids(item, arm)
        gen = self._generate(prompt_ids + suffix, MAX_ANSWER, seed, sample_idx, arm, "N")
        text = self.decode_ids(suffix + gen)
        return Sample(
            item_id=item.item_id,
            model=arm,
            condition="N",
            sample_idx=sample_idx,
            text=text,
            n_cot_tokens=0,
            cot_text="",
            cot_ids=[],
        )

    def sample_P(
        self, item: ProbeItem, arm: str, f_sample: Sample, sample_idx: int, seed: int
    ) -> Sample:
        cot_ids = list(f_sample.cot_ids)
        if not cot_ids and f_sample.cot_text:
            cot_ids = self._encode(f_sample.cot_text)
        half = prefix_ids(cot_ids)
        if not half:
            s = self.sample_N(item, arm, sample_idx, seed)
            return Sample(
                item_id=s.item_id,
                model=s.model,
                condition="P",
                sample_idx=s.sample_idx,
                text=s.text,
                n_cot_tokens=0,
                cot_text="",
                cot_ids=[],
            )
        prompt_ids = self._prompt_ids(item, arm)
        closer = self._close_think if arm == "think" else self._hash
        suffix = half + closer
        gen = self._generate(prompt_ids + suffix, MAX_ANSWER, seed, sample_idx, arm, "P")
        return Sample(
            item_id=item.item_id,
            model=arm,
            condition="P",
            sample_idx=sample_idx,
            text=self.decode_ids(suffix + gen),
            n_cot_tokens=len(half),
            cot_text=self.decode_ids(half),
            cot_ids=half,
        )
