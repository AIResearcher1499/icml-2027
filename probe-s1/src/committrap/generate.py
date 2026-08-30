"""HF generation for F, answer probes, and reforks. CUDA or MPS."""

from __future__ import annotations

import time

from committrap.dummy import Event, ProbeItem
from committrap.entropy import entropy_from_logits

MAX_F = 2048
MAX_ANSWER = 64
MAX_REFORK = 512
PROBE_CLOSE = "</think>\nThe answer is "


def _attn_and_device(torch) -> tuple[str, str, object, str | None]:
    """Return device, attn, dtype, device_map. Never request flash_attention_2."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        print(f"cuda device={name} capability={major}.{minor}", flush=True)
        if major < 8:
            torch.backends.cuda.enable_flash_sdp(False)
            attn = "eager"
        else:
            attn = "sdpa"
        return "cuda", attn, torch.bfloat16, "auto"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps device=Apple GPU attn=eager dtype=float16", flush=True)
        return "mps", "eager", torch.float16, None
    print("cpu attn=eager dtype=float32", flush=True)
    return "cpu", "eager", torch.float32, None


class HFGenerator:
    def __init__(self, model_name: str, temperature: float = 0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.last_tok_s = 0.0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device, attn, dtype, device_map = _attn_and_device(torch)
        print(f"loading {model_name} device={device} attn={attn} dtype={dtype}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn,
        )
        if device_map is None:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._close_think = self._encode("</think>\n")
        self._probe_close = self._encode(PROBE_CLOSE)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_ids(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        for sp in (self.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"):
            if sp:
                text = text.replace(sp, "")
        return text

    def prompt_f(self, item: ProbeItem) -> str:
        messages = [{"role": "user", "content": item.question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _prompt_ids(self, item: ProbeItem) -> list[int]:
        return self.tokenizer(self.prompt_f(item), add_special_tokens=False)["input_ids"]

    def _find_subseq(self, hay: list[int], needle: list[int]) -> int | None:
        if not needle or len(hay) < len(needle):
            return None
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i : i + n] == needle:
                return i
        return None

    def extract_cot_ids(self, gen_ids: list[int]) -> list[int]:
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

    def _generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        seed: int,
        *,
        scores: bool = False,
    ) -> tuple[list[int], list[float] | None]:
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if scores:
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**kwargs)
        dt = time.perf_counter() - t0
        if scores:
            gen = out.sequences[0, input_ids.shape[1] :].tolist()
            ents = [entropy_from_logits(step[0].float().cpu().tolist()) for step in out.scores]
        else:
            gen = out[0, input_ids.shape[1] :].tolist()
            ents = None
        n = len(gen)
        rate = n / max(dt, 1e-6)
        self.last_tok_s = rate
        print(
            f"generate tokens={n} sec={dt:.2f} tok/s={rate:.1f} scores={int(scores)}",
            flush=True,
        )
        return [int(x) for x in gen], ents

    def sample_F(self, item: ProbeItem, sample_idx: int, seed: int) -> Event:
        prompt_ids = self._prompt_ids(item)
        gen_seed = seed + 1009 * sample_idx
        gen, _ = self._generate(prompt_ids, MAX_F, gen_seed)
        cot_ids = self.extract_cot_ids(gen)
        return Event(
            kind="F",
            item_id=item.item_id,
            sample_idx=sample_idx,
            text=self.decode_ids(gen),
            n_cot_tokens=len(cot_ids),
            cot_ids=cot_ids,
        )

    def probe_at(
        self, item: ProbeItem, f_event: Event, cut: int, sample_idx: int, seed: int
    ) -> Event:
        prompt_ids = self._prompt_ids(item)
        prefix = list(f_event.cot_ids[:cut])
        gen_seed = seed + 1009 * sample_idx + 17 * cut
        gen, ents = self._generate(
            prompt_ids + prefix + self._probe_close, MAX_ANSWER, gen_seed, scores=True
        )
        h = sum(ents) / len(ents) if ents else 0.0
        return Event(
            kind="probe",
            item_id=item.item_id,
            sample_idx=sample_idx,
            text=PROBE_CLOSE + self.decode_ids(gen),
            n_cot_tokens=f_event.n_cot_tokens,
            cut=cut,
            H=h,
        )

    def refork_at(
        self,
        item: ProbeItem,
        f_event: Event,
        cut: int,
        sample_idx: int,
        refork_idx: int,
        seed: int,
    ) -> Event:
        prompt_ids = self._prompt_ids(item)
        prefix = list(f_event.cot_ids[:cut])
        gen_seed = seed + 1009 * sample_idx + 101 * cut + 13 * refork_idx
        gen, _ = self._generate(prompt_ids + prefix, MAX_REFORK, gen_seed)
        return Event(
            kind="refork",
            item_id=item.item_id,
            sample_idx=sample_idx,
            text=self.decode_ids(prefix + gen),
            n_cot_tokens=f_event.n_cot_tokens,
            cut=cut,
            refork_idx=refork_idx,
        )
