import json
import re
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Kaggle GPU pools are heterogeneous (T4, P100, and occasionally older
# cards). bitsandbytes 4-bit kernels are unreliable or unsupported below
# compute capability 7.0 (pre-Turing, e.g. Tesla P100 = 6.0), where loading
# can raise or silently produce garbage generations. 4-bit is only
# requested on hardware known to support it well, and the load itself is
# wrapped so a bitsandbytes failure degrades to an fp16 load instead of
# crashing the whole run.
_MIN_4BIT_COMPUTE_CAPABILITY = (7, 0)


def _gpu_supports_4bit():
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(0) >= _MIN_4BIT_COMPUTE_CAPABILITY
    except Exception:
        return False


class QwenModel:
    def __init__(self, name, load_in_4bit=True, max_new_tokens=384):
        self.max_new_tokens = max_new_tokens
        use_4bit = load_in_4bit and _gpu_supports_4bit()
        if load_in_4bit and torch.cuda.is_available() and not use_4bit:
            warnings.warn(
                "GPU compute capability is below 7.0 (e.g. Tesla P100); "
                "disabling 4-bit quantization and loading in fp16 instead."
            )
        quant = None
        if use_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                torch_dtype="auto",
                quantization_config=quant,
            )
        except Exception as exc:
            if quant is None:
                raise
            warnings.warn(
                f"4-bit load failed ({exc}); retrying in fp16 without quantization."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                torch_dtype="auto",
                quantization_config=None,
            )
        self.model.eval()

    @torch.inference_mode()
    def chat(self, system, user, temperature=0.0):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.model.device)
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

def parse_json_object(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found: {text[:300]}")
    return json.loads(match.group(0))
