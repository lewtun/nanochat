"""Configuration for the RH fork of nanochat (requires transformers 5.24)."""
from huggingface_hub.dataclasses import strict
from transformers.models.nanochat.configuration_nanochat import NanoChatConfig


@strict
class RHNanoChatConfig(NanoChatConfig):
    model_type = "rh_nanochat"
    window_pattern: str = "SSSL"
    tokenizer_sha256: str | None = None
    checkpoint_step: int | None = None
    rms_norm_eps: float | None = None
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}
        super().__post_init__(**kwargs)
        if not self.window_pattern or any(c not in 'SL' for c in self.window_pattern.upper()):
            raise ValueError('window_pattern must be a non-empty string of S and L')
        if self.rms_norm_eps is not None or self.final_logit_softcapping != 15.0:
            raise ValueError('RH nanochat requires dtype-default RMSNorm epsilon and softcap 15')
        if self.intermediate_size != 4 * self.hidden_size:
            raise ValueError('RH nanochat requires a 4x intermediate size')
        if self.rope_parameters.get('rope_theta') != 100000.0 or self.rope_parameters.get('rope_type') != 'default':
            raise ValueError('RH nanochat requires default RoPE with theta 100000')
