"""Configuration for the RH fork of nanochat (requires transformers 5.24)."""
from huggingface_hub.dataclasses import strict
from transformers.models.nanochat.configuration_nanochat import NanoChatConfig


@strict
class RHNanoChatConfig(NanoChatConfig):
    model_type = "rh_nanochat"
    window_pattern: str = "SSSL"
