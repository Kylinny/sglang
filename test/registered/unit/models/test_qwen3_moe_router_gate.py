"""Unit tests for the Qwen3-MoE router gate quantization decision — CPU-only.

Regression test for sglang#49185: a compressed-tensors checkpoint may store
the MoE router (`mlp.gate`) quantized (e.g. RedHatAI's W4A16 Qwen3-30B-A3B
checkpoint, whose `ignore` lists only `lm_head`). The router must then be
built with the quant config, so the checkpoint's per-layer `ignore` list
decides for each layer; building it unquantized leaves the stored packed
weights with no parameter to load into and the router is silently never
written (all-NaN router logits, garbage output, no error). Checkpoints that
keep the router in BF16 and list it in `ignore` must stay unquantized.
"""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.models.qwen3_moe import _get_gate_quant_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROUTER_LAYER = "model.layers.0.mlp.gate"

# Mirrors RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16: W4A16
# group-quantized int4 weights for every Linear, only `lm_head` ignored.
_WNA16_WEIGHTS = {
    "actorder": "static",
    "block_structure": None,
    "dynamic": False,
    "group_size": 128,
    "num_bits": 4,
    "observer": "mse",
    "observer_kwargs": {},
    "strategy": "group",
    "symmetric": True,
    "type": "int",
}


def _wna16_config(ignore):
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": dict(_WNA16_WEIGHTS),
                }
            },
        }
    )


# Patch the CUDA capability check: there is no GPU on CPU CI.
_CHECK_SUPPORTED = (
    "sglang.srt.layers.quantization.compressed_tensors."
    "compressed_tensors.CompressedTensorsConfig._check_scheme_supported"
)


class TestGetGateQuantConfig(CustomTestCase):
    def test_no_quant_config_returns_none(self):
        self.assertIsNone(_get_gate_quant_config(None))

    def test_modelslim_config_is_passed_through(self):
        config = ModelSlimConfig()
        self.assertIs(_get_gate_quant_config(config), config)

    def test_compressed_tensors_config_is_passed_through(self):
        config = _wna16_config(ignore=["lm_head"])
        self.assertIs(_get_gate_quant_config(config), config)

    def test_fp8_config_stays_bf16(self):
        # The online Fp8/mxfp8 path keeps the gate in bf16 (unchanged).
        self.assertIsNone(_get_gate_quant_config(Fp8Config()))
        self.assertIsNone(_get_gate_quant_config(Fp8Config(use_mxfp8=True)))


class TestRouterGateScheme(CustomTestCase):
    def test_quantized_router_checkpoint_resolves_wna16_scheme(self):
        """RedHatAI-style config: router not ignored -> quantized linear."""
        config = _wna16_config(ignore=["lm_head"])
        layer = torch.nn.Linear(2048, 128, bias=False)
        with patch(_CHECK_SUPPORTED, return_value=True):
            scheme = config.get_linear_scheme(layer, ROUTER_LAYER)
        self.assertIsInstance(scheme, CompressedTensorsWNA16)

    def test_ignored_router_checkpoint_stays_unquantized(self):
        """cyankiwi-style config: router in `ignore` -> no scheme, so the
        gate resolves to UnquantizedLinearMethod (BF16, unchanged)."""
        config = _wna16_config(ignore=[ROUTER_LAYER, "lm_head"])
        layer = torch.nn.Linear(2048, 128, bias=False)
        with patch(_CHECK_SUPPORTED, return_value=True):
            scheme = config.get_linear_scheme(layer, ROUTER_LAYER)
        self.assertIsNone(scheme)


if __name__ == "__main__":
    unittest.main()
