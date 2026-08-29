"""Revisioned identities shared by loader, kernels, oracle and disk cache."""

KERNEL_ABI_VERSION = "glm53-metal-fp8-v3-tiled8-selected8-lut"
GROUPED_KERNEL_ABI = "glm53-grouped-fp8-v4-simdgroup-mma32"
GROUPED_MEASURED_CROSSOVER_ROUTES = 16
GROUPED_MIN_ROUTES = 256
PACKED_EXPERT_BANK_ABI = "glm53-packed-expert-bank-v1-gate-up-output-major"
PACKED_DECODE_KERNEL_ABI = "glm53-packed-selected8-fp8-v1"
NOPE_DSA_CACHE_ABI_DIRECT = (
    "glm53-nope-dsa-v1"
    "-kv-latent512"
    "-sentinel-minus1"
)
NOPE_DSA_CACHE_ABI_COMPACT = (
    "glm53-nope-dsa-v2"
    "-single-latent512"
    "-compact-indexpool-v2"
    "-kpool4-int64"
    "-rollback16-raw19"
    "-sentinel-minus1"
)
# Backward-compatible name used by the direct-cache probes and manifest.
NOPE_DSA_CACHE_ABI = NOPE_DSA_CACHE_ABI_DIRECT
MLX_VLM_REVISION = "e82d557d9f4b804cb1fc3eaaebc25488ba778a98"
CACHE_IDENTITY_SCHEMA = "glm53-hybrid-state-v1"
