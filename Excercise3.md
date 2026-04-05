# Two KV-Cache Strategies: Grouped Query Attention (GQA) vs. KV-Cache Quantization
## Strategy 1: Grouped Query Attention (GQA)

**Core idea:** Instead of assigning each query head its own dedicated key-value head (as in standard multi-head attention), GQA groups multiple query heads to share a single KV head. For example, a model with 32 query heads and 8 KV heads stores only one-quarter of the KV cache compared to full MHA, while each group's queries still compute attention independently against their shared keys and values.

**Tradeoff:** GQA reduces the diversity of attention patterns the model can express, since grouped query heads are forced to attend over identical key-value representations. In practice, this quality loss is negligible on standard benchmarks. However, it is an architectural decision — it must be chosen at training time and cannot be applied post-hoc to existing MHA models without uptraining.

**Production context:** Choose GQA when designing or selecting a model for high-throughput serving, especially under tensor parallelism where each GPU handles complete KV head groups. Nearly all major production models (Llama 3, Mistral, Gemma) now default to GQA.

## Strategy 2: KV-Cache Quantization

**Core idea:** Store cached key and value tensors in reduced precision (e.g., FP8 or INT4) instead of FP16/BF16. This directly halves (FP8) or quarters (INT4) the per-token memory footprint of the cache without modifying the model architecture or evicting any tokens.

**Tradeoff:** Quantization introduces numerical noise uniformly across all cached representations. At FP8, accuracy degradation is typically below 1%. At 4-bit, mathematical reasoning and precise factual recall can degrade measurably, requiring task-specific validation before deployment.

**Production context:** FP8 KV-cache quantization is the safest first optimization for any deployment — it requires no retraining, composes multiplicatively with GQA, and is natively supported by major serving frameworks like vLLM and TensorRT-LLM.