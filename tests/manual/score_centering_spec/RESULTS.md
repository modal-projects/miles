# Score centering with DFlash

The results support enabling score centering with DFlash and DFlash2 through the `DFLASH` algorithm. This requires SGLang support for probability capture and server checks.

The model tests used `Qwen/Qwen3.8-27B` with `incoai/Qwen3.8-27B-DFlash2` on text prompts. These tests passed the logprob checks and short training runs. Separate sampling tests checked that both DFlash verification paths sample from the expected target distribution.

## What score centering needs

Score centering needs the target probabilities used during generation. Speculative decoding can supply these probabilities when the target model checks the draft tokens.

We checked the returned logprobs against those probabilities. This covers tokens accepted from the draft and tokens selected by the target model. We also checked that Miles assigns each logprob to the correct output token.

With top-k or top-p sampling, Miles needs every token that the sampling filter allows. The tests check that SGLang returns that complete set and its probabilities.

## Test results

| Check | Result |
| --- | --- |
| Returned logprobs | All 1,634 checked speculative output positions passed. The largest error against the target logprobs used during generation was `4.4e-6`, below the `2e-5` test limit. |
| Score-centering gradients | All 90 comparisons passed against an independent reference calculation. The checks cover all three importance-sampling modes. |
| Training with and without DFlash2 | Four runs passed: DFlash2 on/off, each with sampling filters on/off. Each run completed three optimizer steps with nonzero gradients. Updated target weights reached the rollout servers, and later rollouts remained valid. |

The probability tests cover top-32 and top-128 logprobs, top-k/top-p sampling, and temperatures 0.7, 1.0, and 1.3. The training runs use the same starting target weights and the same prompts. Both DFlash2 runs preserve draft weights that are not shared with the target.

## Open finding

We also recalculated next-token probabilities for the same input prefixes. Some results differed substantially from the probabilities recorded during generation. This also occurred without speculative decoding. The cause remains unresolved.

The capture tests show that Miles receives the target probabilities used during generation, within the test limits. They do not explain these differences or establish equal model probabilities with and without speculation. Three training steps also do not establish equal learning outcomes or long-run stability. See [the numerical comparisons](DETAILS.md#numerical-controls) for the measurements.

## Dependency and performance

SGLang must support the server checks and return the required probabilities. The tested fixes are in [the SGLang dependency PR](https://github.com/modal-projects/sglang/pull/69). The tested stock image did not contain all required fixes.

With logprob capture enabled in both cases, DFlash2's median generation throughput was 1.9–2.3 times the throughput without speculation. This short benchmark used eight concurrent requests on one H200. See [the timing results](DETAILS.md#throughput) for the workload and variation between runs.

[Detailed experiment record](DETAILS.md) · [Reproduction instructions](README.md)
