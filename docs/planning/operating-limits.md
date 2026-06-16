# Operating Limits

Operating-limit experiments define how the platform behaves under stress. They
come after the initial inference and RAG baselines because load behavior depends
on the selected model, RAG pipeline, reranker, and token budgets.

## Design Questions

- How many concurrent requests can the single-node deployment handle?
- What does degradation look like when GPU, queue, Redis, or vLLM becomes the
  bottleneck?
- Which fallbacks preserve useful behavior without hiding failures?
- What limits should be exposed as operator configuration?

## Scenarios

- short non-RAG chat;
- RAG chat with baseline retrieval;
- RAG chat with reranking enabled;
- long-context request near budget limit;
- cold LoRA adapter request;
- mixed workload with chat, RAG, and adapter traffic.

## Load Grid

- 1 concurrent request;
- 2 concurrent requests;
- 4 concurrent requests;
- 8 concurrent requests;
- saturation run until rejection, timeout, or unstable latency.

## Metrics

- p50/p95/p99 time to first token;
- p50/p95/p99 full response latency;
- Celery queue depth;
- Redis stream stability;
- vLLM token throughput;
- GPU memory;
- GPU utilization;
- system RAM;
- error rate;
- timeout rate;
- request rejection rate.

## Degradation Policies To Decide

- queue length limit;
- timeout values;
- backpressure behavior;
- rate-limit behavior;
- whether reranking can be disabled under load;
- whether RAG can return a no-context fallback under partial failure;
- whether cold adapter loading is allowed during high load;
- operator alerts and dashboard thresholds.

## Result Table

| Scenario | Safe concurrency | p95 TTFT | p95 full latency | Error rate | Limit |
| --- | ---: | ---: | ---: | ---: | --- |
| short chat | TBD | TBD | TBD | TBD | TBD |
| RAG baseline | TBD | TBD | TBD | TBD | TBD |
| RAG + reranker | TBD | TBD | TBD | TBD | TBD |
| cold adapter | TBD | TBD | TBD | TBD | TBD |
