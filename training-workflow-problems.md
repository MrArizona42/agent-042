# Training Workflow Review

## Scope

This document reviews the LoRA training workflow in this repository end to end:

- Hydra configuration and composition
- PyTorch Lightning integration
- QLoRA and model-loading setup
- MLflow tracking and artifact lineage
- Airflow orchestration and observability
- Notebook-based operator workflows

The goal is not just to list issues, but to identify where the current design fights the frameworks instead of using them idiomatically and reliably.

## Executive Summary

The current workflow is workable as a prototype, but it is not yet robust enough for repeated experimentation, concurrent runs, or clean promotion into a production-oriented registry flow.

The strongest parts are:

- A reasonably small training surface area
- A clear split between training, evaluation, and registry operations
- A sensible choice of Hydra plus Lightning plus MLflow as the core stack

The main weaknesses are:

- Training correctness risks in the data path
- Weak experiment lineage and artifact semantics
- A non-idiomatic Hydra layer that does not actually compose objects
- Drift between Airflow, notebooks, and the canonical Python runtime
- An incomplete and risky QLoRA setup

If I were restructuring this, I would keep Hydra as the orchestration layer, make the runtime object graph instantiable from config, fix the data and checkpoint correctness issues first, then wire training into evaluation so the workflow becomes:

`train -> evaluate -> inspect/promote`

## Findings

### High Severity

#### 1. Long-source samples can end up with almost no supervised target tokens

In `experiments/training/train_adapter/data_module.py:46-58`, the code tokenizes the prompt and target independently, concatenates them, then truncates the combined sequence to `max_len`. Later, `experiments/training/train_adapter/data_module.py:101` masks the prompt tokens from the loss.

Combined with the current prompt and sequence budget in `experiments/training/conf/experiment/train_adapter.yaml:21-25`, this means a long article can consume nearly the entire 768-token budget before the abstract is even added. In that case, the model receives little or no supervised signal for the target text.

This is the single biggest training-quality issue in the current workflow.

#### 2. The deployable artifact is not necessarily the best checkpoint

`experiments/training/train_adapter/pipeline.py:56-59` saves the best checkpoints according to `val_loss`, but `experiments/training/train_adapter/pipeline.py:82-94` exports the final in-memory adapter after training finishes and logs that to MLflow.

That means once training lasts longer than a trivial one-epoch run, the artifact that gets registered or inspected may differ from the checkpoint the callback considered best.

This breaks model selection semantics and makes downstream registration less trustworthy.

#### 3. Checkpointing is not run-scoped

`experiments/training/train_adapter/pipeline.py:52-74` writes checkpoints for every run into the same shared `artifacts/training/checkpoints` directory and uses a shared `default_root_dir`.

That creates several problems:

- concurrent runs can collide
- sweep outputs can mix together
- `last.ckpt` is ambiguous
- resume semantics are unsafe
- provenance is harder to reconstruct after the fact

For an experimentation platform, checkpoint directories must be run-scoped.

#### 4. The 4-bit QLoRA setup is incomplete and mixes device-management models

`experiments/training/train_adapter/modeling.py:39-70` loads a quantized model and enables gradient checkpointing, but the setup is missing pieces that are normally part of a stable k-bit finetuning path:

- no `prepare_model_for_kbit_training`
- no explicit `use_cache = False`
- `device_map: "auto"` from `experiments/training/conf/experiment/train_adapter.yaml:5-12`
- Lightning simultaneously managing accelerator and devices in `experiments/training/train_adapter/pipeline.py:61-80`

That combination is not a clean separation of responsibilities. I would not trust it long term for stable Lightning-based QLoRA runs.

### Medium Severity

#### 5. The Hydra layer is brittle, and some examples are wrong for the current config shape

The actual runtime config is nested under `experiment.*`, but some examples still use top-level override paths.

Examples:

- `experiments/training/train_adapter/start_train.py:9-11` loads a config whose training fields live under `experiment`
- `dags/train_lora.py:105` documents overrides like `trainer.max_epochs=3`
- `experiments/training/lora_training.ipynb:102-105` uses the same incorrect pattern

Those examples should use keys like:

- `experiment.trainer.max_epochs=3`
- `experiment.data.batch_size=...`
- `experiment.training.lr=...`

This is a sign that the config schema is not communicated cleanly to users.

#### 6. Hydra is being used mostly as a typed parameter bag, not as a composition system

`experiments/training/train_adapter/config.py:1-105` gives you structured access after merge, which is good, but the runtime object graph is still hard-coded in `experiments/training/train_adapter/pipeline.py:37-80`.

As a result:

- trainer construction is hard-coded in Python
- callback wiring is hard-coded in Python
- logger creation is hard-coded in Python
- notebooks have to manually rebuild large parts of the runtime instead of instantiating them from config

So the project gets some benefits of Hydra validation, but not the main compositional benefits Hydra is designed to provide.

#### 7. Notebook workflows drift from the canonical runtime

The notebook path is intended to mirror the Python or Airflow path, but it does not actually do so.

Examples:

- `experiments/training/lora_training.ipynb:248-286` rebuilds a trainer inline
- it does not attach the MLflow logger even though the notebook claims to mirror the DAG path
- `experiments/training/lora_ops.ipynb:57-98` is still mostly commented snippets rather than a real operator notebook
- `experiments/training/lora_ops.ipynb:57` searches for `"lora-training"`, while the active training experiment name is `"train_adapter"` in `experiments/training/conf/experiment/train_adapter.yaml:64-65`

This creates drift between the documented workflow and the actual runtime behavior.

#### 8. MLflow logging is shallower and more misleading than the config suggests

The config in `experiments/training/conf/experiment/train_adapter.yaml:61-65` exposes `log_artifacts`, `log_metrics`, and `log_params`, but the code does not honor those flags consistently.

Examples:

- only `log_params` is explicitly checked in `experiments/training/train_adapter/mlflow_utils.py:55-65`
- Hydra artifacts are always uploaded in `experiments/training/train_adapter/mlflow_utils.py:70-79`
- model artifacts are always uploaded in `experiments/training/train_adapter/pipeline.py:91-96`
- `run_name` is fixed to `train_adapter`
- no useful tags are added in `experiments/training/train_adapter/mlflow_utils.py:46-51`

There is also a sequencing issue: Hydra artifacts are uploaded at the start of the run in `experiments/training/train_adapter/pipeline.py:28-29`, so later runtime files written into the Hydra output directory may never make it into MLflow.

#### 9. Airflow observability is weak

`dags/train_lora.py:65-71` buffers subprocess output and prints it only after completion. At the same time, `check=True` is enabled.

Operationally that means:

- no live progress in Airflow logs
- harder debugging for long runs
- on failures, subprocess context is not streamed naturally into task logs

For long-running GPU jobs, streaming logs matter.

#### 10. The precision stack is internally contradictory

The config currently mixes:

- fp16 base model dtype in `experiments/training/conf/experiment/train_adapter.yaml:4-9`
- fp32 4-bit compute dtype in the same block
- Lightning precision `"32-true"` in `experiments/training/conf/experiment/train_adapter.yaml:51`

The inline comment says fp16 mixed precision is intended, but the configured precision says otherwise. This makes it unclear what precision regime the run is actually expected to use.

#### 11. The workflow stops too early

`dags/train_lora.py:8-12` explicitly ends at a `run_id` and pushes inspection, registration, promotion, and sync to notebooks.

Keeping promotion manual is reasonable. Stopping before evaluation is not.

For a repository that already contains evaluation DAGs, I would expect the default training lifecycle to be:

`train -> evaluate -> human decision`

rather than:

`train -> human inspection only`

### Low to Medium Severity

#### 12. There are several smaller signs of immaturity in the runtime

Examples:

- `experiments/training/train_adapter/lit_module.py:18-24` stores `mlflow_cfg` but never uses it
- `experiments/training/train_adapter/lit_module.py:61-68` computes `tokens_per_second` in a way that can include non-training pauses
- `experiments/training/train_adapter/lit_module.py:79-80` optimizes over `self.parameters()` instead of explicitly filtering trainable parameters
- `experiments/training/conf/paths/paths_config.yaml:1` contains a personal machine path as the fallback default
- there are no dedicated tests for the training package under `tests/`

None of these alone is catastrophic, but together they indicate the workflow is still at a prototype maturity level.

## Recommended Direction

### Keep Hydra as the top-level orchestration layer

I would not switch to LightningCLI here.

This repository already uses Hydra-style composition, and introducing LightningCLI would create two competing control planes for configuration.

### Move to a real structured training config

I would define a proper `TrainConfig` in Hydra `ConfigStore` and split the current overloaded `experiment.*` namespace into clearer sections such as:

- `data`
- `model`
- `peft`
- `module`
- `trainer`
- `callbacks`
- `logger`
- `tracking`
- `artifacts`

That would make the config easier to reason about and easier to override safely.

### Use `hydra.utils.instantiate` where it actually helps

Hydra should instantiate the composable, framework-facing parts of the runtime:

- `ArxivDataModule`
- `MLFlowLogger`
- `ModelCheckpoint`
- `EarlyStopping`
- `Trainer`

I would keep explicit Python factories for the domain-specific parts that benefit from custom validation and setup logic:

- tokenizer loading
- base model loading
- PEFT or QLoRA wrapping

That gives you Hydra composition without turning model construction into opaque YAML.

### Redesign the data contract before anything else

The highest-value change is to stop treating prompt and target as a single undifferentiated sequence budget.

I would add fields like:

- `source_max_length`
- `target_max_length`
- `train_on_inputs: false`

I would also log or assert on a metric such as the count or fraction of samples that produce zero target supervision.

For summarization finetuning, this matters more than any logging cleanup.

### Make artifacts run-scoped and semantically explicit

Each run should own a unique artifact root with clear subdirectories:

- `hydra/`
- `checkpoints/`
- `export/`

Then either:

- reload `checkpoint_callback.best_model_path` before export, or
- replace generic checkpoint export with an adapter-aware export callback

The important part is that the MLflow artifact must correspond to the model you actually intend to compare, register, and promote.

### Make MLflow first-class

MLflow should capture enough lineage to make runs auditable and comparable.

I would log at least:

- flattened resolved config
- git SHA
- base model id or local path
- dataset revision or DVC hash
- effective batch size
- trainable parameter count
- Airflow DAG run id
- hardware info

A fixed `run_name="train_adapter"` is not sufficient once the project accumulates real experimentation history.

### Make evaluation part of the default training lifecycle

The workflow should become:

`train -> evaluate -> inspect/promote`

Promotion can stay manual. Evaluation should not.

### Reduce notebook duplication

I would either:

- retire `experiments/training/lora_training.ipynb` as a separate training path, or
- reduce it to a thin notebook that calls the same Python runtime used by Airflow

And `experiments/training/lora_ops.ipynb` should become a real executable operations notebook rather than commented pseudocode.

## Implementation Plan

This plan is intentionally split into three phases so the work can be implemented and reviewed one step at a time.

### Step 1: Fix Training Correctness and Artifact Semantics

This step addresses the issues that directly affect model quality, experiment validity, and artifact trustworthiness.

#### Goals

- Ensure every training sample preserves meaningful target supervision
- Make the QLoRA path internally consistent and safer
- Make checkpoint and export semantics correct
- Make artifacts run-scoped and reproducible

#### Changes

1. Redesign sequence construction in the data module.
	- Add separate config fields for source and target token budgets.
	- Truncate source and target independently.
	- Keep loss masking behavior explicit and configurable.
	- Log the number or ratio of samples with zero target tokens.

2. Clean up the k-bit training path.
	- Add `prepare_model_for_kbit_training` where appropriate.
	- Disable `use_cache` for training.
	- Reconcile `device_map` behavior with Lightning device management.
	- Make precision settings internally consistent.

3. Make checkpointing run-scoped.
	- Create a unique per-run artifact directory.
	- Put checkpoints under that run directory.
	- Ensure sweeps and concurrent runs do not collide.

4. Export the right artifact.
	- Either reload the best checkpoint before export, or
	- implement an adapter export callback bound to the best checkpoint semantics.

5. Tighten runtime correctness details.
	- Remove unused config fields from the Lightning module.
	- Filter optimizer parameters to trainable ones.
	- Improve throughput metric calculation.

#### Deliverables

- Corrected data pipeline
- Safer QLoRA setup
- Run-scoped artifact layout
- Best-model-consistent export behavior

#### Exit Criteria

- No training sample silently loses all target supervision without being visible in metrics
- Exported adapter corresponds to the intended checkpoint
- Concurrent runs no longer share checkpoint directories

### Step 2: Restructure Hydra and Unify Runtime Construction

This step addresses design quality and removes the current mismatch between config, notebooks, and runtime wiring.

#### Goals

- Make Hydra the real composition system instead of just a typed parameter container
- Centralize trainer, callback, and logger construction
- Eliminate config-path confusion and notebook drift

#### Changes

1. Introduce structured config registration through Hydra `ConfigStore`.
	- Define a canonical training config schema.
	- Split `experiment.*` into clearer subdomains.

2. Move composable runtime parts to `_target_`-based config.
	- Data module
	- MLflow logger
	- checkpoint callbacks
	- optional early stopping
	- trainer

3. Keep explicit Python factories where needed.
	- tokenizer construction
	- model loading
	- PEFT or QLoRA wrapping

4. Fix override ergonomics and documentation.
	- Align DAG examples with the actual config schema.
	- Align notebook override examples with the same schema.
	- Remove stale or misleading config examples.

5. Reduce notebook duplication.
	- Make the training notebook call the canonical Python runtime instead of reconstructing it.
	- Make the ops notebook executable rather than commented pseudocode.

#### Deliverables

- Structured Hydra config registered in `ConfigStore`
- Runtime object graph instantiated from config where appropriate
- Correct override examples in docs, DAGs, and notebooks
- Reduced notebook-runtime drift

#### Exit Criteria

- Trainer, callbacks, and logger no longer need to be manually rebuilt in notebooks
- Hydra overrides are consistent across CLI, Airflow, and notebooks
- Runtime composition is easier to extend without editing core pipeline code for every change

### Step 3: Make Tracking, Evaluation, and Operations First-Class

This step turns the workflow from a training script into a more credible experimentation pipeline.

#### Goals

- Improve experiment lineage and observability
- Integrate evaluation into the default lifecycle
- Make the operator workflow usable without hidden tribal knowledge

#### Changes

1. Improve MLflow lineage.
	- Log flattened resolved config.
	- Log git SHA.
	- Log dataset revision or DVC hash.
	- Log hardware info and effective batch size.
	- Use meaningful tags and non-static run naming.
	- Honor tracking flags consistently.

2. Improve Airflow observability.
	- Stream subprocess logs instead of buffering everything.
	- Preserve training context clearly on failure.
	- Include run metadata in task logs.

3. Add a post-train evaluation stage.
	- Wire training output into the existing evaluation framework.
	- Record evaluation results against the same run lineage.
	- Make human promotion decisions based on both training and evaluation outputs.

4. Harden the operator-facing notebook path.
	- Turn `lora_ops.ipynb` into an actual operational notebook.
	- Make run inspection, registration, promotion, and sync concrete and executable.

5. Add targeted tests.
	- data-sequence construction
	- config loading and overrides
	- artifact path generation
	- best-checkpoint export semantics
	- post-train metadata logging

#### Deliverables

- Improved MLflow provenance
- Better Airflow debugging experience
- Training linked to evaluation by default
- More usable operator notebook flow
- Initial training-focused tests

#### Exit Criteria

- A training run produces enough metadata to be audited later
- Evaluation is part of the normal pipeline rather than an optional afterthought
- Operators can inspect and promote runs through a real documented workflow

## Suggested Execution Order

When implementing this plan, I would do the steps in exactly this order:

1. Step 1 first, because correctness and artifact trust are more important than architecture cleanup.
2. Step 2 second, because once correctness is fixed, the runtime can be refactored into a cleaner Hydra-based composition model.
3. Step 3 last, because tracking and orchestration improvements are most valuable once the underlying training semantics are sound.

That sequence minimizes rework and keeps each phase independently reviewable.
