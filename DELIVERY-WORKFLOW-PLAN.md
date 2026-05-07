# Delivery Workflow Plan

## Context

Single ML engineer maintains the infrastructure. A team of researchers uses the deployed agent.
The project should behave like a production system on one node today, while preserving a clean
path toward a larger deployment later.

This document describes the final chosen delivery model only. It does not include temporary mount
schemes or transitional deployment layouts that would later be undone.

---

## Summary Of Decisions

1. The logical project layout stays as it is today. Application-facing paths such as
   `assets/rag_data/<dataset>`, `assets/datasets`, `assets/models`, `assets/adapters`, and
   `artifacts/training` remain the canonical paths used by code.
2. The server uses a release layout, not a long-lived live checkout. The active runtime is a
   Git-free release under `releases/<sha>` with `current` pointing to it.
3. Persistent local state lives outside the release tree under `/home/anton-m/agent-042`:
   - `.env`
   - `.dvc/config.local`
   - `assets/`
   - `artifacts/`
4. Whole `assets/datasets` and whole `assets/rag_data` are valid shared domain mounts in the final
   design. There is no planned intermediate phase where `rag_data` is mounted per dataset.
5. Airflow keeps ownership of RAG data refresh and DVC versioning, but Git write-back happens in a
   temporary clone on a bot branch, not in the active runtime tree.
6. The Airflow ownership problem is fixed by host-managed shared permissions on `/home/anton-m/agent-042`
   (setgid directories plus ACLs for the operator and the effective Airflow and Jupyter container
   UIDs) and targeted bind mounts, not by root `chown` during startup.
7. CI runs on GitHub-hosted runners. The server is for deployment and smoke verification, not for
   the main test suite.
8. The shared-storage cutover is atomic: existing repo-root data is migrated into `/home/anton-m/agent-042`
   shared roots in the same rollout that switches Compose mounts and moves Airflow DVC writes into
   the temp-clone flow.
9. Airflow bot push and PR update use a repo-scoped fine-grained GitHub token stored outside Git in
   the deployment env file.

---

## 1. Problems To Solve

The current setup has five concrete friction points.

1. **No CI gate.** Code is pushed and manually deployed. Broken changes can reach the server
   without automated validation.
2. **Build-from-source on the server.** `docker compose up --build` on the node makes deployments
   slow, less reproducible, and dependent on local build toolchain state.
3. **Airflow ownership corruption.** `airflow-prepare-dirs` currently fixes write access by
   recursively changing ownership on bind-mounted project paths.
4. **Runtime and Git are mixed together.** The same tree acts as deployment target, mutable Airflow
   workspace, and Git workspace.
5. **DVC refresh stops halfway.** Airflow already does `dvc add` and `dvc push`, so it creates
   real pointer-file changes that need a Git path to become durable source state.

---

## 2. Final Operating Model

Single node. Single running stack. One active deployed release at a time.

Branch roles:

- `main` — production branch
- `develop` — integration branch
- `feature/*` — short-lived development branches
- `data-sync/*` — bot-managed branches for Airflow-created `.dvc` updates

Every push to GitHub is validated automatically. Feature-branch images are built only when an
exact feature commit needs server-side testing. Deploys are always manual and commit-specific.

The active runtime is always a disposable release tree:

```text
/home/anton-m/agent-042/releases/<sha>/
/home/anton-m/agent-042/current -> /home/anton-m/agent-042/releases/<sha>/
```

Deploying means materializing a fresh release from one exact commit without `.git`, pointing
`current` at it, and running Compose from that release.

Code delivery flow:

```text
developer laptop        GitHub Actions                server
─────────────────       ───────────────────────       ─────────────────────────────
git push (any branch)
               →    ci.yml                        (validation on every push)
                  build-images.yml            (automatic on push to develop/main after merge,
                                                          manual for feature SHAs)
                            →    ghcr.io registry
                  deploy.yml (manual)       →  stream exact Git archive to fresh release dir
                                                          →  docker compose pull + up
```

Data-sync flow:

```text
Airflow DAG
   → refresh shared payload under assets/rag_data/<dataset>
   → create temp clone
   → restore .dvc/config.local into temp clone
   → stage shared payload into the clone under the standard repo path as a real directory
     (hardlinking files when possible, regular copy otherwise)
   → dvc add + dvc push
   → commit updated .dvc files and matching .gitignore changes
   → push bot branch (for example data-sync/develop)
   → open or update PR into develop
   → delete temp clone
```

The active runtime tree never becomes the Git workspace for data sync.

---

## 3. Server Filesystem Contract

This is the authoritative server layout.

```text
/home/anton-m/agent-042/
  .env
  .dvc/
    config.local
  assets/
    models/
    adapters/
    datasets/
    rag_data/
  artifacts/
    training/
  releases/
    <sha>/
   current -> /home/anton-m/agent-042/releases/<sha>/
```

### 3.1 Release tree

Properties:

- created from one exact Git commit
- materialized without `.git`; if a clone-based fallback is ever used, `.git` must still be
   removed immediately
- safe to delete and recreate on every deploy
- contains Git-tracked source, manifests, DAGs, notebooks, docs, tracked `.dvc` pointers, and
  tracked `.dvc/config`

### 3.2 Preserved local config and secrets

`/home/anton-m/agent-042/.env`

- canonical deployment env file
- passed directly to Compose via `--env-file`
- not copied into each release
- must set `PROJECT_ROOT=/home/anton-m/agent-042/current`
- stores stable deploy-time config such as `GITHUB_REPOSITORY` and
   `GITHUB_DATA_SYNC_TOKEN`

`/home/anton-m/agent-042/.dvc/config.local`

- canonical local DVC credential file
- not committed to Git
- made available to Airflow by bind mount
- host permissions should grant read access to the Airflow worker UID via ACL; the runtime does
   not need broad write access to this file
- copied only into the temporary clone when a DVC or Git write-back job runs

### 3.3 Preserved shared project data

`/home/anton-m/agent-042/assets` and `/home/anton-m/agent-042/artifacts` are the persistent shared roots.

Properties:

- created by the operator, notebooks, Airflow jobs, or model sync jobs
- survive release replacement
- not committed to Git
- mounted back into the runtime at the same logical paths the application expects
- `assets/datasets` is populated only by explicit operator or workflow actions such as
   `dvc pull` or prefetch jobs; deploy automation mounts it but does not hydrate it
- Phase 2 includes a one-time migration from the current repo-root-backed locations into these
   shared roots before Compose is switched over

### 3.4 Service-owned state

Service-owned state remains outside the project tree in Docker volumes or service-specific storage.

Already appropriate in the current topology:

| Volume | Owner |
|---|---|
| `mlflow_pg_data` | postgres |
| `qdrant_data` | qdrant |
| `rabbitmq_data` | rabbitmq |
| `redis_data` | redis |
| `airflow_logs` | airflow |
| `grafana_data` | grafana |
| `prometheus_data` | prometheus |
| `redisinsight_data` | redisinsight |

### 3.5 What comes from Git automatically

These do not need separate persistent copies because each release restores them from Git:

- `src/`
- `dags/`
- `infra/`
- `experiments/`
- `pyproject.toml`, lock files, `.pre-commit-config.yaml`
- `assets/**/*.dvc`
- `.dvc/config`
- `src/shared/knowledge_bases.json`

---

## 4. State Taxonomy And Mount Rules

### 4.1 Project checkout state

Git-tracked source and metadata restored into each release.

Examples:

- `src/`
- `dags/`
- `infra/`
- `experiments/`
- `assets/**/*.dvc`
- `.dvc/config`

### 4.2 Local config and secrets

Machine-local, operator-managed files.

Examples:

- `/home/anton-m/agent-042/.env`
- `/home/anton-m/agent-042/.dvc/config.local`

### 4.3 Shared project data

Mutable project data that must be visible to the server filesystem, Airflow, and Jupyter.

Examples:

- `assets/models`
- `assets/adapters`
- `assets/datasets`
- `assets/rag_data/<dataset>`
- `artifacts/training`

### 4.4 Service-owned runtime state

Examples:

- PostgreSQL data
- Qdrant storage
- Redis data
- RabbitMQ data
- Grafana and Prometheus state

### 4.5 Disposable working state

Examples:

- release directories under `/home/anton-m/agent-042/releases/`
- Airflow temporary clones under `/tmp/agent-042-*`
- short-lived export or staging directories created during deploy

### 4.6 Mounting rule

Do **not** bind-mount the whole runtime-tree `assets/` directory.

Reason:

- `assets/` mixes mutable payload trees with Git-tracked metadata and pointer files
- mounting the whole parent would hide tracked repo content that should come from the release

Do mount whole domain subtrees where the runtime needs materialized payloads but does not need to
edit the Git-tracked metadata in place.

Final contract:

- `${ASSETS_ROOT}/models` → service-specific model mount
- `${ASSETS_ROOT}/adapters` → service-specific adapter mount
- `${ASSETS_ROOT}/datasets` → runtime `assets/datasets`
- `${ASSETS_ROOT}/rag_data` → runtime `assets/rag_data`
- `${ARTIFACTS_ROOT}/training` → runtime `artifacts/training`

There is no separate-dataset mount phase in this plan.

Why whole `assets/rag_data` is still correct:

- the active runtime needs the materialized payloads under `assets/rag_data/<dataset>`
- the Git-tracked `assets/rag_data/*.dvc` files matter only inside the temporary clone
- the temp clone reconstructs those tracked files from Git and stages shared payload directories
  under the same repo paths before running `dvc add`

That is the key architectural boundary that makes whole `assets/rag_data` consistent.

---

## 5. Fixing Airflow Ownership

The root cause is not “Airflow writes files.” The root cause is “Airflow writes into host bind
mounts whose ownership model was never prepared for container users.”

The final fix is:

1. bind-mount only the selected shared roots and domain subtrees described above
2. create one shared host group, for example `agent042`, for operator-owned roots
3. set that group on `/home/anton-m/agent-042/assets`, `/home/anton-m/agent-042/artifacts`, and `/home/anton-m/agent-042/.dvc`
4. apply setgid on writable directories and default ACLs on `/home/anton-m/agent-042/assets` and
   `/home/anton-m/agent-042/artifacts`
5. grant explicit ACL entries to the effective Airflow worker and Jupyter container UIDs
6. grant the Airflow worker UID read access to `/home/anton-m/agent-042/.dvc/config.local`
7. remove `airflow-prepare-dirs` from the steady-state topology

Example intent on the server:

```text
group: agent042
shared writable roots:
   /home/anton-m/agent-042/assets
   /home/anton-m/agent-042/artifacts
shared read-only local secret:
    /home/anton-m/agent-042/.dvc/config.local
access model:
   operator user + agent042 group + explicit ACLs for Airflow/Jupyter UIDs
```

### 5.1 Practical Phase 2 host-prep sequence

The current Compose file does not inject a host `agent042` group into containers. That is still
compatible with this plan. The operator should prepare the shared roots once on the host and grant
access with ACLs to the effective UIDs reported by the Airflow worker and Jupyter containers.

Recommended helper:

```bash
sudo bash scripts/setup_shared_root_permissions.sh --deploy-user "<server-login>"
```

Useful overrides:

- `--server-root /home/anton-m/agent-042`
- `--env-file <checkout>/.env`
- `--compose-file <checkout>/infra/compose/docker-compose.yaml`
- `--airflow-uid <uid>` / `--airflow-gpu-uid <uid>` / `--jupyter-uid <uid>` when UID detection
   should not use Compose

The helper expands to roughly this host sequence:

```bash
export DEPLOY_USER="<server-login>"
export CHECKOUT_ROOT="<server-checkout-path>"

sudo apt-get install -y acl
sudo groupadd --force agent042
sudo usermod -aG agent042 "$DEPLOY_USER"

sudo install -d -o "$DEPLOY_USER" -g agent042 -m 2775 \
   /home/anton-m/agent-042 \
   /home/anton-m/agent-042/assets \
   /home/anton-m/agent-042/assets/models \
   /home/anton-m/agent-042/assets/adapters \
   /home/anton-m/agent-042/assets/datasets \
   /home/anton-m/agent-042/assets/rag_data \
   /home/anton-m/agent-042/artifacts \
   /home/anton-m/agent-042/artifacts/training \
   /home/anton-m/agent-042/.dvc

AIRFLOW_UID="$(docker compose --env-file .env -f infra/compose/docker-compose.yaml run --rm --no-deps --entrypoint id airflow-worker -u | tr -d '\r')"
JUPYTER_UID="$(docker compose --env-file .env -f infra/compose/docker-compose.yaml run --rm --no-deps --entrypoint id jupyter -u | tr -d '\r')"

sudo chgrp -R agent042 /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts /home/anton-m/agent-042/.dvc
sudo find /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts -type d -exec chmod 2775 {} +
sudo setfacl -R -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts
sudo setfacl -R -d -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts

sudo install -o "$DEPLOY_USER" -g agent042 -m 640 "$CHECKOUT_ROOT/.dvc/config.local" /home/anton-m/agent-042/.dvc/config.local
sudo setfacl -m u:${AIRFLOW_UID}:r /home/anton-m/agent-042/.dvc/config.local
```

If `airflow-worker-gpu` reports a different UID than `airflow-worker`, add the same ACL entries for
that UID as well. After `usermod -aG`, the operator should re-login or run `newgrp agent042`
before relying on the new host group membership.

Consequences:

- no root `chown -R` on startup
- no `airflow-prepare-dirs` sidecar in steady state
- Jupyter, Airflow, and the server user see the same shared project data
- the active runtime tree stays disposable and mostly read-only
- missing access after Phase 2 is treated as a host-prep bug, not as a reason to restore
   `airflow-prepare-dirs`

---

## 6. Airflow DVC And Git Flow

The chosen policy is that Airflow keeps DVC behavior and also completes the Git side.

### 6.1 What Airflow owns

Airflow owns:

- data acquisition for the current RAG datasets and future additions
- writing refreshed payloads into shared `assets/rag_data/<dataset>` directories
- `dvc add` and `dvc push`
- Qdrant refresh after data versioning

### 6.2 What it does not own anymore

Airflow does **not** use the active runtime tree as its Git workspace.

For each refresh run it should:

1. write fresh payload to the shared host path under `assets/rag_data/<dataset>`
2. create a temporary clone of the repo in a disposable work directory
3. restore `.dvc/config.local` into that temporary clone
4. stage the shared payload into the clone at the standard repo path as a real directory so
   `assets/rag_data/<dataset>` inside the temp clone is acceptable to `dvc add`
5. prefer hardlinks for staged files when the filesystem allows it and fall back to regular copies
   when it does not
6. run `dvc add` and `dvc push` there
7. commit the resulting `.dvc` pointer-file changes and any `.gitignore` updates to a bot branch
8. push that bot branch and open or update a PR into the protected branch using the configured
   GitHub token
9. delete the temporary clone

This cutover must ship atomically with the shared-storage mount switch. Switching Compose to the
Git-free release layout without also moving `dvc add` into the temp clone would leave Airflow with
nowhere valid to persist the resulting pointer-file changes.

After this cutover, DVC write operations in the active release tree are out of contract. Any DAG
that needs `dvc add`, `dvc push`, or Git write-back must use the temp-clone path.

### 6.3 Bot branch policy

Bot branches are integration lanes for machine-made pointer updates.

Examples:

- `data-sync/develop`
- optionally `data-sync/main` later if a separate production sync lane is needed

Current policy:

- Airflow targets `data-sync/develop`
- resulting PRs land in `develop`
- promotion to `main` stays on the normal human-reviewed path
- bot commits must include both the changed `.dvc` files and any `.gitignore` edits produced by
   `dvc add`

### 6.4 Why this makes new datasets operationally simple

Under this design, adding a new RAG dataset means:

1. add the new dataset-producing logic or DAG
2. write payload into `assets/rag_data/<new-dataset>`
3. let the temp-clone DVC flow create or update the matching `.dvc` pointer file

It does **not** require a new Compose bind mount for that dataset.

---

## 7. Branch And Release Workflow

### 7.1 Branch structure

```text
main               ← production
develop            ← integration
feature/*          ← human development branches
data-sync/*        ← bot-managed DVC pointer branches
```

Rules:

- no direct pushes to `develop` or `main`
- all human code merges happen through GitHub pull requests
- Airflow may push only to `data-sync/*`
- bot branches are never deployed directly

### 7.2 Typical feature flow

1. Create `feature/my-change` from `develop`.
2. Push; CI runs on GitHub-hosted runners.
3. If server testing is needed:
   - choose the exact commit SHA
   - manually trigger `build-images.yml` for that branch and SHA
   - manually trigger `deploy.yml` with the same branch and SHA
4. Open a PR to `develop`.
5. Merge on GitHub after CI passes.
6. When `develop` is stable, open a PR from `develop` to `main`.

### 7.3 Runtime release flow

The active server path is `/home/anton-m/agent-042/current`.

`deploy.yml` creates a fresh release for one exact commit and then promotes it to the active
runtime path.

Contract:

1. on the GitHub runner, checkout the exact requested SHA
2. create `/home/anton-m/agent-042/releases/<sha>/` on the server
3. stream `git archive <sha>` from the runner into that directory so the release contains only
   Git-tracked files and no `.git` metadata
4. verify the release renders cleanly with the external env file and shared-root mounts
5. repoint `/home/anton-m/agent-042/current` to the new release
6. start Compose with the exact immutable image tag for that branch and SHA
7. keep the previous `current` target available until smoke checks pass

This is intentionally different from `git switch` in a long-lived live checkout.

### 7.4 Rollback

Rollback is commit-based, not branch-based.

Rollback means redeploying an older known-good SHA and pointing `current` back to the older release
or recreating it from the older SHA.

Immutable `<branch-slug>-<sha>` image tags must be retained for rollback.

Operational default: keep the newest five successful release directories on the server, including
the current one, and prune older releases only after the replacement release passes smoke checks.

---

## 8. CI/CD Design

### 8.1 `ci.yml` — validation

CI runs on GitHub-hosted runners.

Trigger rules:

- every push to any branch
- every PR targeting `develop` or `main`

```yaml
on:
  push:
    branches:
      - '**'
  pull_request:
    branches:
      - main
      - develop
```

This includes `data-sync/*` pushes so bot-generated `.dvc` PRs are still validated before merge.
Image publishing remains a separate workflow and is not triggered from `data-sync/*` pushes.

### 8.2 `build-images.yml` — image publishing

Policy:

- automatic for the merge commit that lands on `develop` or `main` through a PR merge
   (including auto-merge)
- manual `workflow_dispatch` for exact feature-branch SHAs

Tags per build:

- `<branch-slug>-<short-sha>` — immutable
- `<branch-slug>-latest` — mutable pointer

Implementation rule:

- build images directly from the checked-out GitHub Actions workspace with `docker buildx build`
   (or an equivalent direct build command), not with `docker compose build`
- do not depend on deploy-time `PROJECT_ROOT` interpolation inside GitHub-hosted CI
- the automatic publish trigger must be keyed to successful `ci.yml` completion for the exact
   post-merge SHA on `develop` or `main`, not merely to branch activity; this avoids ambiguity when
   auto-merge or multiple queued PR merges land close together

### 8.3 `deploy.yml` — manual deploy trigger

Inputs:

- `branch` — required
- `sha` — required

Behavior:

1. verify CI passed for that exact commit
2. verify every required GHCR image manifest exists for that exact `branch + sha`
3. checkout the exact SHA on the runner and stream `git archive` to the server
4. SSH into the server and materialize `/home/anton-m/agent-042/releases/<sha>/`
5. preflight the new release with `docker compose config -q`
6. set `IMAGE_TAG=<branch-slug>-<sha>` for that deployment without rewriting the canonical env
   file
7. repoint `current`, then run `docker compose pull` and `docker compose up -d`
8. run smoke checks
9. if smoke checks fail, repoint `current` to the previous release and redeploy that SHA

Deploy never builds implicitly and never deploys an uncontrolled branch tip.

Minimum smoke checks:

- `docker compose ps` shows the expected long-running services up
- the gateway health endpoint responds successfully
- the embeddings service health endpoint responds successfully
- the reranker service health endpoint responds successfully

---

## 9. Compose And Storage Changes Required

### 9.1 Keep project paths stable, change backing storage

Application-facing paths stay the same. Backing storage changes.

Examples:

```yaml
# Shared writable data
- ${ASSETS_ROOT}/datasets:/opt/airflow/project/assets/datasets
- ${ASSETS_ROOT}/rag_data:/opt/airflow/project/assets/rag_data
- ${ARTIFACTS_ROOT}/training:/opt/airflow/project/artifacts/training

# Stable model and adapter caches
- ${ASSETS_ROOT}/models:/models
- ${ASSETS_ROOT}/adapters:/adapters

# External DVC local config for Airflow temp-clone jobs
- ${DVC_CONFIG_LOCAL_PATH}:/opt/airflow/project/.dvc/config.local:ro
```

Jupyter should stop mounting `${PROJECT_ROOT}/assets` from the checkout and instead mount the same
shared host assets root, for example:

```yaml
- ${ASSETS_ROOT}:/home/jovyan/assets:rw
```

Its code-facing mounts can still come from the active release, but asset reads and writes should go
through the external shared root rather than the disposable checkout.

### 9.2 Remove runtime dependence on `airflow-prepare-dirs`

Once host permissions are managed correctly and the temp-clone DVC path is in place,
`airflow-prepare-dirs` should be removed from the steady-state topology.

In the current checkout-based rollout, that means the operator must pre-create the shared roots and
apply the ACL/bootstrap sequence above before the first Airflow start that uses the external mounts.

The release tree should no longer be treated as a writable `.dvc` workspace after this point.

### 9.3 Switch project-owned services from `build:` to `image:`

Each project-owned service gets an `image:` key alongside or replacing `build:` so the server can
pull CI-built images instead of rebuilding locally.

### 9.4 External deployment env vars

The deployment env contract should include at least:

```dotenv
PROJECT_ROOT=/home/anton-m/agent-042/current
ASSETS_ROOT=/home/anton-m/agent-042/assets
ARTIFACTS_ROOT=/home/anton-m/agent-042/artifacts
DVC_CONFIG_LOCAL_PATH=/home/anton-m/agent-042/.dvc/config.local
GITHUB_REPOSITORY=<owner>/<repo>
GITHUB_DATA_SYNC_TOKEN=<fine-grained-pat>
```

The env file itself should live at `/home/anton-m/agent-042/.env` and be passed directly to Compose.

Scope clarification:

- `PROJECT_ROOT` in `/home/anton-m/agent-042/.env` is a host-side Compose interpolation value that points to
   `/home/anton-m/agent-042/current`
- container-local variables named `PROJECT_ROOT` may still point at in-container paths such as
   `/opt/airflow/project`; they are not the same path and should be treated as a different scope
- `IMAGE_TAG` is deployment-scoped and should be injected by `deploy.yml` for the target release,
   rather than rewritten permanently in the canonical env file on every deploy

---

## 10. Implementation Order

### Phase 1 — Finalize server root contract

Define and document the deployment root layout under `/home/anton-m/agent-042`.

- `.env`
- `.dvc/config.local`
- `assets/`
- `artifacts/`
- `releases/`
- `current` symlink

Files changed:

- `DELIVERY-WORKFLOW-PLAN.md`
- `.env.example`
- `infra/README.md`

### Phase 2 — Shared storage cutover and Airflow DVC boundary

This is the first runtime-affecting storage milestone.

- migrate the current repo-root-backed `assets/rag_data`, any required materialized
   `assets/datasets`, and `artifacts/training` into `/home/anton-m/agent-042/assets` and
   `/home/anton-m/agent-042/artifacts`
- switch Compose and Jupyter to the final shared-root mounts
- use whole `assets/datasets` and whole `assets/rag_data`
- add external `.dvc/config.local` availability for Airflow
- implement Airflow temp-clone DVC and Git write-back
- remove `airflow-prepare-dirs` from steady state

This phase must ship as one rollout. Do not switch the runtime to the Git-free release layout while
the DAGs still run `dvc add` in the active runtime tree.

Risk note: Phase 2 lands before hosted CI exists in this plan, so the DAG refactor and storage
cutover ship with local-only validation. For a solo engineer that is acceptable, but it should be
treated as an explicit gap and covered by manual `pytest` plus targeted DAG and deploy smoke checks
before merging the Phase 2 PR.

Files changed:

- `infra/compose/docker-compose.yaml`
- `.env.example`
- `infra/README.md`
- `dags/arxiv_rag_update.py`
- `dags/pytorch_docs_rag_update.py`
- helper for Airflow Git sync

### Phase 3 — CI validation

Add hosted GitHub Actions validation.

- `.github/workflows/ci.yml`

### Phase 4 — Image registry

Add CI image publishing and wire Compose to `image:` tags.

- `.github/workflows/build-images.yml`
- `infra/compose/docker-compose.yaml`

The workflow should build images directly from the GitHub Actions checkout, not through Compose.

### Phase 5 — Clean release deploy

Add manual deploy automation for exact SHAs and `releases/current` promotion.

- `.github/workflows/deploy.yml`
- optional deploy helper script if needed

This phase also adds release retention and rollback-safe smoke checks.

There is intentionally no earlier phase that mounts `assets/rag_data` per dataset and then later
changes it. The plan goes directly to the final whole-subtree contract.

---

## 11. Files To Create Or Modify

| File | Action | Purpose |
|---|---|---|
| `.github/workflows/ci.yml` | Create | pre-commit + pytest on every push |
| `.github/workflows/build-images.yml` | Create | Build and push project images to ghcr.io from direct GitHub Actions build contexts |
| `.github/workflows/deploy.yml` | Create | Manual clean-release deploy by exact branch + SHA, including archive materialization, smoke checks, and rollback |
| `infra/compose/docker-compose.yaml` | Modify | Final shared-root mounts, Jupyter asset mount correction, image tags, and no steady-state `airflow-prepare-dirs` dependency |
| `.env.example` | Modify | Add `ASSETS_ROOT`, `ARTIFACTS_ROOT`, `DVC_CONFIG_LOCAL_PATH`, `GITHUB_REPOSITORY`, `GITHUB_DATA_SYNC_TOKEN`, `IMAGE_TAG`, and related deploy vars |
| `infra/README.md` | Modify | Document the server root contract and deploy flow |
| `dags/arxiv_rag_update.py` | Modify | Move DVC and Git write-back to the temp-clone flow |
| `dags/pytorch_docs_rag_update.py` | Modify | Move DVC and Git write-back to the temp-clone flow |
| `scripts/migrate_shared_state.sh` | Create | One-time migration of current repo-root-backed shared data into `/home/anton-m/agent-042` |
| `scripts/setup_shared_root_permissions.sh` | Create | One-time helper to create shared roots, discover runtime UIDs, and apply ACL-based permissions |
| helper for Airflow Git sync | Create | Shared implementation of temp clone, staged dataset copy or hardlinking, commit, push, and PR update |

Assumptions intentionally dropped from the plan:

- no long-lived live Git runtime checkout
- no named-volume plan for shared `assets/rag_data` or `artifacts/training`
- no manual-only DVC lane
- no temporary per-dataset `rag_data` bind-mount phase
- no steady-state root `airflow-prepare-dirs`

---

## 12. Growth Path

The chosen architecture scales cleanly.

| Now (single node) | Later (multi-node or K8s) |
|---|---|
| Fresh release directories under one host path | Immutable release artifacts or image-only deploys in a cluster |
| Shared host `assets/` and `artifacts/` roots | PVCs, shared NFS, or object-backed mounts |
| GitHub-hosted CI runners | Optional self-hosted runners for GPU or heavy integration jobs |
| Airflow temp clone + bot branch | Dedicated data-sync service or GitHub App |
| Manual `workflow_dispatch` deploy | GitOps controller or release promotion pipeline |

None of these later steps require changing the in-project paths used by the application. The point
of the design is to keep the logical repo structure stable while separating runtime, config,
shared data, and Git operations in a production-like way.
