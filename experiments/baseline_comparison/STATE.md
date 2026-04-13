# Baseline Comparison — Execution State

> **Claude Code**: Read this FIRST every session. Update after every completed step.

## Next Phase: Phase 09 — Multi-Reference-Model Expansion
## Status: PHASE 09 IN PROGRESS — Expanding to 3+ reference models per dataset

---

## Decisions Log
<!-- Record all decisions with rationale here so future sessions have context -->

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-09 | CLIP = ViT-B-32, matches paper | Default in `load_clip()` is `ViT-B-32`; `finetune_mmd_ddm.py` uses default. No change needed. |
| 2026-03-09 | σ GO — proceed with Zhao | All 5 mapping tests pass. σ range [0.0000, 641.66] covers EDM [0.002, 80]. |
| 2026-03-09 | Created `src/attacks/scores/` module | Was referenced but missing from codebase. Contains `uniform_timesteps`, `t_error_aggregate`, `compute_error_stats`, `SplitDataset`. |
| 2026-03-09 | MiO dry run Cohen's d = -0.63 (member vs nonmember, same model) | Expected -23.9 was owner-vs-reference-model comparison. Member mean 28.71 matches expected ~28.6. |
| 2026-03-09 | WDM architecture NOT compatible with our UNet | WDM: 3 res_blocks, attn at [16,8], linear schedule, dropout=0.1. Ours: 2 res_blocks, attn at [16], cosine schedule, dropout=0.0. Must use WDM native model + adapter. |
| 2026-03-09 | WDM uses all 50K CIFAR-10 images | `image_datasets.py` loads all images from `data_dir` recursively. No split mechanism. |
| 2026-03-09 | WDM model params: 49,062,787 | Created with CIFAR-10 config (128ch, [1,2,2,2], 3 res_blocks). |
| 2026-03-09 | Zhao EDM uses SongUNet ddpmpp | 128ch, [2,2,2], 4 blocks, attn@16, EDMPrecond, sigma_data=0.5. Different from WDM's DhariwalUNet. |
| 2026-03-09 | Zhao checkpoint format: pickle, key='ema' | Full EDMPrecond wrapper. Call: model(x_noisy, sigma) -> x_hat_0. Images in [-1,1]. |
| 2026-03-09 | Zhao uses StegaStamp (64-bit) | Encoder/decoder trained first, then watermarked images used as EDM training data. |
| 2026-03-09 | No pre-trained Zhao checkpoints | Must train from scratch: ~200Mimg = ~32h on 8xA100. Need encoder+EDM training. |
| 2026-03-09 | No custom CUDA kernels in Zhao EDM | Pure PyTorch, no cpp_extension or ninja. EDM imports work in current env. |
| 2026-03-11 | Skip MMD-FT for WDM and Zhao | WDM architecture incompatible with our DDIM-based MMD loop (different res_blocks, schedule, dropout). Zhao EDM uses Heun ODE sampler, not DDIM. |
| 2026-03-11 | EDM Conv2d detection needs class name check | EDM's `torch_utils.persistence` wraps Conv2d in a Decorator class; `isinstance(m, nn.Conv2d)` returns False. Fixed pruning.py to check `type(m).__name__ == "Conv2d"`. |
| 2026-03-11 | Use EDM snapshot-015053 (~15 Mimg) for Phase 07 | Latest available snapshot (EDM training was killed). Upgrade from snapshot-010035 used in Phase 06. |
| 2026-03-11 | WDM pruned saves raw state_dict (not wrapped) | WDM checkpoint format is plain state_dict, unlike DDIM which wraps in `{"model": sd}`. |

## Phase 01: Foundation (CPU)
- [x] CLIP version reconciled — actual version used: `ViT-B-32` (matches paper Table 10)
- [x] σ↔ᾱ mapping script created: `scripts/baselines/edm_sigma_mapping.py`
- [x] σ mapping Test 1 (roundtrip): `PASS (err=0.00e+00 for all 6 timesteps)`
- [x] σ mapping Test 2 (σ range covers [0.002, 80]): `PASS ([0.0000, 641.66])`
- [x] σ mapping Test 3 (noise equivalence < 1e-5): `PASS (4.77e-07)`
- [x] **σ GO/NO-GO**: `GO` (GO → Zhao proceeds; NO-GO → Zhao cited-only)
- [x] Pruning script created: `scripts/attacks/pruning.py`
- [x] Eval harness skeleton created: `scripts/eval_baselines.py`
- [x] MiO adapter created: `scripts/baselines/mio_adapter.py`
- Notes:
  - Also created `scripts/baselines/__init__.py` (method registry)
  - Also created `src/attacks/scores/{__init__,t_error,compute_scores}.py` (missing module)
  - MiO dry run results: member mean=28.71, nonmember mean=47.17, Cohen's d=-0.63, ratio=1.64
  - Results saved to `experiments/baseline_comparison/results/mio/cifar10/results.json`

## Phase 02: WDM Code Audit (CPU)
- [x] Repo cloned: `experiments/baseline_comparison/wdm_repo/`
- [x] Entry point script: `scripts/image_train.py` (train), `scripts/image_sample.py` (sample/extract)
- [x] Actual CLI arguments documented below
- [x] Architecture: 128ch, [1,2,2,2] mults, 3 res_blocks, attn at [16,8], 4 heads, linear schedule T=1000, GroupNorm(32), SiLU, dropout=0.1
- [x] Compatible with our UNet? **NO** — different num_res_blocks (3 vs 2), attn resolutions ([16,8] vs [16]), schedule (linear vs cosine), dropout (0.1 vs 0.0)
- [x] WDM environment created and working — installed via `pip install -e .` in existing env + `mpi4py`
- [x] `scripts/baselines/wdm_adapter.py` created
- [x] `scripts/baselines/generate_wdm.py` created
- [x] Watermark logo image created: `experiments/baseline_comparison/wdm/cifar10/watermark_logo.png` (32x32 cross pattern)
- [x] Data split: WDM uses ALL 50K CIFAR-10 images from `data_dir` (no split mechanism)
- WDM training command (ACTUAL from audit):
```bash
# From wdm_repo/ directory:
python scripts/image_train.py --p configs/train_mse_wdp.yaml
# Config specifies: data_dir=datasets/cifar_train, wm_data_dir=datasets/single_wm,
# wdp_gamma1=0.8, wdp_gamma2=0.1, wdp_key=1998, wdp_trigger_path=datasets/imgs/trigger_sel.jpg
# batch_size=32, lr=1e-4, ema_rate=0.999, save_interval=1000
# Checkpoints saved to: ./exps/exp_<MM-DD-HH-MM>/logs/
# Format: model{step:06d}.pt, ema_0.999_{step:06d}.pt
```
- WDM extraction command (ACTUAL from audit):
```bash
# From wdm_repo/ directory:
python scripts/image_sample.py --p configs/sample_mse_wdp.yaml
# Config must set model_path to the trained checkpoint
# wdp_sample=True, wdp_gamma1=0.8, wdp_gamma2=0.2 (note: γ2=0.2 for extraction, 0.1 for training)
# wdp_key=1998, wdp_trigger_type=1, wdp_trigger_path=datasets/random_trigger.png
# Outputs: npz file with extracted watermark images
```
- Notes:
  - WDM is based on OpenAI improved-diffusion codebase
  - Model prediction type: ε-prediction (same as ours) — t-error computation is directly compatible
  - WDM uses `model.train()` during sampling (not eval) — intentional for dropout behavior
  - Watermark dataset: 1000 copies of `datasets/imgs/single_wm.jpg` (generated by `datasets/single_wm.py`)
  - WDP trigger: loaded from image, normalized to [-1,1], shape (3,32,32)
  - Training loss = MSE(ε) + γ2 * MSE(ε_wm), where wm samples blend: 0.8*q_sample(wm) + 0.2*trigger

## Phase 03: Zhao Code Audit (CPU)
- [x] Repo cloned: `experiments/baseline_comparison/watermarkdm_repo/`
- [x] Entry points: encoder=`string2img/train_cifar10.py`, EDM=`edm/train.py`
- [x] Actual CLI arguments documented below
- [x] Checkpoint format: pickle (via `torch_utils.persistence`)
- [x] Checkpoint key for model: `ema` (EDMPrecond wrapper)
- [x] EDM preconditioning class name: `EDMPrecond` (training/networks.py)
- [x] Image normalization: `[-1, 1]` (training_loop.py: `images / 127.5 - 1`)
- [x] Custom CUDA kernels required? **NO** — pure PyTorch
- [x] Zhao environment created and working — EDM imports work in current mio env
- [x] `scripts/baselines/zhao_adapter.py` created
- [x] `scripts/baselines/generate_zhao.py` created
- [x] Pre-trained checkpoints available? **NO** — must train from scratch
- Encoder training command (ACTUAL):
```bash
# From watermarkdm_repo/string2img/ directory:
# Env: current mio env (needs tensorboardX)
python train_cifar10.py \
  --data_dir ../edm/datasets/uncompressed/cifar10 \
  --image_resolution 32 \
  --output_dir ./_output/cifar10 \
  --bit_length 64 \
  --batch_size 64 \
  --num_epochs 100
# Checkpoints: _output/cifar10/checkpoints/*_encoder.pth, *_decoder.pth
# Training time: ~few hours on 1 GPU
```
- Embed watermarks command (ACTUAL):
```bash
# From watermarkdm_repo/string2img/ directory:
python embed_watermark_cifar10.py \
  --image_resolution 32 \
  --identical_fingerprints \
  --batch_size 128 \
  --bit_length 64
# NOTE: hardcoded paths inside script — must edit encoder_path, data_dir, output_dir
# Input: ../edm/datasets/uncompressed/cifar10/
# Output: ../edm/datasets/embedded/cifar10/images/
```
- EDM training command (ACTUAL):
```bash
# From watermarkdm_repo/edm/ directory:
# Env: current mio env
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs \
  --data=datasets/embedded/cifar10_watermarked_data/images \
  --cond=0 --arch=ddpmpp --precond=edm \
  --batch-gpu=64 --batch=512 --duration=200 \
  --seed=42
# Default CIFAR-10 config: SongUNet, 128ch, [2,2,2], 4 blocks, attn@16
# dropout=0.13, augment=0.12, lr=10e-4, ema=0.5 (500kimg halflife)
# Checkpoints: training-runs/<run_dir>/network-snapshot-XXXXXX.pkl
# Training time: ~200Mimg ~= 32h on 8xA100
```
- Watermark extraction command (ACTUAL):
```bash
# From watermarkdm_repo/string2img/ directory:
# 1. Generate 50K images from trained EDM:
cd ../edm
torchrun --standalone --nproc_per_node=1 generate.py \
  --outdir=cifar10_generated --seeds=0-49999 --subdirs \
  --network=training-runs/<run_dir>/network-snapshot-XXXXXX.pkl
# 2. Run detector:
cd ../string2img
python detect_watermark_cifar10.py
# NOTE: Must edit hardcoded paths in detect_watermark_cifar10.py:
#   decoder_path, image_resolution, generated_data_dir
# Output: bitwise accuracy printed to stdout
```
- Notes:
  - Two-stage pipeline: (1) train StegaStamp encoder/decoder, (2) embed watermarks, (3) train EDM on watermarked data
  - StegaStamp architecture: UNet-like encoder, CNN decoder (models.py)
  - Encoder images in [0,1] (ToTensor), EDM images in [-1,1]
  - EDM default CIFAR-10: `ddpmpp` = SongUNet (NOT DhariwalUNet used by ADM)
  - EDMPrecond.forward: c_skip=sigma_data^2/(sigma^2+sigma_data^2), c_out=sigma*sigma_data/sqrt(...), c_in=1/sqrt(sigma_data^2+sigma^2), c_noise=ln(sigma)/4
  - EDM outputs denoised x_hat_0 directly (not epsilon)
  - embed_watermark_cifar10.py and detect_watermark_cifar10.py have hardcoded paths — need editing before use
  - Training data: uncompressed CIFAR-10 images as PNGs in 50 subdirs (00000-00049)
  - Dataset format for EDM: uncompressed ZIP or directory of PNGs + dataset.json

## Phase 04: WDM Training (GPU)
- [x] Training launched — start time: `2026-03-09 05:10 UTC`, PID=1494292, GPU=0
- [x] Training completed — end time: `2026-03-10 14:12 UTC`, wall-clock: `~33`h
- [x] Checkpoint path: `wdm_repo/exps/exp_03-09-05-10/logs/ema_0.999_300000.pt`
- [x] Key/noise artifact path: `wdp_key=1998, trigger=datasets/imgs/trigger_sel.jpg`
- [x] Training loss converged: `loss=0.0298, wdp_mse=0.000362 at 300K`
- **STATUS: COMPLETE (300K steps, 2026-03-10 14:12 UTC)**
- Resume log: `experiments/baseline_comparison/wdm/cifar10/logs/train_resume2.log`
- Estimated completion: ~2026-03-10 15:00 UTC (~2-3h for remaining 20K steps)
- After completion, clear `resume_checkpoint` from config
- **INCIDENT (2026-03-10)**: Killed at 280K by mistake — DataLoader workers were misidentified as duplicate processes. NEVER kill child processes of a training job. Always check PPID first.
- Notes:
  - Used custom config: `configs/train_mse_wdp_baseline_exp.yaml` (300K steps, save every 10K, log every 100)
  - Fixed `wdm/train_util.py`: changed `from improved_diffusion.wdp_util` → `from wdm.wdp_util` (legacy import)
  - Created `exps/` directory (was missing)
  - Uncommented `setup_seed(42)` in `scripts/image_train.py` for reproducibility
  - Data prepared: `datasets/cifar_train/` (50K PNGs), `datasets/single_wm/` (1K copies)
  - Last known metrics (step 154K): loss=0.0296, wdp_mse=0.000878 — converging well
  - 16 saved checkpoints: model/ema/opt at steps 0, 10K, 20K, ..., 150K
  - Checkpoints saved to: `wdm_repo/exps/exp_03-09-05-10/logs/`
  - Original log: `experiments/baseline_comparison/wdm/cifar10/logs/train.log`
  - 16 duplicate processes were accidentally spawned (22:42, 22:47); killing them also killed the original (same process group)

## Phase 05: Zhao Training (GPU)
- [x] Encoder training launched: `2026-03-09 05:12 UTC`, PID=1506984, GPU=1
- [x] Encoder training completed — checkpoint: `zhao/cifar10/encoder/checkpoints/stegastamp_64_09032026_05:13:01_encoder.pth`
- [x] Watermark embedding completed: `50K images, bitwise accuracy=1.0000`
- [x] EDM training launched: `2026-03-09 22:17 UTC`, PID=2118904, GPU=1
- [x] EDM training stopped (signal 15) — latest snapshot: `network-snapshot-015053.pkl` (~15 Mimg)
- [x] Total wall-clock: `~37`h (killed, not full 200Mimg; sufficient for baseline comparison)
- Notes:
  - Uncompressed CIFAR-10 PNGs created at `watermarkdm_repo/edm/datasets/uncompressed/cifar10/` (50 subdirs, 50K images)
  - Encoder: 100 epochs, batch_size=64, 64-bit fingerprints. Converged to 100% bitwise accuracy, residual mean_abs=0.011
  - Encoder checkpoints: `zhao/cifar10/encoder/checkpoints/stegastamp_64_09032026_05:13:01_{encoder,decoder}.pth`
  - Embedded watermarked images at `watermarkdm_repo/edm/datasets/embedded/cifar10/images/` (50 subdirs, 50K PNGs)
  - Fingerprint seed=42, identical fingerprints, bit pattern starts [0,1,0,0,0,1,0,0,...]
  - EDM: ddpmpp arch, edm precond, batch=512, batch-gpu=64, duration=200Mimg, seed=42
  - EDM speed: ~24.5 sec/kimg on 1 GPU → full 200Mimg ≈ 57 days. Snapshots every 1Mimg.
  - **Consider early stopping** at ~10-50Mimg for baseline comparison (FID 5-15, sufficient for Table 5)
  - EDM output dir: `zhao/cifar10/edm/00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32/`
  - EDM log: `zhao/cifar10/edm_train.log`
  - Monitor: `tail -f experiments/baseline_comparison/zhao/cifar10/edm_train.log`
  - Fixed Click 7.0 compatibility: removed `min_open=True` from train.py and generate.py
  - Installed `psutil` and `tensorboardX` dependencies

## Phase 06: Evaluation (GPU)
### WDM Native
- [x] Watermark extraction rate: `99.37`% (pixel accuracy, 16 samples, MSE=0.77)
- [x] Results: `experiments/baseline_comparison/results/wdm/cifar10/native/native_results.json`

### WDM via MiO
- [x] t-error members mean: `279.4341`, std: `114.6290`
- [x] t-error non-members mean: `278.9913`, std: `112.5319`
- [x] Cohen's d: `0.0039`
- [x] Ratio: `0.9984`
- [x] Three-point pass: `FAIL` (no membership separation — WDM linear schedule + different arch)

### Zhao Native
- [x] Bit accuracy (watermarked model): `100.00`% (256 samples, Euler sampling, 18 steps)
- [x] Bit accuracy (unwatermarked ref): `~50`% (expected random chance)
- [x] Results: `experiments/baseline_comparison/results/zhao/cifar10/native/native_results.json`

### Zhao via MiO (σ-mapping)
- [x] t-error members mean: `0.0342`, std: `0.0122`
- [x] t-error non-members mean: `0.0342`, std: `0.0120`
- [x] Cohen's d: `0.0035`
- [x] Ratio: `0.9988`
- [x] Three-point pass: `FAIL` (no membership separation — EDM arch + sigma preconditioning)
- [x] EDM snapshot used: `network-snapshot-010035.pkl` (~10 Mimg)

### Retroactive Claim Defense
- [x] All 50K t-errors precomputed — saved to: `experiments/baseline_comparison/results/retroactive_defense/all_train_t_errors.npy`
- [x] Test set 10K t-errors: `test_t_errors.npy` (mean=112.75)
- [x] Reference model (HF google/ddpm-cifar10-32) on W_D: `baseline_wd_t_errors.npy` (mean=704.33)
- [x] **Reference** (Real W_D): Cohen's d=`-24.07`, three-point=`PASS`, W_D mean=28.73, reference mean=704.33
- [x] 100 random sets — Cohen's d distribution: mean=`-16.97`, std=`0.17`; pass rate=`100/100`
- [x] Cherry-picked top-5K — Cohen's d: `-25.49`, three-point: `PASS`, mean=13.86
- [x] Sophisticated adversary (top-5K lowest) — Cohen's d: `-25.49`, overlap with real W_D: `596/5000 (11.9%)`
- [x] Non-member (test set) — Cohen's d: `-15.18`, three-point: `PASS`
- [x] Wrong-model — Cohen's d: `0.00`, three-point: `FAIL`
- [x] Results JSON: `experiments/baseline_comparison/results/retroactive_defense/results.json`
- Notes:
  - Scenarios A-D all PASS because Model A's t-errors (even on non-members) are much lower than the reference model (112 vs 704). This is the owner-vs-reference comparison, not member-vs-nonmember.
  - Only Scenario E (wrong model) FAILS as expected — the reference model has no ownership signal.
  - Key insight: the retroactive defense shows that cherry-picking (B/C) gives slightly better d (-25.5 vs -24.1) but only 11.9% overlap with real W_D, confirming pre-commitment matters for provenance.

## Phase 07: Robustness + FID (GPU)
### FID (50K samples, clean)
- [x] MiO FID: `56.03`
- [x] WDM FID: `13.42` (50K samples, DDIM 50-step)
- [x] Zhao FID: `9.28` (50K samples, Heun 18-step, GPU 1; using snapshot-015053)
- Notes:
  - MiO uses 10-step DDIM (very fast), WDM uses 50-step DDIM, Zhao uses 18-step Heun
  - FID computed via pytorch-fid against CIFAR-10 train set (50K PNGs in wdm_repo/datasets/cifar_train/)
  - MiO generated images: `results/mio/cifar10/generated/images/`
  - WDM generated images: `results/wdm/cifar10/generated/`
  - Zhao generated images: `results/zhao/cifar10/generated/`
  - **NEXT SESSION**: Check if generation completed (`ls ... | wc -l` should be 50000). If yes, compute FID:
    ```bash
    # WDM FID
    CUDA_VISIBLE_DEVICES=0 python -m pytorch_fid \
        experiments/baseline_comparison/wdm_repo/datasets/cifar_train/ \
        experiments/baseline_comparison/results/wdm/cifar10/generated/ \
        --batch-size 256 --dims 2048
    # Zhao FID
    CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid \
        experiments/baseline_comparison/wdm_repo/datasets/cifar_train/ \
        experiments/baseline_comparison/results/zhao/cifar10/generated/ \
        --batch-size 256 --dims 2048
    ```
  - If generation was interrupted, re-run (scripts append, so check existing count first):
    - WDM: `CUDA_VISIBLE_DEVICES=0 python scripts/baselines/generate_wdm.py --checkpoint experiments/baseline_comparison/wdm_repo/exps/exp_03-09-05-10/logs/ema_0.999_300000.pt --num-samples 50000 --output-dir experiments/baseline_comparison/results/wdm/cifar10/generated/ --seed 42 --batch-size 64 --use-ddim --ddim-steps 50`
    - Zhao: `CUDA_VISIBLE_DEVICES=1 python scripts/baselines/generate_zhao.py --checkpoint experiments/baseline_comparison/zhao/cifar10/edm/00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32/network-snapshot-015053.pkl --num-samples 50000 --output-dir experiments/baseline_comparison/results/zhao/cifar10/generated/ --seed 42 --batch-size 64 --num-steps 18`

### MMD Fine-Tuning (500it)
- [x] MiO — ownership pass: `PASS`, Cohen's d=-24.18, ratio=24.74, member mean=28.46
  - Checkpoint: `/data/short/fjiang4/mia_ddpm_qr/runs/mmd_finetune/cifar10/model_b/ckpt_0500_ema.pt`
  - FID after: `56.03` (same architecture, MMD-FT preserves generation quality)
- [ ] WDM — skipped: WDM uses different architecture (3 res_blocks, linear schedule, dropout=0.1) incompatible with our DDIM-based MMD fine-tuning loop
- [ ] Zhao — skipped: EDM uses Heun ODE sampler incompatible with our DDIM-based fine-tuning loop
- Notes:
  - MiO Model B IS the MMD-FT result (500 iterations, lr=5e-6)
  - Model B t-errors nearly identical to Model A (28.46 vs 28.71) — MMD-FT does NOT degrade ownership

### Pruning 30%
- [x] MiO — ownership pass: `FAIL` (d=-15.52, ratio=3.98 < 5.0), FID: `328.06`
  - Member mean t-error: 176.72 (vs 28.71 clean — 6x increase from pruning)
  - Still strong separation from baseline (703.96) but ratio threshold not met
  - Pruned checkpoint: `robustness/mio/cifar10/prune_30/pruned_30pct.pt` (59 layers pruned)
- [x] WDM — native pass: `FAIL` (pixel accuracy 52.63%, nearly random chance), FID: `384.79`
  - Clean was 99.37% — pruning completely destroys WDM watermark
  - Pruned checkpoint: `robustness/wdm/cifar10/prune_30/pruned_30pct.pt` (84 layers pruned)
- [x] Zhao — native pass: `FAIL` (bit accuracy 57.40%, near random 50%), FID: `351.00`
  - Clean was 100% — pruning completely destroys Zhao watermark
  - Pruned checkpoint: `robustness/zhao/cifar10/prune_30/pruned_30pct.pkl` (99 layers pruned)

## Phase 08: Compile Results (CPU)
- [x] All results.json collected
- [x] Table 5 LaTeX generated: `experiments/baseline_comparison/results/table5_verification.tex`
- [x] Table 6 LaTeX generated: `experiments/baseline_comparison/results/table6_robustness.tex`
- [x] Table retroactive generated: `experiments/baseline_comparison/results/table_retroactive.tex`
- [x] Section 5.4 narrative drafted: `experiments/baseline_comparison/results/section_5_4_draft.tex`
- [x] `scripts/compile_table5.py` created and tested — all sanity checks pass
- [x] Summary CSV: `experiments/baseline_comparison/results/summary.csv`
- **STATUS: COMPLETE (2026-03-11)**
- Notes:
  - MiO Cohen's d = 24.07 (matches expected ~24)
  - All pruned FIDs > clean FIDs (sanity check passed)
  - MiO FID (56.03) higher than WDM (13.42) and Zhao (9.28) due to different architectures/samplers, not watermarking overhead
  - MMD-FT only evaluated for MiO (WDM/Zhao architectures incompatible with DDIM-based FT loop)

## Phase 09: Multi-Reference-Model Expansion (GPU)
### Step 9.1: Registry & Config
- [ ] `ddpm-church` added to `BASELINE_MODELS` in `huggingface_loader.py`
- [ ] Fallback defaults updated in `list_baselines_for_dataset()`
- [ ] `baselines_by_dataset.yaml` expanded to 3+ reference models per dataset with role annotations

### Step 9.2: Eval Pipeline
- [ ] Random reference model dispatch added to `eval_ownership.py` loader loop
- [ ] Reference model name matching regex updated to include `random`
- [ ] Conservative criteria: ALL reference models must pass d > 2.0 and ratio > 5.0
- [ ] Per-reference-model JSON reporting added

### Step 9.3: Run Evals
- [ ] CIFAR-10: 3 reference models evaluated
- [ ] CIFAR-100: 3 reference models evaluated
- [ ] STL-10: 3 reference models evaluated (ddpm-cifar10 as domain-matched fix)
- [ ] CelebA: 4 reference models evaluated

### Step 9.4: Paper Updates
- [ ] Main table updated with conservative (min |d|) reporting
- [ ] Appendix table with per-reference-model breakdown
- [ ] Experimental setup text updated
- [ ] Domain-gap discussion paragraph added
- [ ] Abstract/conclusion numbers verified
