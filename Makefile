.PHONY: debug-scores fitcheck-qr

debug-scores:
	python tools/debug_scores.py \
	  --in  results/fitcheck/ddim_cifar10/*/artifacts/scores/eval_in.pt \
	  --out results/fitcheck/ddim_cifar10/*/artifacts/scores/eval_out.pt

fitcheck-qr:
	bash scripts/fitcheck.sh \
	  --ckpt runs/ddim_cifar10/main/ema_40000.ckpt \
	  --run_qr
