"""Command-line version of the Colab notebook.

Example:
  python scripts/run_experiments.py --get_root ../GET --dataset busi \
      --vae_ckpt ../GET/SD-VAE-weight/768-v-ema-first-stage-VAE.ckpt --out results/busi
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_ptq.experiments import DEFAULT_CONFIGS, run_experiments  # noqa: E402
from get_ptq.figures import make_all_figures  # noqa: E402
from get_ptq.report import summary_markdown  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--get_root", required=True)
    ap.add_argument("--dataset", default="busi")
    ap.add_argument("--data_dir", help="default: <get_root>/Dataset/<dataset>")
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS),
                    help=f"subset of: {', '.join(DEFAULT_CONFIGS)}")
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--calib_method", default="max", choices=["max", "entropy"])
    ap.add_argument("--max_test", type=int, default=None, help="only for quick tests; omit for the whole test set")
    a = ap.parse_args()

    configs = {k: DEFAULT_CONFIGS[k] for k in a.configs}
    data_dir = a.data_dir or os.path.join(a.get_root, "Dataset", a.dataset)
    res = run_experiments(a.get_root, a.dataset, data_dir, a.vae_ckpt, a.out, configs=configs,
                          n_calib=a.n_calib, calib_method=a.calib_method, max_test=a.max_test)
    md = summary_markdown(res["summary"], res["env"])
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    print(md)
    focus = [c for c in ("int8_all", "mix_int8_fp16", "fp16") if c in configs]
    for p in make_all_figures(a.out, data_dir, {**DEFAULT_CONFIGS, **configs}, focus or ["fp32"], dataset=a.dataset):
        print("figure:", p)


if __name__ == "__main__":
    main()
