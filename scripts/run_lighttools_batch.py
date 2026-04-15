"""LightTools batch runner for ground truth collection.

Usage:
    python scripts/run_lighttools_batch.py
    python scripts/run_lighttools_batch.py --plan data/lt_checkpoint/simulation_plan.json
    python scripts/run_lighttools_batch.py --model C:/LightTools/UDFPS.lts --n_designs 40
    python scripts/run_lighttools_batch.py --resume  # resume from checkpoint

Max budget: 200 runs total.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.data.lhs_sampler import generate_lhs_samples, save_simulation_plan
from backend.data.lighttools_runner import LightToolsRunner


def parse_args():
    parser = argparse.ArgumentParser(description="LightTools Batch Runner")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .lts LightTools model file")
    parser.add_argument("--plan", type=str, default=None,
                        help="Path to simulation plan JSON")
    parser.add_argument("--n_designs", type=int, default=40,
                        help="Number of LHS design samples")
    parser.add_argument("--n_angles", type=int, default=5,
                        help="Number of angle values")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="data/lt_results",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "data" / "lt_checkpoint" / "batch.log"),
        ],
    )
    logger = logging.getLogger("lt_batch")

    # Load or generate plan
    if args.plan:
        with open(args.plan) as f:
            plan_data = json.load(f)
        configs = plan_data["configs"]
        logger.info(f"Loaded plan: {len(configs)} simulations")
    else:
        logger.info(f"Generating LHS plan: {args.n_designs} designs x {args.n_angles} angles")
        plan = generate_lhs_samples(args.n_designs, n_angles=args.n_angles)

        plan_path = ROOT / "data" / "lt_checkpoint" / "simulation_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        save_simulation_plan(plan, str(plan_path))
        configs = plan["all_configs"]
        logger.info(f"Plan saved: {plan_path}")

    n_total = len(configs)
    if n_total > 200:
        logger.warning(f"Total sims ({n_total}) exceeds budget (200). Truncating.")
        configs = configs[:200]

    logger.info(f"Total simulations: {len(configs)}")

    # Find model path
    model_path = args.model
    if model_path is None:
        logger.error("No model path specified. Use --model <path.lts>")
        logger.info("Example: python scripts/run_lighttools_batch.py "
                     "--model C:/LightTools/UDFPS_Phase1.lts")
        sys.exit(1)

    # Run
    output_dir = str(ROOT / args.output)
    runner = LightToolsRunner(model_path=model_path, max_retries=3)

    try:
        runner.connect()
        results = runner.run_batch(
            configs=configs,
            output_dir=output_dir,
            checkpoint_every=10,
        )

        n_ok = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {n_ok}/{len(results)} successful")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise
    finally:
        runner.disconnect()


if __name__ == "__main__":
    main()
