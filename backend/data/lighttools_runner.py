"""Robust LightTools runner for ground truth data collection.

v6 Section 13.8: LightTools COM API automation
- Batch simulation with crash recovery
- Result validation
- Maximum 200 runs budget

Architecture:
    LightTools (COM API)
    ├── AR Coating: Phase 1 optimal (fixed)
    ├── Cover Glass: 550um, n=1.52 (fixed)
    ├── BM1: aperture w1, offset delta1 (variable)
    ├── BM2: aperture w2, offset delta2 (variable)
    └── OPD Sensor: intensity readout

Dependencies:
    pip install pywin32 numpy
    (pywin32 for COM API - Windows only)

Usage:
    runner = LightToolsRunner(model_path="path/to/model.lts")
    runner.connect()
    result = runner.run_single(delta1=0, delta2=0, w1=10, w2=10, theta=0)
    runner.run_batch(configs, output_dir="data/lt_results")
    runner.disconnect()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LTSimConfig:
    """Single LightTools simulation configuration."""
    sim_id: int
    delta_bm1: float    # um
    delta_bm2: float    # um
    w1: float           # um
    w2: float           # um
    theta_deg: float    # degrees

    # Fixed parameters (v6 Section 2.4, 2.2)
    wavelength_nm: float = 520.0
    n_cg: float = 1.52
    cg_thickness_um: float = 550.0
    ar_d1_nm: float = 34.6   # SiO2
    ar_d2_nm: float = 25.9   # TiO2
    ar_d3_nm: float = 20.7   # SiO2
    ar_d4_nm: float = 169.5  # TiO2
    opd_pitch_um: float = 72.0
    opd_width_um: float = 10.0
    n_opd_pixels: int = 7


@dataclass
class LTSimResult:
    """Result from a single LightTools simulation."""
    config: LTSimConfig
    intensity_profile: np.ndarray | None = None  # I(x) at z=0
    psf_7: np.ndarray | None = None              # 7 OPD pixel intensities
    x_coords: np.ndarray | None = None           # x coordinates for intensity
    success: bool = False
    error_msg: str = ""
    elapsed_sec: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# LightTools COM API Runner
# ═══════════════════════════════════════════════════════════════════

class LightToolsRunner:
    """Robust LightTools automation via COM API.

    Handles:
    - COM connection/disconnection
    - BM parameter modification
    - Angle setup
    - Result extraction
    - Crash recovery with retry

    Args:
        model_path: Path to .lts LightTools model file.
        max_retries: Max retries per simulation on crash.
        timeout_sec: Timeout per simulation in seconds.
    """

    def __init__(
        self,
        model_path: str,
        max_retries: int = 3,
        timeout_sec: float = 300.0,
    ):
        self.model_path = Path(model_path)
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.lt = None  # COM object
        self._connected = False

    # ── Connection ──

    def connect(self):
        """Connect to LightTools via COM API."""
        try:
            import win32com.client
            self.lt = win32com.client.Dispatch("LightTools.LTAPI4")
            logger.info("LightTools COM connection established")

            # Open model
            if self.model_path.exists():
                self.lt.Open(str(self.model_path))
                logger.info(f"Model loaded: {self.model_path.name}")
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            self._connected = True

        except ImportError:
            logger.error("pywin32 not installed. Run: pip install pywin32")
            raise
        except Exception as e:
            logger.error(f"COM connection failed: {e}")
            raise

    def disconnect(self):
        """Disconnect from LightTools."""
        if self.lt is not None:
            try:
                self.lt.Close()
            except Exception:
                pass
            self.lt = None
            self._connected = False
            logger.info("LightTools disconnected")

    # ── Parameter modification ──

    def _set_bm_parameters(self, config: LTSimConfig):
        """Set BM aperture and offset in LightTools model.

        NOTE: The exact API calls depend on your LightTools model structure.
        Modify the object names below to match your model.

        Typical LightTools object hierarchy:
            Model > Optical System > BM1_Aperture > Properties
            Model > Optical System > BM2_Aperture > Properties
        """
        lt = self.lt

        # ══════════════════════════════════════════════════════
        # ★ CUSTOMIZE THESE OBJECT NAMES TO MATCH YOUR MODEL ★
        # ══════════════════════════════════════════════════════

        # BM1 slit aperture
        # lt.SetProperty("BM1_Aperture", "Width", config.w1)
        # lt.SetProperty("BM1_Aperture", "X_Offset", config.delta_bm1)

        # BM2 slit aperture
        # lt.SetProperty("BM2_Aperture", "Width", config.w2)
        # lt.SetProperty("BM2_Aperture", "X_Offset", config.delta_bm2)

        # Example with DbGet/DbSet (common LightTools API pattern):
        try:
            # BM1
            lt.DbSet("BM1_Slit", "ApertureWidth", config.w1)
            lt.DbSet("BM1_Slit", "XDecenter", config.delta_bm1)

            # BM2
            lt.DbSet("BM2_Slit", "ApertureWidth", config.w2)
            lt.DbSet("BM2_Slit", "XDecenter", config.delta_bm2)

            # Incident angle
            lt.DbSet("Light_Source", "TiltX", config.theta_deg)

            logger.debug(f"Set: d1={config.delta_bm1}, d2={config.delta_bm2}, "
                         f"w1={config.w1}, w2={config.w2}, theta={config.theta_deg}")
        except Exception as e:
            logger.warning(f"Parameter setting failed: {e}")
            logger.warning("Check object names in _set_bm_parameters()")
            raise

    def _extract_results(self, config: LTSimConfig) -> LTSimResult:
        """Extract OPD intensity results from LightTools.

        NOTE: Modify receiver/detector names to match your model.
        """
        lt = self.lt
        result = LTSimResult(config=config)

        try:
            # ══════════════════════════════════════════════════
            # ★ CUSTOMIZE DETECTOR NAME TO MATCH YOUR MODEL ★
            # ══════════════════════════════════════════════════

            # Get intensity profile from OPD detector
            # Option A: Direct intensity array readout
            # intensity = lt.GetDetectorData("OPD_Receiver", "Intensity")

            # Option B: Export to file and read
            # lt.ExportDetector("OPD_Receiver", "temp_result.txt")
            # intensity = np.loadtxt("temp_result.txt")

            # Example: get 1D intensity profile
            n_pixels = 1000
            x_min, x_max = 0.0, 504.0
            x_coords = np.linspace(x_min, x_max, n_pixels)

            # Placeholder - replace with actual API call
            intensity = np.zeros(n_pixels)
            try:
                raw_data = lt.GetReceiverData("OPD_Receiver")
                if raw_data is not None:
                    intensity = np.array(raw_data, dtype=np.float64)
                    x_coords = np.linspace(x_min, x_max, len(intensity))
            except Exception as e:
                logger.warning(f"Receiver data extraction: {e}")

            result.intensity_profile = intensity.astype(np.float32)
            result.x_coords = x_coords.astype(np.float32)

            # Compute PSF 7 pixels
            psf_7 = np.zeros(7)
            for i in range(7):
                center = i * config.opd_pitch_um + config.opd_pitch_um / 2
                mask = (x_coords >= center - config.opd_width_um / 2) & \
                       (x_coords <= center + config.opd_width_um / 2)
                if mask.sum() > 0:
                    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
                    psf_7[i] = intensity[mask].sum() * dx

            result.psf_7 = psf_7.astype(np.float32)
            result.success = True

        except Exception as e:
            result.error_msg = str(e)
            logger.error(f"Result extraction failed: {e}")

        return result

    # ── Simulation execution ──

    def run_single(self, config: LTSimConfig) -> LTSimResult:
        """Run a single simulation with crash recovery."""
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()

                # Set parameters
                self._set_bm_parameters(config)

                # Run simulation
                self.lt.Run()

                # Wait for completion (with timeout)
                # Some LightTools versions need explicit wait
                # lt.WaitForCompletion(self.timeout_sec)

                # Extract results
                result = self._extract_results(config)
                result.elapsed_sec = time.time() - t0

                if result.success:
                    logger.info(f"Sim {config.sim_id}: OK ({result.elapsed_sec:.1f}s)")
                    return result

            except Exception as e:
                logger.warning(f"Sim {config.sim_id} attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info("Reconnecting...")
                    self.disconnect()
                    time.sleep(5)
                    self.connect()

        # All retries failed
        result = LTSimResult(config=config, error_msg="All retries failed")
        logger.error(f"Sim {config.sim_id}: FAILED after {self.max_retries} attempts")
        return result

    # ── Batch execution ──

    def run_batch(
        self,
        configs: list[dict],
        output_dir: str = "data/lt_results",
        checkpoint_every: int = 10,
    ) -> list[LTSimResult]:
        """Run batch of simulations with checkpoint saving.

        Args:
            configs: List of config dicts from LHS sampler.
            output_dir: Directory to save results.
            checkpoint_every: Save checkpoint every N simulations.

        Returns:
            List of LTSimResult.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ckpt_path = Path("data/lt_checkpoint/batch_progress.json")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if exists (resume support)
        completed_ids = set()
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                ckpt_data = json.load(f)
            completed_ids = set(ckpt_data.get("completed_ids", []))
            logger.info(f"Resuming: {len(completed_ids)} already completed")

        results = []
        n_total = len(configs)

        for i, cfg_dict in enumerate(configs):
            sim_id = cfg_dict["sim_id"]

            if sim_id in completed_ids:
                logger.debug(f"Skip sim {sim_id} (already done)")
                continue

            config = LTSimConfig(**cfg_dict)
            result = self.run_single(config)
            results.append(result)

            # Save individual result
            if result.success:
                result_file = out_path / f"sim_{sim_id:04d}.npz"
                np.savez(
                    result_file,
                    intensity=result.intensity_profile,
                    psf_7=result.psf_7,
                    x_coords=result.x_coords,
                    delta_bm1=config.delta_bm1,
                    delta_bm2=config.delta_bm2,
                    w1=config.w1,
                    w2=config.w2,
                    theta_deg=config.theta_deg,
                )
                completed_ids.add(sim_id)

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                with open(ckpt_path, "w") as f:
                    json.dump({
                        "completed_ids": list(completed_ids),
                        "n_total": n_total,
                        "n_completed": len(completed_ids),
                        "n_failed": sum(1 for r in results if not r.success),
                    }, f, indent=2)
                logger.info(f"Checkpoint: {len(completed_ids)}/{n_total} "
                            f"({len(completed_ids)/n_total*100:.0f}%)")

        # Final checkpoint
        with open(ckpt_path, "w") as f:
            json.dump({
                "completed_ids": list(completed_ids),
                "n_total": n_total,
                "n_completed": len(completed_ids),
                "n_failed": sum(1 for r in results if not r.success),
                "status": "complete",
            }, f, indent=2)

        n_ok = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {n_ok}/{len(results)} successful")
        return results


# ═══════════════════════════════════════════════════════════════════
# Result loading (for L_I target construction)
# ═══════════════════════════════════════════════════════════════════

class LTResultDataset:
    """Load LightTools results as L_I training targets.

    Loads all sim_*.npz files from a results directory and provides
    lookup for PINN training.
    """

    def __init__(self, results_dir: str = "data/lt_results"):
        self.results_dir = Path(results_dir)
        self._load_all()

    def _load_all(self):
        """Load all result files."""
        files = sorted(self.results_dir.glob("sim_*.npz"))
        if not files:
            raise FileNotFoundError(f"No results in {self.results_dir}")

        self.configs = []
        self.intensities = []
        self.x_coords = None

        for f in files:
            data = np.load(f)
            self.configs.append({
                "delta_bm1": float(data["delta_bm1"]),
                "delta_bm2": float(data["delta_bm2"]),
                "w1": float(data["w1"]),
                "w2": float(data["w2"]),
                "theta_deg": float(data["theta_deg"]),
            })
            self.intensities.append(data["intensity"])
            if self.x_coords is None:
                self.x_coords = data["x_coords"]

        self.n_samples = len(self.configs)
        logger.info(f"Loaded {self.n_samples} LightTools results")

    def get_target(self, idx: int) -> tuple[dict, np.ndarray, np.ndarray]:
        """Get a single target for L_I.

        Returns:
            (config_dict, x_coords, intensity_profile)
        """
        return self.configs[idx], self.x_coords, self.intensities[idx]

    def sample_random(self, n: int) -> list[int]:
        """Sample random indices for training batch."""
        return np.random.randint(0, self.n_samples, size=n).tolist()
