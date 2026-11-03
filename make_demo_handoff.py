"""Generate synthetic enhanced strips so recognition work can start independently."""
import argparse
import json
from pathlib import Path
import numpy as np
from PipelineConfig import NORMALIZED_SHAPE, ROTATION_ANGLES


def make_demo_handoff(destination):
    """Create deterministic shape-only fixture; no recognition claims are valid.

    Broadcast copies share a simple ramp before serialization, limiting memory.
    All eyes intentionally have the same pattern, so no identity signal exists.
    """
    strip = np.broadcast_to(np.linspace(0, 255, NORMALIZED_SHAPE[1]), NORMALIZED_SHAPE)
    train_labels = np.repeat(np.arange(1, 109), 3)
    test_labels = np.repeat(np.arange(1, 109), 4)
    train_ids = np.array([f"{eye:03d}_1_{i}" for eye in range(1,109) for i in range(1,4)])
    test_ids = np.array([f"{eye:03d}_2_{i}" for eye in range(1,109) for i in range(1,5)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, contract_version=1, synthetic=True,
        train_images=np.broadcast_to(strip, (324,7,*NORMALIZED_SHAPE)),
        test_images=np.broadcast_to(strip, (432,*NORMALIZED_SHAPE)),
        train_labels=train_labels, test_labels=test_labels,
        train_ids=train_ids, test_ids=test_ids, angles_deg=np.array(ROTATION_ANGLES),
        provenance=json.dumps({"synthetic": True, "purpose": "interface-only", "contract": 1}))
    print(f"Synthetic fixture written: {destination}. Never report its scores as results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_preprocessed.npz"))
    make_demo_handoff(parser.parse_args().output)
