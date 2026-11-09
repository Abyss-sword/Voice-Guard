"""Run a separate output-anchor protocol with the audited first-round runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_joint_recovery as runner
from vguard.anchored_joint_recovery import refine_anchored

runner.PROTOCOL = Path('docs/evidence/2026-11-09-joint-anchored/protocol.json')
runner.refine_joint = refine_anchored

if __name__ == '__main__': runner.main()
