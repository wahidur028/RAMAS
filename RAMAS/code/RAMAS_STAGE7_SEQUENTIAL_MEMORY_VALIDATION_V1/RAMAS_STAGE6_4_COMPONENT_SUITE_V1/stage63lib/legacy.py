"""Import the bundled, unchanged Stage 6.1 implementation explicitly."""
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'stage61'
sys.path.insert(0, str(VENDOR))
import run_stage6_1 as legacy
from stage6lib import source_adapter as source
from stage6lib.agent import SYSTEM_PROMPT, ControlledProvider
from stage6lib.contracts import validate_decision
from stage6lib.memory import EpisodicMemory, state_vector
from stage6lib.trust import RegimeTrust, blend_exposure
