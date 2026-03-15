"""
DEPRECATED: This script has been renamed to reinforce_nash_trap.py
================================================================
The original filename 'ppo_nash_trap.py' was misleading because the
actual algorithm is independent REINFORCE (not PPO). This wrapper
exists for backward compatibility with existing pipelines and results.

For the actual implementation, see: reinforce_nash_trap.py
"""
import sys
import os
import warnings

warnings.warn(
    "ppo_nash_trap.py is deprecated. Use reinforce_nash_trap.py instead. "
    "The algorithm is independent REINFORCE, not PPO.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and run the renamed module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from reinforce_nash_trap import *  # noqa: F401, F403

if __name__ == "__main__":
    from reinforce_nash_trap import main
    main()
