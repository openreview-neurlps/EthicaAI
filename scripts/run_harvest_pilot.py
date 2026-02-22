"""Harvest Pilot Runner"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation.jax.run_full_pipeline import run_full_pipeline

# Override config to use Harvest
config_override = {"ENV_NAME": "harvest"}

print("Starting Harvest Pilot...")
run_full_pipeline(
    scale="pilot", 
    svo_angles={"selfish": 0.0, "altruistic": 1.309},  # Test extreme cases only
    seeds=[42], 
    output_dir="simulation/outputs/pilot_harvest",
    config_override=config_override
)
print("Harvest Pilot Complete!")
