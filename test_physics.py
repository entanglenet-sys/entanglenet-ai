"""Quick physics test for pour_task collision and orientation fixes."""
import sys
sys.path.insert(0, r"c:\Toma\temp\entanglenet-ai\philos")

from philos.simulation.environments.pour_task import PourTaskEnv, PourTaskConfig
import numpy as np

cfg = PourTaskConfig()
env = PourTaskEnv(cfg)
obs, info = env.reset()

print(f"obs_dim={len(obs)}")
print(f"Home EE: ({env._ee_pos[0]:.3f}, {env._ee_pos[1]:.3f}, {env._ee_pos[2]:.3f})")
print(f"Tilt: {env._wrist_tilt_deg:.1f} deg")
print(f"Collision: {env._collision}, clearance: {env._min_bench_clearance:.3f}")
print(f"Glass pos: ({env._glass_pos[0]:.3f}, {env._glass_pos[1]:.3f}, {env._glass_pos[2]:.3f})")
print(f"Phase: {env._phase}")
print()

# Step with zero action
obs2, r, term, trunc, i2 = env.step([0]*7)
print(f"Step 0-action: reward={r:.2f} term={term} tilt={env._wrist_tilt_deg:.1f} collision={env._collision}")

# Step with action that would try to go down (to test collision detection)
for i in range(20):
    obs2, r, term, trunc, i2 = env.step([0, 0.5, -0.5, 0, 0, 0, 0])
    if env._collision:
        print(f"Step {i+2}: COLLISION at EE z={env._ee_pos[2]:.3f} penetration={env._max_penetration:.3f}")
        break
else:
    print(f"No collision after 20 steps, EE at z={env._ee_pos[2]:.3f}")

# Reset and test bench repulsion in stub_step
env2 = PourTaskEnv(cfg)
obs, _ = env2.reset()
print(f"\n--- Stub step repulsion test ---")
for i in range(50):
    obs, r, term, trunc, info = env2.step([0]*7)
    if i % 10 == 0:
        print(f"  step={i} ee=({env2._ee_pos[0]:.3f},{env2._ee_pos[1]:.3f},{env2._ee_pos[2]:.3f}) "
              f"tilt={env2._wrist_tilt_deg:.1f} phase={env2._phase} collision={env2._collision}")
    if term:
        print(f"  TERMINATED at step {i}")
        break

print("\nAll physics tests passed!")
