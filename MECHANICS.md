# Bouldering2D — Mechanics Reference

This file documents how the player body and environment work. Edit it to describe your design preferences; it is not auto-generated.

---

## Player Body

The climber is a simplified stick-figure with a **pelvis** as the root and 12 articulated joints.

### Limb Segments

Lengths are defined in `PlayerConfig` (`src/bouldering2d/config.py`):

| Segment         | Config key              | Default (m) |
|-----------------|-------------------------|-------------|
| Lower torso     | `torso_lower_length`    | 0.55        |
| Upper torso     | `torso_upper_length`    | 0.45        |
| Neck            | `neck_length`           | 0.15        |
| Upper arm       | `upper_arm_length`      | 0.42        |
| Forearm         | `lower_arm_length`      | 0.42        |
| Hand            | `hand_length`           | 0.10        |
| Thigh           | `upper_leg_length`      | 0.52        |
| Shin            | `lower_leg_length`      | 0.52        |

### Joint Limits (radians)

Hard stops are applied — velocity is zeroed when a limit is hit (`player_model.py`).

| Joint        | Min    | Max   | Notes                         |
|--------------|--------|-------|-------------------------------|
| neck         | −0.6   | 0.6   |                               |
| spine        | −0.5   | 0.6   | Slight forward lean at rest   |
| l_shoulder   | −2.6   | 0     |                               |
| r_shoulder   | 0      | 2.6   |                               |
| l_elbow      | −2.6   | −0.1  | Elbow can only bend inward    |
| r_elbow      |  0.1   | 2.6   | Elbow can only bend inward    |
| l_hip        | −1.5   | -0.3  |                               |
| r_hip        | 0.3    | 1.5   |                               |
| l_knee       |  0.0   | 2     | Knees can only bend backward  |
| r_knee       |  0.0   | 2     | Knees can only bend backward  |

### Joint Dynamics

Each joint receives a torque action from the policy each step.

| Parameter           | Config key            | Default | Effect                                         |
|---------------------|-----------------------|---------|------------------------------------------------|
| Torque scale        | `torque_scales` (per-joint) | neck 1.5 / spine 4.0 / shoulder 3.5 / elbow 3.0 / hip 7.0 / knee 6.5 | Per-joint max torque in `MuscleConfig` |
| Joint damping       | `joint_damping`       | 0.92    | Velocity multiplier per step (1.0 = no damping)|
| Pelvis drag         | `pelvis_drag`         | 0.92    | Pelvis velocity multiplier per step            |

---

## Physics

Implemented in `physics.py`. The pelvis is the only mass point — joint positions are purely kinematic.

### Forces applied each step

1. **Gravity** — constant downward acceleration: `9.81 m/s²`
2. **Constraint spring** — when limbs are attached, a spring pulls the pelvis toward the position implied by the kinematic chain: `spring_force = (pelvis_target − pelvis) × constraint_spring`. `pelvis_target` is the mean of all per-attachment targets, where each target is `hold_pos − (endpoint − pelvis)`. Bending a limb whose endpoint is on an overhead hold shortens the offset, raising the target and pulling the pelvis up. Random flailing produces no net upward force — height must be earned through deliberate body mechanics.
3. **Constraint damping** — `damping_force = −pelvis_velocity × constraint_damping` applied alongside the spring when attached. Default `constraint_damping = 8.0`.

### Fall condition

If the pelvis drops below `y = −2.0`, the episode ends with a fall penalty.

---

## Grip / Contact

The policy outputs 4 grip actions (left hand, right hand, left foot, right foot), each in the range −1…1.

| Command range | Behaviour                                |
|---------------|------------------------------------------|
| < −0.3        | Release: detach from current hold        |
| −0.3 … 0.3   | Hold: keep current attachment, no search |
| > 0.3         | Grab: try to attach to the nearest hold  |

| Parameter              | Config key               | Default | Effect                                              |
|------------------------|--------------------------|---------|-----------------------------------------------------|
| Attach radius          | `attach_radius`          | 0.45 m  | Limb end-point must be within this distance to grab |
| Auto-release distance  | `contact_release_dist`   | 0.60 m  | If attached limb drifts beyond this, hold released  |

---

## Hold Field

Holds are generated procedurally and scroll with the climber.

| Parameter       | Config key          | Default | Effect                              |
|-----------------|---------------------|---------|-------------------------------------|
| Spacing (y)     | `hold_spacing_y`    | 1.00 m  | Vertical density of hold rows       |
| X jitter        | `hold_jitter_x`     | 0.8     | Horizontal spread around wall centre|
| Hold size       | `hold_size`         | 0.16 m  | Visual radius (does not affect grip) |
| Holds per row   | —                   | 3–5     | Sampled uniformly each row          |

A start cluster of 6 holds is always placed symmetrically within reach of the starting position.

---

## Muscle Fatigue

Each joint has an independent fatigue value in `[0.0, 1.0]` (0 = fresh, 1 = exhausted). Fatigue only accumulates within an episode — `recovery_rate` exists in `MuscleConfig` but is not applied during `step()`.

Accumulation per step:
```
fatigue[j] += fatigue_rate[j] × normalized_effort[j] × dt
```

where `normalized_effort = |applied_torque| / torque_scale`.

High fatigue reduces the effective torque a muscle can produce:
```
effective_torque_scale = torque_scale × angle_multiplier × (1 − fatigue_penalty_scale × fatigue)
```

This creates an incentive to **move smart**: minimise unnecessary joint effort to keep muscles fresh for hard moves.

| Parameter             | Config key              | Default                        |
|-----------------------|-------------------------|--------------------------------|
| Fatigue rate          | `fatigue_rate`          | 0.03–0.14 per joint            |
| Fatigue penalty scale | `fatigue_penalty_scale` | 0.60                           |

Fatigue is included in the observation vector (per joint, mapped `[0,1] → [−1,1]`).

---

## Reward

Multi-component reward computed each step (`RewardConfig`):

```
reward = ascent_weight × Δy
       − energy_penalty_weight × stamina_spent
       + contact_bonus × new_contacts
       + holding_bonus × attached_count
       − fall_penalty  (if fell)
       − idle_penalty  (if 0 contacts)
       − stale_contact_penalty × stale_count
       + reach_shaping_weight × (reach_potential_now − reach_potential_prev)
```

| Component              | Config key                  | Default | Notes                                         |
|------------------------|-----------------------------|---------|-----------------------------------------------|
| Ascent weight          | `ascent_weight`             | 20.0    | Dominant signal                               |
| Energy penalty         | `energy_penalty_weight`     | 0.002   | Penalises total joint effort                  |
| Contact bonus          | `contact_bonus`             | 0.01    | One-time per new grip                         |
| Holding bonus          | `holding_bonus`             | 0.005   | Per step per attached limb                    |
| Fall penalty           | `fall_penalty`              | 25.0    | Applied once on fall                          |
| Idle penalty           | `idle_penalty`              | 0.002   | Per step with zero contacts                   |
| Stale contact penalty  | `stale_contact_penalty`     | 0.02    | Per step per hold below `stale_threshold`     |
| Stale threshold        | `stale_threshold`           | −0.5 m  | Hold this far below pelvis counts as stale    |
| Reach shaping weight   | `reach_shaping_weight`      | 0.05    | Potential-based: reward free limbs toward holds|

---

---

## Episode Parameters

| Parameter         | Config key         | Default | Effect                         |
|-------------------|--------------------|---------|--------------------------------|
| Simulation rate   | `dt`               | 1/30 s  | Physics timestep               |
| Max steps         | `max_steps`        | 1200    | ~40 s at 30 Hz                 |
| Ascent target     | `ascent_target`    | 20.0 m  | Height at which episode is won |
| Gravity           | `gravity`          | 9.81    | m/s²                           |
