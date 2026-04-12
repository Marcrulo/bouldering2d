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
| Torque scale        | `joint_torque_scale`  | 5.0     | Multiplies raw policy action (range −1…1)      |
| Joint damping       | `joint_damping`       | 0.92    | Velocity multiplier per step (1.0 = no damping)|
| Pelvis drag         | `pelvis_drag`         | 0.85    | Pelvis velocity multiplier per step            |

---

## Physics

Implemented in `physics.py`. The pelvis is the only mass point — joint positions are purely kinematic.

### Forces applied each step

1. **Gravity** — constant downward force: `gravity = 9.81 m/s²`
2. **Up-pull** — when any limb is attached, a force pulls the pelvis toward the centroid of all attached holds: `direction × (3.8 + 0.95 × num_attached)`
3. **Control lift** — effort from joint torques generates a small upward assist: `min(34.0, total_effort × 0.13)`
4. **Support damping** — extra upward damping of 18 m/s² when at least one limb is attached
5. **Grip boost** — additional 5 m/s² upward per extra attachment beyond the first
6. **Lateral pull** — gentle horizontal correction toward the support centroid: `Δx × 2.0`

### Fall condition

If the pelvis drops below `y = −2.0`, the episode ends with a fall penalty.

---

## Grip / Contact

The policy outputs 4 grip actions (left hand, right hand, left foot, right foot), each in the range −1…1.

| Command range | Behaviour                                |
|---------------|------------------------------------------|
| < −0.4        | Release: detach from current hold        |
| −0.4 … 0.4   | Hold: keep current attachment, no search |
| > 0.4         | Grab: try to attach to the nearest hold  |

Distances are derived from `attach_radius` (default `0.22 m`):

- **Attach threshold**: `attach_radius × 0.7` — limb end-point must be within this to grab
- **Auto-release threshold**: `attach_radius × 1.2` — if an attached limb drifts beyond this, the hold is released

---

## Hold Field

Holds are generated procedurally and scroll with the climber.

| Parameter       | Config key          | Default | Effect                              |
|-----------------|---------------------|---------|-------------------------------------|
| Spacing (y)     | `hold_spacing_y`    | 0.35 m  | Vertical density of hold rows       |
| X jitter        | `hold_jitter_x`     | 0.8     | Horizontal spread around wall centre|
| Hold size       | `hold_size`         | 0.16 m  | Visual radius (does not affect grip) |
| Holds per row   | —                   | 8–13    | Sampled uniformly each row          |

A start cluster of 6 holds is always placed symmetrically within reach of the starting position.

---

## Stamina

Stamina starts at `100.0` and drains each step. When it hits `0.0` the episode ends.

Drain formula (per second):

```
drain = base_drain_per_sec
      + effort × movement_coeff
      + max(0, 2 − attachments) × tension_coeff × 10
```

| Parameter          | Config key            | Default |
|--------------------|-----------------------|---------|
| Base drain/sec     | `base_drain_per_sec`  | 0.65    |
| Movement cost      | `movement_coeff`      | 0.09    |
| Tension cost       | `tension_coeff`       | 0.06    |
| Fall penalty       | `fall_penalty`        | 8.0     |

Tension cost increases when fewer than 2 limbs are attached (hanging by one limb is costly).

---

## Reward

| Component       | Formula                                             | Config key               | Default |
|-----------------|-----------------------------------------------------|--------------------------|---------|
| Ascent          | `Δy × ascent_weight`                                | `ascent_weight`          | 6.0     |
| Energy penalty  | `−stamina_spent × energy_penalty_weight`            | `energy_penalty_weight`  | 0.02    |
| Contact bonus   | `new_contacts × contact_bonus`                      | `contact_bonus`          | 0.03    |
| Idle penalty    | `−idle_penalty` if `|Δy| < 0.001` and attached     | `idle_penalty`           | 0.002   |
| Fall penalty    | `−fall_penalty` on fall                             | `fall_penalty`           | 25.0    |

---

## Episode Parameters

| Parameter         | Config key         | Default | Effect                         |
|-------------------|--------------------|---------|--------------------------------|
| Simulation rate   | `dt`               | 1/30 s  | Physics timestep               |
| Max steps         | `max_steps`        | 1200    | ~40 s at 30 Hz                 |
| Ascent target     | `ascent_target`    | 20.0 m  | Height at which episode is won |
| Gravity           | `gravity`          | 9.81    | m/s²                           |
