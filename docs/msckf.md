# MSCKF derivation as implemented

This is the exact math in `insight_vio/msckf.py`, `insight_vio/imu.py` and
`insight_vio/triangulation.py`. Notation: `R_wb` rotates body (IMU) vectors into the
world frame, `[v]x` is the skew matrix, `Exp`/`Log` are the SO(3) maps, quaternions are
Hamilton `[w, x, y, z]`. World gravity `g = [0, 0, -9.81]`.

## 1. State and error state

Nominal state:

```
x = { R_wb, p_wb, v_wb, b_g, b_a ; (R_1, p_1), ..., (R_N, p_N) }
```

where `(R_i, p_i)` are clones of the IMU pose at the last N camera times.
Error state (dimension `15 + 6N`), **right/local** perturbation on rotations:

```
R = R̂ Exp(δθ)     p = p̂ + δp     v = v̂ + δv     b = b̂ + δb
δx = [δθ δp δv δb_g δb_a | δθ_1 δp_1 | ... | δθ_N δp_N]
```

The original MSCKF uses JPL quaternions with a global-ish error definition; we use
Hamilton + right perturbation to match Forster's preintegration, so every Jacobian
below differs in sign/frame from the 2007 paper but is equivalent.
`lie.jpl_to_hamilton` documents the conversion (`R_jpl(q) = R_ham(q)^T` for the same four
numbers, and `q1 ⊗_jpl q2 = q2 ⊗_ham q1`).

## 2. IMU model

```
ω_m = ω + b_g + n_g          a_m = R_wbᵀ (a_w − g) + b_a + n_a
ḃ_g = n_wg                   ḃ_a = n_wa
```

`n_*` are white with continuous-time densities σ_g, σ_a, σ_wg, σ_wa (Kalibr / EuRoC
units). A discrete sample over `dt` has variance `σ²/dt`.

## 3. Preintegration (Forster et al. 2017)

Between camera frames `i` and `j`, with the bias estimate held at `b̂`:

```
ΔR_{k+1} = ΔR_k Exp((ω_k − b̂_g) dt)
Δv_{k+1} = Δv_k + ΔR_k (a_k − b̂_a) dt
Δp_{k+1} = Δp_k + Δv_k dt + ½ ΔR_k (a_k − b̂_a) dt²
```

Noise `η = [δφ, δv, δp]` propagates as `η_{k+1} = A η_k + B n_k` with

```
A = [ ΔR_{k,k+1}ᵀ            0      0 ]       B = [ J_r(ω dt) dt     0          ]
    [ −ΔR_k [ã]x dt          I      0 ]           [ 0                ΔR_k dt    ]
    [ −½ ΔR_k [ã]x dt²       I dt   I ]           [ 0                ½ΔR_k dt²  ]
```

and bias Jacobians (used for first-order bias correction *and* in the filter's `F`):

```
J^p_a += J^v_a dt − ½ ΔR_k dt²          J^p_g += J^v_g dt − ½ ΔR_k [ã]x J^R_g dt²
J^v_a −= ΔR_k dt                        J^v_g −= ΔR_k [ã]x J^R_g dt
J^R_g  = ΔR_{k,k+1}ᵀ J^R_g − J_r dt
```

Tests: `tests/test_imu.py` checks the preintegrated deltas against ground truth, the
Jacobians against finite differences, and the covariance against a 400-run Monte Carlo.

## 4. Propagation

State prediction:

```
R_j = R_i ΔR      v_j = v_i + g T + R_i Δv      p_j = p_i + v_i T + ½ g T² + R_i Δp
```

Linearising (right perturbation, `R_i Δv ≈ R̂_i Δv − R̂_i [Δv]x δθ_i`):

```
δθ_j = ΔRᵀ δθ_i + J^R_g δb_g                                   + η_φ
δp_j = δp_i + T δv_i − R̂_i[Δp]x δθ_i + R̂_i(J^p_g δb_g + J^p_a δb_a) + R̂_i η_p
δv_j = δv_i − R̂_i[Δv]x δθ_i + R̂_i(J^v_g δb_g + J^v_a δb_a)        + R̂_i η_v
δb_j = δb_i + random walk
```

giving `F` (15×15) and `Q = G Σ_η Gᵀ + diag(0, 0, 0, σ_wg² T, σ_wa² T)`
(`msckf.propagation_jacobians`, checked against finite differences in
`tests/test_msckf.py`). Covariance:

```
P_II ← F P_II Fᵀ + Q        P_IC ← F P_IC        (clone blocks unchanged)
```

Simplification: the bias random walk's effect *inside* the interval on Δv, Δp is
ignored (standard for 20–30 Hz camera intervals).

## 5. State augmentation

At every image the current IMU pose is cloned: `J = ∂(δθ_c, δp_c)/∂δx` selects the
IMU `δθ, δp`, and

```
P ← [ P      P Jᵀ  ]
    [ J P    J P Jᵀ]
```

## 6. Measurement model

Camera `k` has extrinsics `(R_bc, p_bc)` (camera → body). For a feature at world
position `p_f` seen from clone `(R, p)`:

```
p_b = Rᵀ (p_f − p)          p_c = R_bcᵀ (p_b − p_bc)          z = [p_c.x/p_c.z, p_c.y/p_c.z] + n
```

`z` is in *normalized undistorted* coordinates (the front-end undistorts with the
hand-written radtan / equidistant inverse), noise `σ = σ_px / f`. Jacobians:

```
J_π = 1/z [[1, 0, −x/z], [0, 1, −y/z]]
∂z/∂δθ_c = J_π R_bcᵀ [p_b]x        ∂z/∂δp_c = −J_π R_bcᵀ Rᵀ        ∂z/∂p_f = J_π R_bcᵀ Rᵀ
```

Stereo simply adds rows from camera 1 at the same clone (with its own extrinsics), so
stereo and mono share one code path.

## 7. Triangulation

Gauss-Newton / Levenberg-Marquardt on the anchor inverse-depth parameters
`(α, β, ρ) = (x/z, y/z, 1/z)` in the first observing camera (`triangulation.py`):

```
h_i = R_i [α, β, 1]ᵀ + ρ t_i,     ẑ_i = h_i[:2] / h_i[2]
```

initialised with the linear midpoint-of-rays solution, rejected if any depth is out of
`[min_depth, max_depth]` or the RMS reprojection error is too large.

## 8. Null-space projection

Stacking all `M` observations: `r = H_x δx + H_f δp_f + n` (`2M` rows). The feature
error is unknown and correlated with the state, so we project onto the left null space
`A` of `H_f` (`2M × (2M−3)`, from a full QR of `H_f`):

```
r_o = Aᵀ r = Aᵀ H_x δx + Aᵀ n = H_o δx + n_o,     Cov(n_o) = σ² I   (A orthonormal)
```

## 9. Chi-squared gating

Per feature, with `S = H_o P H_oᵀ + σ² I`:

```
γ = r_oᵀ S⁻¹ r_o  <  χ²_{0.95}(2M − 3)      else reject the track as an outlier
```

`stats.chi2_quantile` is a hand-written incomplete-gamma inverse (no SciPy).

## 10. Update

All accepted features are stacked; if there are more rows than states the system is
compressed with a thin QR (`H = Q₁ T_H`, `r ← Q₁ᵀ r`). Then the standard EKF:

```
S = H P Hᵀ + σ² I,   K = P Hᵀ S⁻¹,   δx = K r,   P ← (I − K H) P (I − K H)ᵀ + K σ² I Kᵀ
```

The correction is applied with `R ← R Exp(δθ)` (IMU state and every clone), additive
elsewhere. Rotations are re-orthonormalised via SVD.

## 11. Feature / window management

* A track that is not observed in the newest frame and spans ≥ 2 clones is used and
  then discarded.
* When the window exceeds `max_clones`, every feature seen by the oldest clone is used
  (if it spans ≥ 2 clones), then the oldest clone is marginalised by deleting its rows
  and columns from `P` (exact for an EKF: marginalising a Gaussian).

## 12. Time offset and extrinsics

`t_imu = t_cam + time_offset` (Kalibr's `timeshift_cam_imu` convention) is applied to
every image timestamp before propagation. Extrinsics and `time_offset` are fixed,
configurable parameters; online estimation (adding `δT_bc`, `δt_d` to the state as in
Li & Mourikis 2013/2014) is on the roadmap.

## Known limitations (honest list)

* No first-estimate Jacobians (FEJ) / observability-constrained EKF: the filter is
  slightly over-confident in yaw over long runs.
* No SLAM features (long-lived landmarks in the state) - pure MSCKF drift only.
* No zero-velocity update; a headset left on a desk drifts slowly.
