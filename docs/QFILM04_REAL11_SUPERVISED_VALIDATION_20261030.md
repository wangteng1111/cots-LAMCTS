# Qfilm 0.4: supervised 11-prescription usability validation, 2026-10-30

## Scope and scientific claims

The requested matrix is **11 prescriptions × 3 radial full-frame fields (h=0,0.5,1) × 7 wavelengths (430,470,510,550,590,630,670 nm) = 231 monochromatic conditions**, each with actual CPU/CUDA execution where supported. Frame: 36×24 mm; principal detector pitch: 0.25 µm; initial PSF support: 0.64 mm (2560² detector pixels). Frequency samples: 5,10,15,20,30,40,50,60,70,80 lp/mm.

Files are pinned DeepLens public, product-associated/patent prescriptions, **not empirical measurements or certified factory prescriptions**. Source commit is `911bb9d334fce7e20a4407c4506ac683b52ff246`. Every file has a SHA256 in `validation/fixtures/qfilm04_real11_source_sha256.json`; file bytes must match it at execution. A successful numerical result still does not validate coatings, wavelength-dependent flux, manufacturing tolerances, physical sensor, stray light or measured lens MTF. Keep `training_eligible=false`, `physics_certified=false`, and `empirically_calibrated=false`.

Names: ef100mm_f2.8, ef35mm_f2.0, ef40mm_f2.8, ef50mm_f1.8, ef85mm_f1.8, rf16mm_f2.8, rf24mm_f1.8, rf35mm_f1.8, rf50mm_f1.8, sigma70mm_f2.8, yongnuo_50mm_f1.8.

## Runtime and queues

Bridge v0.6 actually started at **2026-10-30 00:02:48 Asia/Shanghai**, PID477557. Its runtime remains pinned at d22f69ba6ec69ae0921a24c125b701ad33714fc0. The new test runner does not require another service restart. Existing workers and checkpoint files are retained.

Three priority jobs remain in `remote/prepared/bridge-v06/`:

- 000-000-011-qfilm06-ef40-blue-20261029-01
- 000-000-012-qfilm06-ef40-green-20261029-01
- 000-000-013-qfilm06-rf50-red-20261029-01

The 11 new job IDs are `000-000-013-qfilm06-zz-real11-NN-NAME-20261030`, NN=00..10 in the name order above, under **remote/queue/**. They pass the existing explicit service include expression and the ordinary request validation. They do not expand or bypass the prepared-request whitelist. Each request pins an exact code commit, executes `validation/qfilm04_real11_supervised.py --mode complete --name NAME` through the workstation CUDA venv, has four CPU threads and a 48-hour independent worker deadline. Durable claims prohibit replay. No job is represented as running until an actual claim/result exists.

Prepared priority requests sort before ordinary queue requests. The deployed supervisor also refuses another full task while any pre-upgrade remediation worker is still alive. This protects the active blue/green computations. It can delay the new matrix; queued is not started or completed.

## Acceptance stages

1. Import and verify source bytes; prepare the corrected float64 wavefront at Npupil256. A failure is an explicit unsupported/error terminal for that sampled condition.
2. Run a small centered tile at **the same 0.25 µm pitch** on CPU and CUDA. Workstation default is 32² pixels (8 µm support); local CPU screening can use 16² (4 µm support). Record actual size/device and distinguish these from the full 0.64 mm support. Test complex OTF, raw intensity and raw integral differences ≤1e-6 and equal active-node counts. **Tiles are arithmetic diagnostics, never full PSFs or usable labels.**
3. Before costly full integration, enumerate necessary pupil/support/pitch/joint grid sets and their conservative `n² × detector_pixels²` cost using the same budget accounting as the solver. Test the required phase guards. Reject if no combination can possibly satisfy the independent verifier under the declared finite policy. This is a necessary-condition rejection, not a claim of convergence.
4. For an admitted condition with passed CPU/CUDA tile control, run the original full convergence solver; freeze the actual first-device grid sequence and replay on the second. Persist per-grid raw 2D `.npy` arrays with SHA256/size/shape manifests (binary arrays stay on the workstation); text checkpoints and native terminal evidence upload through Bridge.
5. Recompute convergence from native raw complex OTF and raw integral records using the independent verifier, including explicit wavelength, field, source hash, plan hash, actual distinct CPU/CUDA executions, and unchanged original tolerances. `full_condition_verified=true` requires all checks. Neither process exit code0 nor `passed` flags alone establishes usability.

Per-lens artifacts: `matrix-checkpoint.json`, `matrix-terminal.json`, `progress.json`, and `NAME-hH-wW/condition-terminal.json`. Admitted full solves also store `NAME-hH-wW-{c,g}-terminal.json`, the phase plan, raw manifests and a native summary under the condition directory. Partial/missing/failed/full-budget-rejected conditions have `J_task=null`; no old FFT score is substituted.

## Declared numerical policy and feasibility

Pupil candidates256,512,1024,2048,4096; supports0.64,0.96,1.28 mm; detector pitch0.25 µm plus0.125 µm verification; phase target0.25 cycles, alarm0.5. Pupil/support complex OTF tolerances0.002; pitch0.001; joint complex OTF/MTF0.005; relative raw integral0.005. CPU/GPU comparisons≤1e-6. Pair budget per device per condition: **150,000,000,000,000**. These criteria have not been loosened.

A necessary six-grid combination can cost at least **132,392,366,899,200** budget units even when the selected pupil is only512. Under the current150T budget, all affordable combinations select pupil512 on the0.96mm support and therefore require a phase-safe pupil256 at that support. Any selected pupil≥1024 cannot complete the required combined refinement within the current budget. A phase-unsafe pupil256 at0.96mm thus makes full independent acceptance impossible under this policy. Current EF40/550nm axis test fails this requirement (bound1.380916929 cycles versus0.25), before any full integration.

`POLICY_RESOURCE_INFEASIBLE` means this combination of solver/grid policy/budget cannot deliver a verified result. It is not evidence that a physical lens is bad, nor that increasing resources or changing the numerical integration method could never work. The new runner does not silently raise the budget, lower resolution or loosen error tolerances. Existing long-running priority jobs are preserved and may independently produce explicit resource/unconverged terminals.

## Monitoring and completion

Read `remote/agent_status.json`, `remote/health/bridge-v06-start.json`, and `remote/health/qfilm-three-verification.json` on workstation-results. Use payload observation timestamps rather than branch commit time. `running_job=null` does not include pre-upgrade adopted workers; consult latest native progress and the local adopted-process inventory when available. Only fresh progress at separate times establishes continuing advancement. Stale monitoring evidence indicates uncertainty, not proof of a stalled GPU.

An hourly ChatGPT notification task was requested but **creation failed: the account already has5 active tasks, the plan limit**. Do not describe notifications as enabled. Existing unrelated tasks were not changed. Workstation Bridge itself continues its local30-second adoption/verification loop and asynchronous publication. A user choice of an existing task to pause is needed to create the additional hourly notification task.

Usability conclusions must distinguish: successful source import; supported field/wavefront; arithmetic CPU/CUDA agreement; full numerical convergence; physical transport/reference validation; empirical calibration; practical runtime. Report the numerator and denominator for all231 conditions, and for lenses with all21 full conditions verified. A policy-infeasible or unsupported terminal is a completed diagnostic, not a passed full PSF test.
