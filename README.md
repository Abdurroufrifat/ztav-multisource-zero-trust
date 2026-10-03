# Multi-Source Context-Aware Zero Trust Security for Self-Driving Vehicles

Research code and reproducibility materials for a **multi-source context-aware Zero Trust security framework for autonomous vehicles**.

This repository contains the **July / Paper 1 project only**. It belongs to the original `D:\ztav_project` research line and must remain separate from later Paper 2 work.

> **Research prototype:** This repository is for academic research and experimentation. It is not production automotive safety software and is not intended for vehicle certification or deployment.

## Research Problem

Autonomous vehicles depend on several security-relevant sources, including in-vehicle CAN traffic, GNSS, V2X context, ECU or device identity, vehicle-state consistency, and source freshness or availability.

A detector that performs well in its training domain can still fail under domain shift, sparse attacks, stale observations, source loss, spoofing, or previously unseen attack patterns. This project studies whether continuously combining multiple sources can support more reliable Zero Trust decisions.

## Core Architecture

The research framework contains four main layers:

1. **Source evidence** - CAN anomaly evidence, GNSS consistency, V2X consistency, identity integrity, vehicle-state consistency, and source-quality information.
2. **Context and temporal memory** - temporal persistence, recovery behavior, drift, missing context, sparse attacks, conflicting evidence, and startup-baseline poisoning.
3. **Continuous trust** - interpretable trust and risk scores with reliability-aware evidence fusion and source contribution tracking.
4. **Graded enforcement** - security actions such as `ALLOW`, `VERIFY`, `RESTRICT`, and `SAFE_FALLBACK` instead of a single binary decision.

The early `ztav_phase0.py` file provides a lightweight standard-library prototype of the continuous-trust concept.

## Research Questions

The publication protocol evaluates:

- whether multi-source context improves end-to-end detection over CAN-only and context-only baselines;
- whether persistent graded enforcement improves the security and availability trade-off;
- robustness to sparse attacks, source loss, stale data, compromised sources, and conflicting evidence;
- generalization across datasets, captures, attacks, and operational domains;
- runtime, CPU, memory, and latency cost.

See `docs/PUBLICATION_RESEARCH_PROTOCOL.md` for the full predeclared protocol.

## Data and Experimental Sources

| Source | Role |
| --- | --- |
| **CICIoV2024** | Primary CAN development and internal evaluation |
| **SUMO** | Controlled GNSS, V2X, identity, vehicle-state, and attack-context simulation |
| **HCRL / Car-Hacking** | External CAN-domain and domain-shift evaluation |
| **ROAD** | Secondary external vehicle/capture evaluation and zero-shot testing |
| **GEM-CAN** | Additional locked external-confirmation workflow |

Large datasets, trained models, generated results, and SUMO artifacts are intentionally excluded from Git.

The `.gitignore` excludes:

```text
data/
models/
results/
sumo/
```

## Repository Structure

```text
ztav-multisource-zero-trust/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- ztav_phase0.py
|
|-- docs/
|   `-- PUBLICATION_RESEARCH_PROTOCOL.md
|
|-- src/
|   |-- 01_inspect_ciciov2024.py
|   |-- 02_audit_and_split_ciciov2024.py
|   |-- 03_build_window_dataset.py
|   |-- 04_train_binary_baselines.py
|   |-- 05_stress_test_binary_generalization.py
|   |-- 06_group_disjoint_binary_evaluation.py
|   |-- 07_build_sumo_context_testbed.py
|   |-- 08_run_sumo_attack_experiments.py
|   |-- 09_evaluate_context_aware_zero_trust.py
|   |-- 10_repeated_seed_and_sensitivity.py
|   |-- ...
|   |-- 30A_publication_readiness_audit.py
|   |-- 30B_publication_statistical_analysis.py
|   |-- 30C_publication_source_robustness.py
|   |-- 30D_publication_efficiency_benchmark.py
|   |-- 30E_untouched_final_confirmation.py
|   |-- 30F_gem_can_locked_external_confirmation.py
|   |-- 30G_final_publication_readiness_review.py
|   |-- 31_freeze_final_zero_trust_policy.py
|   `-- 32_build_thesis_publication_package.py
|
|-- PUBLICATION_ROBUSTNESS_ADDENDUM.md
|-- road_attack_metadata.json
|-- road_signal_attack_metadata.json
|-- road_signal_file_manifest.csv
`-- road_signal_schema_samples.txt
```

The numbered scripts preserve the development and evaluation sequence for reproducibility.

## Main Experimental Stages

### Phase 0 - Zero Trust Prototype

Run:

```powershell
python ztav_phase0.py
```

Run the self-test:

```powershell
python ztav_phase0.py --self-test
```

### CICIoV2024 CAN Pipeline

The first stages inspect and audit CICIoV2024, create leakage-aware splits and windowed features, train binary baselines, and evaluate generalization.

Representative scripts:

```text
01_inspect_ciciov2024.py
02_audit_and_split_ciciov2024.py
03_build_window_dataset.py
04_train_binary_baselines.py
05_stress_test_binary_generalization.py
06_group_disjoint_binary_evaluation.py
```

### Multi-Source SUMO Evaluation

Later stages introduce reproducible context and simulated attacks:

```text
07_build_sumo_context_testbed.py
08_run_sumo_attack_experiments.py
09_evaluate_context_aware_zero_trust.py
```

Further stages evaluate repeated seeds, sensitivity, attack severity, hybrid CICIoV/SUMO replay, feature-family ablation, drift-aware CAN trust gates, session normalization, and temporal persistence.

### External Validation

The repository contains external and domain-shift studies using HCRL/Car-Hacking, ROAD, and GEM-CAN workflows.

Negative external-validation results are retained instead of removed. This is an explicit reproducibility rule of the project.

### Publication Confirmation and Freeze

Important later-stage scripts include:

```text
30A_publication_readiness_audit.py
30B_publication_statistical_analysis.py
30C_publication_source_robustness.py
30C2_publication_observability_claim_audit.py
30D_publication_efficiency_benchmark.py
30D2_publication_resource_completion.py
30E0_hcrl_confirmation_capacity_audit.py
30E_untouched_final_confirmation.py
30F0_gem_can_schema_capacity_audit.py
30F1_freeze_gem_can_confirmation_protocol.py
30F_gem_can_locked_external_confirmation.py
30G_final_publication_readiness_review.py
31_freeze_final_zero_trust_policy.py
32_build_thesis_publication_package.py
```

`31_freeze_final_zero_trust_policy.py` records the selected research policy and hashes supporting evidence rather than retraining or recalibrating detectors.

`32_build_thesis_publication_package.py` creates a manuscript-ready evidence package from the frozen result set.

## Leakage and Reproducibility Controls

The research protocol includes controls intended to reduce optimistic evaluation:

- identical CAN signatures should not cross training, validation, and test partitions;
- capture or session identity should not cross development and confirmation when disjoint evaluation is required;
- thresholds should be selected from training, validation, or declared calibration data;
- test labels should not be used for model fitting or threshold selection;
- overlapping windows should not be treated as statistically independent observations;
- development diagnostics and confirmation results should remain separate;
- failed or negative experiments should remain part of the research record.

Historical scripts and metadata are kept so that the research path remains auditable.

## Source-Robustness Limitation

The project records limitations found during development.

The robustness audit found that some source-loss conditions did not satisfy the predeclared robustness margin. It also identified an observability limit: a compromised source that falsely reports itself as healthy cannot necessarily be distinguished from a genuinely healthy source unless independent integrity, freshness, availability, attestation, or cross-source evidence makes the failure observable.

These findings are documented in `PUBLICATION_ROBUSTNESS_ADDENDUM.md` and are retained rather than hidden through post-hoc tuning.

## Windows Setup

Clone the repository:

```powershell
git clone https://github.com/Abdurroufrifat/ztav-multisource-zero-trust.git
cd ztav-multisource-zero-trust
```

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

The Phase-0 prototype uses only the Python standard library. Later stages require additional scientific Python and simulation dependencies depending on the experiment.

A typical local research workspace contains:

```text
ztav-multisource-zero-trust/
|-- data/
|-- models/
|-- results/
|-- sumo/
|-- src/
`-- ...
```

The ignored directories must be restored from the corresponding research artifacts before rerunning later numbered stages.

## Publication Evidence

The project separates evidence into detection performance, source-quality observability, Zero Trust enforcement behavior, availability cost, external-domain behavior, statistical uncertainty, runtime/resource cost, and failed or unsupported hypotheses.

The final packaging stage prepares frozen evidence for a Master's thesis and journal manuscript without silently modifying earlier experimental results.

## Important Scope Note

This repository represents the original **multi-source context-aware Zero Trust autonomous-vehicle research project developed in the July / Paper 1 line**.

It does **not** contain the separate later Paper 2 project or its development workspace.

Do not mix artifacts, results, scripts, evidence directories, or publication claims between the two projects.

## Author

**Abdur Rouf**

GitHub: [Abdurroufrifat](https://github.com/Abdurroufrifat)

Repository: [ztav-multisource-zero-trust](https://github.com/Abdurroufrifat/ztav-multisource-zero-trust)

## License

This repository is distributed under the **MIT License**.

See [LICENSE](LICENSE) for the complete license text.

Datasets, third-party software, external research artifacts, and simulation resources remain subject to their respective licenses and terms.

## Disclaimer

This project is an academic research prototype for cybersecurity experimentation.

It should not be used as a production vehicle-security controller, automotive safety mechanism, certification system, or substitute for validated safety and cybersecurity engineering processes.
