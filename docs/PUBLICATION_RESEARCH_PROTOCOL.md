# Publication Research Protocol

## Multi-Source Context-Aware Zero Trust Security for Self-Driving Vehicles

Version: 1.0 pre-freeze protocol  
Intended outputs: Master's thesis and IEEE/SCIE journal manuscript  
Project status: research prototype; not production automotive safety software

## 1. Artifact-protection rule

The existing datasets, source scripts, trained models, thresholds, manifests and
result files are historical research evidence. They must not be overwritten,
deleted, renamed or silently regenerated. All remaining experiments use new
versioned scripts and new result directories. Step 31 is the final freeze and
must be run only after Steps 30A-30E are completed and reviewed.

Failed experiments are retained. In particular, the ROAD external results from
Steps 28-30 are negative external-validation evidence, not files to remove or
results to hide.

## 2. Research problem

An autonomous vehicle receives security-relevant evidence from heterogeneous
sources such as the in-vehicle CAN bus, GNSS, V2X communications, device or ECU
identity, and physical vehicle state. A single detector may be accurate in its
training domain but unreliable under sparse attacks, sensor compromise,
missing data or cross-vehicle domain shift. The project investigates whether a
graded Zero Trust policy can continuously combine these sources and select a
safety-aware action without granting implicit trust to any single source.

## 3. Proposed contribution

The proposed contribution is a research architecture with four layers:

1. **Source evidence:** CAN anomaly score, GNSS consistency, V2X consistency,
   identity integrity and vehicle-state consistency.
2. **Context and memory:** source-quality checks, drift indicators, temporal
   persistence and recovery evidence.
3. **Continuous trust:** an interpretable trust score plus source-attribution
   record rather than a single opaque binary classification.
4. **Graded enforcement:** `ALLOW`, `VERIFY`, `RESTRICT` and `SAFE_FALLBACK`,
   with re-verification and recovery rules.

The architecture follows the NIST Zero Trust principle that no subject, device
or source receives implicit trust based only on location or ownership, and that
policy decisions use continually evaluated evidence.

## 4. Research questions

- **RQ1:** Does multi-source context improve end-to-end attack detection over
  CAN-only and context-only baselines under leakage-safe evaluation?
- **RQ2:** Does graded persistent enforcement provide a better
  security-availability trade-off than instant or hard-guard decisions?
- **RQ3:** How robust is the method to sparse attacks, source loss, stale data,
  source compromise and conflicting evidence?
- **RQ4:** How well does the method generalize across datasets, captures,
  attacks and operational domains?
- **RQ5:** What latency, CPU and memory costs are introduced by continuous
  multi-source trust evaluation?

## 5. Predeclared hypotheses

- **H1:** The proposed multi-source policy has higher paired end-to-end F1 than
  the strongest single-source baseline on the confirmation runs.
- **H2:** The persistent graded policy maintains mean false-positive rate at or
  below 0.05 while preserving end-to-end F1 of at least 0.90 in each declared
  attack-density condition.
- **H3:** Temporal persistence improves sparse-CAN attack recall relative to
  the frozen 100-frame CAN gate without violating the 0.05 mean enforcement-FPR
  constraint.
- **H4:** Loss of any one non-CAN context source causes no more than a 0.10
  absolute reduction in end-to-end F1 relative to the complete proposed policy.
- **H5:** A compromised source cannot independently force `ALLOW`; conflicting
  high-risk evidence must result in `VERIFY`, `RESTRICT` or `SAFE_FALLBACK`.

H1-H5 are evaluated only on the new predeclared confirmation protocol. Results
already inspected during development remain development or exploratory
evidence and are labelled accordingly.

## 6. Threat model

### Protected assets

- safe vehicle control and actuation;
- authenticity and integrity of CAN messages;
- GNSS, V2X and identity evidence;
- availability of legitimate vehicle operation;
- auditability of security decisions.

### Adversary capabilities

- inject, replay, spoof or falsify CAN traffic;
- spoof GNSS or falsify V2X context;
- compromise one identity or context source;
- create sparse, slow or intermittent attacks;
- poison a startup baseline;
- cause source loss, delay or stale observations.

### Assumptions

- cryptographic roots of trust and secure hardware are outside the experimental
  implementation unless explicitly modelled;
- the adversary does not alter the offline evaluation code or ground-truth
  files;
- no single sensor or detector is assumed trustworthy by default;
- this work evaluates detection and policy behaviour, not vehicle homologation
  or safety certification.

## 7. Dataset roles and leakage control

| Dataset/source | Role | Permitted use |
|---|---|---|
| CICIoV2024 | Primary CAN development and internal confirmation | Group-disjoint feature-signature splits and frozen confirmation |
| SUMO | Reproducible GNSS/V2X/identity/vehicle-state context | Multi-source replay, controlled attacks and recovery experiments |
| HCRL/Car-Hacking | External CAN-domain evaluation | External calibration/holdout and domain-shift analysis as declared |
| ROAD | Secondary external vehicle/capture evaluation | Frozen zero-shot confirmation and documented negative result |

Leakage controls:

- identical CAN signatures must not cross training, validation and test;
- capture/session identity must not cross development and confirmation;
- thresholds are chosen on training/validation or declared calibration data;
- test labels must not be used for model fitting or threshold selection;
- the unit of statistical inference is the independent run, seed, capture or
  attack session, not each overlapping window;
- all post-hoc ROAD experiments are labelled development diagnostics and cannot
  restore the status of the original frozen confirmation.

## 8. Outcomes

### Primary outcomes

- paired end-to-end F1 of the proposed policy versus the strongest eligible
  baseline on confirmation runs;
- mean enforcement false-positive rate on healthy/recovery phases.

### Secondary outcomes

- precision, recall, balanced accuracy, MCC, PR-AUC and ROC-AUC;
- per-attack-family and per-density recall;
- false-negative rate and worst-case capture performance;
- detection latency in windows and milliseconds;
- `ALLOW`, `VERIFY`, `RESTRICT` and `SAFE_FALLBACK` distribution;
- recovery time and healthy-recovery FPR;
- source attribution accuracy;
- CPU time, throughput, peak memory and model size.

### Mandatory subgroup reporting

- representative, low (1-5), medium (6-20) and high (21-100) malicious-frame
  densities;
- each dataset and attack family;
- each source-loss and compromised-source condition;
- each confirmation seed or capture;
- macro, pooled and worst-capture false-positive rates.

## 9. Statistical analysis plan

1. Preserve paired runs across methods using the same seed, attack source and
   density condition.
2. Report the mean, standard deviation, median, interquartile range and 95%
   confidence interval for run-level outcomes.
3. Use clustered bootstrap confidence intervals with run/capture as the
   resampling unit; do not treat overlapping windows as independent.
4. Compare the proposed method with each predeclared baseline using a paired
   permutation test or Wilcoxon signed-rank test when distributional assumptions
   are not justified.
5. Report absolute effect size (`proposed - baseline`) and relative change.
6. Apply Holm correction within each family of multiple baseline comparisons.
7. Report exact sample sizes, missing runs and exclusions.
8. Keep negative and non-significant results in the thesis and manuscript.

Statistical significance does not replace practical requirements. FPR,
worst-case recall, latency and recovery behaviour are reported independently of
p-values.

## 10. Remaining pre-freeze stages

### Step 30A — Publication-readiness audit

Read-only inventory of existing datasets, code, models, metrics, validation
stages and missing publication evidence. It writes only to a new audit result
directory.

### Step 30B — Statistical confirmation

- construct one run-level comparison table;
- calculate clustered 95% confidence intervals;
- perform paired tests and Holm correction;
- produce effect-size and uncertainty plots;
- keep development and confirmation results separate.

### Step 30C — Source robustness and policy safety

- remove each context source individually;
- introduce stale and delayed evidence;
- compromise one source at a time;
- create conflicting-source cases;
- verify that one source cannot independently force `ALLOW`;
- quantify F1 degradation, unsafe-allow rate and fallback availability cost.

### Step 30D — Efficiency and deployability

- cold-start and warm-start measurements;
- per-window median, p95 and p99 inference latency;
- throughput;
- CPU utilization;
- process peak memory;
- model and evidence-buffer size;
- results separated by method and window scale.

The experiment reports measurements on the actual research computer. It does
not claim embedded-ECU performance without testing embedded hardware.

### Step 30E — Untouched final confirmation

- lock models, thresholds, rules and analysis code before execution;
- use confirmation seeds/captures not used in Steps 30B-30D for tuning;
- run the complete proposed policy and all declared baselines;
- evaluate H1-H5 once;
- record failures without post-hoc retuning.

### Step 31 — Final freeze

Run `31_freeze_final_zero_trust_policy.py` only after Steps 30A-30E pass review.
The freeze packages the selected policy, evidence inventory, hashes, supported
claims, negative external results and limitations.

## 11. Publication claims and boundaries

### Claims that may be supported

- multi-source evidence improved the evaluated security-availability trade-off
  under the declared CICIoV-SUMO protocol;
- leakage-safe and group-disjoint splitting materially changed the credibility
  of CAN results;
- graded policy decisions enabled explicit verification, restriction and safe
  fallback behaviour;
- external ROAD testing exposed a cross-vehicle generalization limitation.

### Claims that must not be made

- universal zero-shot detection across all vehicles or CAN schemas;
- production readiness, safety certification or compliance certification;
- protection against every attacker or compromised combination of sources;
- causal conclusions from a non-causal dataset experiment;
- independent confirmation for any result tuned on the same confirmation data.

## 12. Standards positioning

- NIST SP 800-207, Zero Trust Architecture:
  https://csrc.nist.gov/pubs/sp/800/207/final
- ISO/SAE 21434:2021, Road vehicles — Cybersecurity engineering:
  https://www.iso.org/standard/70918.html
- UNECE UN Regulation No. 155, Cybersecurity and Cybersecurity Management
  System:
  https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security

The thesis may align its threat analysis and evidence lifecycle with these
documents, but must not claim formal certification or regulatory compliance.

## 13. Thesis structure

1. Introduction and research objectives
2. Background: autonomous-vehicle security and Zero Trust
3. Literature review and research gap
4. Threat model and proposed multi-source architecture
5. Data, preprocessing and leakage-safe methodology
6. Experimental design and statistical analysis
7. Results
8. Robustness, external validation and falsification
9. Discussion, limitations and threats to validity
10. Conclusion and future work

## 14. Manuscript positioning

The journal manuscript should focus on one central contribution: a graded,
multi-source, context-aware Zero Trust policy evaluated with leakage-safe,
cross-domain and falsification protocols. The Master's thesis may contain the
complete engineering history and all supporting experiments; the journal paper
should present the final protocol, strongest baselines, ablations, statistical
evidence, external results and limitations.

