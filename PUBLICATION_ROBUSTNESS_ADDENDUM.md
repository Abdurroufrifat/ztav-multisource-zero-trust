# Publication Robustness Addendum

Version: 1.0 post-development, pre-confirmation

Date: 2026-07-14

Applies to: `PUBLICATION_RESEARCH_PROTOCOL.md` version 1.0

## Purpose

Step 30C was a development falsification audit. It found that the frozen Step 25
policy did not meet the predeclared H4 single-context-source-loss margin for
GNSS, V2X, or identity evidence. It also found that the policy safely handled a
detected source compromise and conflicting high-risk evidence, but could not
handle a Byzantine source that falsely reported healthy while exposing no
independent integrity, availability, freshness, or corroboration warning.

This addendum does not change H1-H5, any model, any threshold, or any previous
result. It locks the interpretation of those development results before the
untouched Step 30E confirmation.

## Locked interpretation

1. H4 remains unsupported in development for GNSS, V2X, and identity loss. The
   negative result must be reported even if Step 30E later produces a different
   estimate.
2. The detected-compromise and conflicting-evidence portions of H5 are
   supported only when the relevant quality or conflict condition is observable.
3. The universal form of H5 is not supported. A perfectly false-healthy source
   can be observationally identical to a genuinely healthy source under the
   current Step 25 input interface.
4. Treating every no-alarm row as compromised would not solve this limitation;
   it would reject ordinary benign operation and invalidate the availability
   objective.
5. No post-hoc threshold tuning or relabelling is permitted before Step 30E.

## Architecture requirement for a future policy version

A future policy may claim tolerance to source loss or compromise only after it
adds and independently validates explicit source-quality evidence, including:

- authenticated source identity and attestation status;
- heartbeat availability and freshness age;
- replay and monotonic-counter checks;
- independent cross-source plausibility or quorum evidence;
- separate attack, source-quality, and enforcement-action outputs.

Known source-quality failure should trigger a graded `VERIFY`, `RESTRICT`, or
`SAFE_FALLBACK` action without automatically being counted as an attack label.
An undetected false-healthy source remains outside the supported claim unless
independent evidence makes that condition observable.

## Confirmation and publication rule

Step 30E must evaluate the original frozen hypotheses once on untouched units
and retain all failures. The thesis and paper must distinguish:

- detection performance;
- source-quality observability;
- enforcement safety;
- availability cost; and
- unsupported universal or production-safety claims.

Step 31 remains blocked until Step 30E and the final reproducibility review are
complete.
