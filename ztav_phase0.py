#!/usr/bin/env python3
"""Phase-0 prototype for a context-aware Zero Trust engine for automated vehicles.

This educational baseline uses only Python's standard library. It demonstrates:

1. Multi-source security evidence fusion.
2. Reliability-aware continuous trust calculation.
3. Trust memory (EWMA) so one noisy reading does not cause unstable decisions.
4. Critical-failure caps for invalid identity or device posture.
5. Safety-aware policy decisions: allow, verify, restrict, deny, or safe fallback.

It is a research prototype, not production automotive safety/security software.

Run:
    python ztav_phase0.py
    python ztav_phase0.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping


class Decision(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP_VERIFY = "STEP_UP_VERIFY"
    RESTRICT = "RESTRICT"
    DENY = "DENY"
    SAFE_FALLBACK = "SAFE_FALLBACK"


@dataclass(frozen=True)
class Evidence:
    """One source's trust evidence.

    score: 0.0 is completely suspicious; 1.0 is fully trustworthy.
    quality: source freshness/availability/confidence in [0, 1].
    critical_failure: an explicit hard failure, such as an invalid certificate.
    """

    score: float
    quality: float = 1.0
    critical_failure: bool = False

    def __post_init__(self) -> None:
        for name, value in (("score", self.score), ("quality", self.quality)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class AccessRequest:
    subject: str
    resource: str
    action: str
    safety_critical: bool = False
    high_privilege: bool = False


@dataclass(frozen=True)
class TrustResult:
    instantaneous_trust: float
    continuous_trust: float
    risk: float
    critical_failures: tuple[str, ...]
    contributions: Mapping[str, float]


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    trust: float
    risk: float


DEFAULT_WEIGHTS: Dict[str, float] = {
    # Zero Trust identity and device posture.
    "identity": 0.15,
    "device_posture": 0.13,
    # In-vehicle communication behavior.
    "can_behavior": 0.20,
    # Cross-source physical plausibility and external collaboration.
    "gnss_imu_consistency": 0.17,
    "v2x_consistency": 0.10,
    "sensor_control_consistency": 0.20,
    # Timeliness prevents stale but otherwise plausible context from being reused.
    "freshness": 0.05,
}


@dataclass
class ContinuousTrustEngine:
    weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    memory: float = 0.65
    initial_trust: float = 0.80
    identity_failure_cap: float = 0.25
    posture_failure_cap: float = 0.35
    _trust: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory < 1.0:
            raise ValueError("memory must be in [0, 1)")
        if not self.weights or any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("all source weights must be positive")
        self._trust = self.initial_trust

    def reset(self, trust: float | None = None) -> None:
        self._trust = self.initial_trust if trust is None else trust

    def evaluate(self, evidence: Mapping[str, Evidence]) -> TrustResult:
        missing = set(self.weights) - set(evidence)
        if missing:
            raise ValueError(f"missing evidence sources: {sorted(missing)}")

        weighted_sum = 0.0
        active_weight = 0.0
        contributions: Dict[str, float] = {}

        for source, weight in self.weights.items():
            item = evidence[source]
            effective_weight = weight * item.quality
            contribution = effective_weight * item.score
            contributions[source] = contribution
            weighted_sum += contribution
            active_weight += effective_weight

        # Very low data quality is itself risky; unavailable sources must not
        # produce a falsely confident score by simply disappearing.
        total_weight = sum(self.weights.values())
        coverage = active_weight / total_weight
        fused = weighted_sum / active_weight if active_weight else 0.0
        instantaneous = fused * coverage

        failures = tuple(
            source for source, item in evidence.items() if item.critical_failure
        )
        if evidence["identity"].critical_failure:
            instantaneous = min(instantaneous, self.identity_failure_cap)
        if evidence["device_posture"].critical_failure:
            instantaneous = min(instantaneous, self.posture_failure_cap)

        updated = self.memory * self._trust + (1.0 - self.memory) * instantaneous

        # A verified hard failure takes effect immediately; it should not be
        # hidden by the historical trust accumulated before compromise.
        if evidence["identity"].critical_failure:
            updated = min(updated, self.identity_failure_cap)
        if evidence["device_posture"].critical_failure:
            updated = min(updated, self.posture_failure_cap)

        self._trust = max(0.0, min(1.0, updated))
        return TrustResult(
            instantaneous_trust=instantaneous,
            continuous_trust=self._trust,
            risk=1.0 - self._trust,
            critical_failures=failures,
            contributions=contributions,
        )


@dataclass(frozen=True)
class SafetyAwarePolicyEngine:
    allow_threshold: float = 0.75
    privileged_threshold: float = 0.85
    restrict_threshold: float = 0.50

    def decide(self, request: AccessRequest, trust: TrustResult) -> PolicyResult:
        value = trust.continuous_trust
        failures = ", ".join(trust.critical_failures)

        if trust.critical_failures:
            if request.safety_critical:
                return PolicyResult(
                    Decision.SAFE_FALLBACK,
                    f"critical failure ({failures}); use trusted local controller/minimal-risk behavior",
                    value,
                    trust.risk,
                )
            return PolicyResult(
                Decision.DENY,
                f"critical failure ({failures}); isolate subject and require re-attestation",
                value,
                trust.risk,
            )

        required = self.privileged_threshold if request.high_privilege else self.allow_threshold
        if value >= required:
            return PolicyResult(Decision.ALLOW, "trust satisfies the action threshold", value, trust.risk)

        if request.safety_critical:
            return PolicyResult(
                Decision.SAFE_FALLBACK,
                "insufficient trust for the requested actuator command; preserve a minimal-risk local function",
                value,
                trust.risk,
            )

        if value >= self.allow_threshold and request.high_privilege:
            return PolicyResult(
                Decision.STEP_UP_VERIFY,
                "ordinary trust is adequate, but this privileged action needs stronger verification",
                value,
                trust.risk,
            )

        if value >= self.restrict_threshold:
            return PolicyResult(
                Decision.RESTRICT,
                "permit only low-risk telemetry/read operations while collecting more evidence",
                value,
                trust.risk,
            )

        return PolicyResult(
            Decision.DENY,
            "trust is below the minimum threshold; quarantine the subject",
            value,
            trust.risk,
        )


def make_evidence(**changes: Evidence) -> Dict[str, Evidence]:
    """Create a healthy observation and override sources for an attack scenario."""

    normal = {
        "identity": Evidence(0.98),
        "device_posture": Evidence(0.95),
        "can_behavior": Evidence(0.94),
        "gnss_imu_consistency": Evidence(0.96),
        "v2x_consistency": Evidence(0.90),
        "sensor_control_consistency": Evidence(0.95),
        "freshness": Evidence(0.98),
    }
    unknown = set(changes) - set(normal)
    if unknown:
        raise ValueError(f"unknown evidence sources: {sorted(unknown)}")
    normal.update(changes)
    return normal


def scenario_stream() -> Iterable[tuple[str, Mapping[str, Evidence]]]:
    # Three healthy windows establish normal behavior.
    for _ in range(3):
        yield "normal", make_evidence()

    # GNSS disagrees with IMU/vehicle motion and neighboring V2X reports.
    for _ in range(3):
        yield "gps_spoofing", make_evidence(
            gnss_imu_consistency=Evidence(0.05),
            v2x_consistency=Evidence(0.30),
            sensor_control_consistency=Evidence(0.65),
        )

    # CAN commands conflict with the vehicle's observed physical response.
    for _ in range(3):
        yield "can_injection", make_evidence(
            can_behavior=Evidence(0.05),
            sensor_control_consistency=Evidence(0.15),
        )

    # A compromised ECU fails attestation. This is a hard Zero Trust failure.
    yield "compromised_ecu", make_evidence(
        identity=Evidence(0.10, critical_failure=True),
        device_posture=Evidence(0.05, critical_failure=True),
        can_behavior=Evidence(0.25),
    )

    # Recovery requires repeated healthy evidence; trust is not instantly restored.
    for _ in range(5):
        yield "recovery", make_evidence()


def run_demo() -> None:
    trust_engine = ContinuousTrustEngine()
    policy_engine = SafetyAwarePolicyEngine()
    request = AccessRequest(
        subject="steering_ecu",
        resource="steering_actuator",
        action="set_steering_angle",
        safety_critical=True,
        high_privilege=True,
    )

    print("Multi-Source Context-Aware Zero Trust — Phase 0")
    print("Research simulation only; not for deployment in a real vehicle.\n")
    print(f"{'tick':>4}  {'scenario':<17} {'instant':>8} {'trust':>8} {'risk':>8}  decision")
    print("-" * 76)

    for tick, (scenario, evidence) in enumerate(scenario_stream(), start=1):
        trust = trust_engine.evaluate(evidence)
        decision = policy_engine.decide(request, trust)
        print(
            f"{tick:>4}  {scenario:<17} "
            f"{trust.instantaneous_trust:>8.3f} "
            f"{trust.continuous_trust:>8.3f} "
            f"{trust.risk:>8.3f}  {decision.decision.value}"
        )


def self_test() -> None:
    engine = ContinuousTrustEngine()
    policy = SafetyAwarePolicyEngine()
    ordinary = AccessRequest("v2x_unit", "telemetry", "read")
    critical = AccessRequest("steering_ecu", "steering", "write", safety_critical=True)

    healthy = engine.evaluate(make_evidence())
    assert healthy.continuous_trust > 0.80
    assert policy.decide(ordinary, healthy).decision == Decision.ALLOW

    for _ in range(4):
        attacked = engine.evaluate(make_evidence(
            can_behavior=Evidence(0.0),
            sensor_control_consistency=Evidence(0.0),
            gnss_imu_consistency=Evidence(0.2),
        ))
    assert attacked.continuous_trust < policy.allow_threshold
    assert policy.decide(critical, attacked).decision == Decision.SAFE_FALLBACK

    failed = engine.evaluate(make_evidence(
        identity=Evidence(0.0, critical_failure=True),
    ))
    assert failed.continuous_trust <= engine.identity_failure_cap
    assert policy.decide(ordinary, failed).decision == Decision.DENY

    print("All Phase-0 self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    args = parser.parse_args()
    self_test() if args.self_test else run_demo()


if __name__ == "__main__":
    main()
