import pytest

from backend.services.risk.risk_engine import calculate_risk


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "LOW"), (0.2999, "LOW"), (0.30, "MEDIUM"),
     (0.6999, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")]
)
def test_risk_bands(score, expected):
    assert calculate_risk(score)["risk_level"] == expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_invalid_probability(score):
    with pytest.raises(ValueError):
        calculate_risk(score)
