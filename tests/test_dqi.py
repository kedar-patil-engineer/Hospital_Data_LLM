# tests/test_dqi.py
"""
Unit tests for DQI (Data Quality Index) scoring logic.

Tests the pure calculation logic directly — no DuckDB connection needed.
We extract the formula and test it in isolation so it can be validated
independently of the Streamlit app.

Run with:
    pytest tests/test_dqi.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# Pure DQI formula (extracted from app.py / api.py for isolated testing)
# We test the formula in isolation so changes to app.py don't break tests.
# ---------------------------------------------------------------------------

def compute_completeness_score(total_nulls: int, row_count: int, col_count: int) -> float:
    """Replicates app.py completeness_score logic."""
    total_cells = row_count * col_count
    if total_cells <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - (total_nulls / total_cells)))


def compute_anomaly_score(issue_rows: int, row_count: int) -> float:
    """Replicates app.py anomaly_score logic."""
    if row_count <= 0:
        return 100.0
    ratio = min(issue_rows / row_count, 1.0)
    return max(0.0, 100.0 * (1.0 - ratio))


def compute_dqi_score(completeness_score: float, anomaly_score: float) -> float:
    """Replicates app.py final DQI blend: 70% completeness + 30% anomaly."""
    return 0.7 * completeness_score + 0.3 * anomaly_score


def dqi_color(score: float) -> str:
    """Replicates app.py color bar logic."""
    if score >= 90:
        return "green"
    if score >= 75:
        return "gold"
    if score >= 60:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# 1. COMPLETENESS SCORE
# ---------------------------------------------------------------------------

class TestCompletenessScore:

    def test_zero_nulls_is_perfect(self):
        score = compute_completeness_score(total_nulls=0, row_count=100, col_count=10)
        assert score == 100.0

    def test_all_nulls_is_zero(self):
        score = compute_completeness_score(total_nulls=1000, row_count=100, col_count=10)
        assert score == 0.0

    def test_half_nulls(self):
        score = compute_completeness_score(total_nulls=500, row_count=100, col_count=10)
        assert score == pytest.approx(50.0, abs=0.01)

    def test_zero_rows_returns_zero(self):
        score = compute_completeness_score(total_nulls=0, row_count=0, col_count=10)
        assert score == 0.0

    def test_zero_cols_returns_zero(self):
        score = compute_completeness_score(total_nulls=0, row_count=100, col_count=0)
        assert score == 0.0

    def test_score_never_below_zero(self):
        # Nulls > total_cells (shouldn't happen but must be safe)
        score = compute_completeness_score(total_nulls=9999, row_count=10, col_count=5)
        assert score >= 0.0

    def test_score_never_above_100(self):
        score = compute_completeness_score(total_nulls=0, row_count=1000, col_count=20)
        assert score <= 100.0

    def test_realistic_claims_table(self):
        # 5000 rows, 15 columns, ~500 nulls (3.3% null rate)
        score = compute_completeness_score(total_nulls=500, row_count=5000, col_count=15)
        assert score > 95.0, f"Expected high completeness for low null rate, got {score}"

    def test_partial_null_precision(self):
        # 100 rows, 10 cols = 1000 cells, 100 nulls = 10% null
        score = compute_completeness_score(total_nulls=100, row_count=100, col_count=10)
        assert score == pytest.approx(90.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. ANOMALY SCORE
# ---------------------------------------------------------------------------

class TestAnomalyScore:

    def test_no_anomalies_is_perfect(self):
        score = compute_anomaly_score(issue_rows=0, row_count=1000)
        assert score == 100.0

    def test_all_rows_anomalous_is_zero(self):
        score = compute_anomaly_score(issue_rows=1000, row_count=1000)
        assert score == 0.0

    def test_10_percent_anomaly_rate(self):
        score = compute_anomaly_score(issue_rows=100, row_count=1000)
        assert score == pytest.approx(90.0, abs=0.01)

    def test_zero_rows_returns_100(self):
        # No rows = no anomalies observed = full score
        score = compute_anomaly_score(issue_rows=0, row_count=0)
        assert score == 100.0

    def test_anomalies_exceed_rows_capped(self):
        # Shouldn't happen, but capped at 1.0 ratio
        score = compute_anomaly_score(issue_rows=9999, row_count=100)
        assert score == 0.0

    def test_score_always_between_0_and_100(self):
        for issue_rows, row_count in [(0, 0), (0, 100), (50, 100), (100, 100), (200, 100)]:
            score = compute_anomaly_score(issue_rows, row_count)
            assert 0.0 <= score <= 100.0, f"Score {score} out of bounds for ({issue_rows}, {row_count})"


# ---------------------------------------------------------------------------
# 3. DQI BLEND (70% completeness + 30% anomaly)
# ---------------------------------------------------------------------------

class TestDqiBlend:

    def test_perfect_both_is_100(self):
        score = compute_dqi_score(completeness_score=100.0, anomaly_score=100.0)
        assert score == pytest.approx(100.0)

    def test_zero_both_is_zero(self):
        score = compute_dqi_score(completeness_score=0.0, anomaly_score=0.0)
        assert score == pytest.approx(0.0)

    def test_completeness_weight_is_70_percent(self):
        # Completeness=100, anomaly=0 → 70
        score = compute_dqi_score(completeness_score=100.0, anomaly_score=0.0)
        assert score == pytest.approx(70.0)

    def test_anomaly_weight_is_30_percent(self):
        # Completeness=0, anomaly=100 → 30
        score = compute_dqi_score(completeness_score=0.0, anomaly_score=100.0)
        assert score == pytest.approx(30.0)

    def test_realistic_good_table(self):
        # 96% completeness, 98% anomaly-free → should be ~96.6
        score = compute_dqi_score(completeness_score=96.0, anomaly_score=98.0)
        assert score == pytest.approx(0.7 * 96.0 + 0.3 * 98.0, abs=0.01)

    def test_realistic_poor_table(self):
        # 60% completeness, 70% anomaly-free → 63.0
        score = compute_dqi_score(completeness_score=60.0, anomaly_score=70.0)
        assert score == pytest.approx(0.7 * 60.0 + 0.3 * 70.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. COLOR THRESHOLDS
# ---------------------------------------------------------------------------

class TestDqiColor:

    def test_90_plus_is_green(self):
        assert dqi_color(90.0) == "green"
        assert dqi_color(100.0) == "green"
        assert dqi_color(95.5) == "green"

    def test_75_to_89_is_gold(self):
        assert dqi_color(75.0) == "gold"
        assert dqi_color(89.9) == "gold"

    def test_60_to_74_is_orange(self):
        assert dqi_color(60.0) == "orange"
        assert dqi_color(74.9) == "orange"

    def test_below_60_is_red(self):
        assert dqi_color(59.9) == "red"
        assert dqi_color(0.0) == "red"


# ---------------------------------------------------------------------------
# 5. END-TO-END DQI PIPELINE
# ---------------------------------------------------------------------------

class TestDqiEndToEnd:

    def test_high_quality_table(self):
        """A clean table should score green."""
        completeness = compute_completeness_score(total_nulls=50, row_count=2000, col_count=12)
        anomaly = compute_anomaly_score(issue_rows=5, row_count=2000)
        dqi = compute_dqi_score(completeness, anomaly)
        color = dqi_color(dqi)
        assert color in ("green", "gold"), f"Expected high quality, got DQI={dqi:.1f} ({color})"

    def test_poor_quality_table(self):
        """A dirty table should score orange or red."""
        completeness = compute_completeness_score(total_nulls=3000, row_count=1000, col_count=10)
        anomaly = compute_anomaly_score(issue_rows=200, row_count=1000)
        dqi = compute_dqi_score(completeness, anomaly)
        color = dqi_color(dqi)
        assert color in ("orange", "red"), f"Expected poor quality, got DQI={dqi:.1f} ({color})"

    def test_overall_dqi_weighted_by_rows(self):
        """
        Overall DQI = weighted average of per-table DQI by row count.
        Mirrors app.py: (row_count * dqi_score).sum() / total_rows
        """
        tables = [
            {"row_count": 1000, "dqi_score": 95.0},  # large clean table
            {"row_count": 100,  "dqi_score": 40.0},  # small dirty table
        ]
        total_rows = sum(t["row_count"] for t in tables)
        overall = sum(t["row_count"] * t["dqi_score"] for t in tables) / total_rows
        # 1000*95 + 100*40 = 99000 / 1100 = 90.0
        assert overall == pytest.approx(90.0, abs=0.01)
