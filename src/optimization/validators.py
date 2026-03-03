"""
Validation modules for overfitting detection and parameter stability.

These validators ensure strategies are robust and not overfit to historical data.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from src.utils.logger import Logger

logger = Logger(__name__)


@dataclass
class StabilityResult:
    """Results from stability testing."""

    is_stable: bool
    perturbation_scores: List[float]
    original_score: float
    mean_perturbed: float
    std_perturbed: float
    coefficient_of_variation: float
    max_degradation: float


class StabilityValidator:
    """
    Test parameter stability through perturbation analysis.

    Perturbs each parameter by +/-10% and validates that performance
    doesn't degrade significantly, indicating robustness.
    """

    def __init__(
        self,
        perturbation_pct: float = 0.10,
        max_degradation: float = 0.15,
        num_trials: int = 5,
    ):
        """
        Initialize stability validator.

        Args:
            perturbation_pct: Percentage to perturb parameters
            max_degradation: Maximum allowed performance degradation
            num_trials: Number of perturbation trials per parameter
        """
        self.perturbation_pct = perturbation_pct
        self.max_degradation = max_degradation
        self.num_trials = num_trials

        logger.info(
            f"StabilityValidator: perturbation={perturbation_pct:.0%}, "
            f"max_degradation={max_degradation:.0%}, trials={num_trials}"
        )

    def test_stability(
        self,
        parameters: Dict[str, float],
        evaluation_function: Callable[[Dict[str, float]], float],
        original_score: float,
    ) -> StabilityResult:
        """
        Test parameter stability through systematic perturbation.

        Args:
            parameters: Dictionary of parameter names and values
            evaluation_function: Function that evaluates parameters and returns fitness
            original_score: Original fitness score with unperturbed parameters

        Returns:
            StabilityResult with test outcomes
        """
        logger.info(f"Testing stability for {len(parameters)} parameters")

        perturbed_scores = []

        # Test each parameter
        for param_name, param_value in parameters.items():
            if not isinstance(param_value, (int, float)):
                logger.debug(f"Skipping non-numeric parameter: {param_name}")
                continue

            # Try both positive and negative perturbations
            for direction in [-1, 1]:
                for trial in range(self.num_trials):
                    # Create perturbed parameters
                    perturbed_params = parameters.copy()
                    perturbation = (
                        param_value
                        * self.perturbation_pct
                        * direction
                        * (0.8 + 0.4 * np.random.random())
                    )

                    # Ensure parameter stays positive (for most trading parameters)
                    perturbed_value = max(1.0, param_value + perturbation)
                    perturbed_params[param_name] = perturbed_value

                    # Evaluate perturbed strategy
                    try:
                        score = evaluation_function(perturbed_params)
                        perturbed_scores.append(score)

                        logger.debug(
                            f"Perturbed {param_name}: {param_value:.2f} -> "
                            f"{perturbed_value:.2f}, score: {score:.3f}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Evaluation failed for {param_name}: {e}"
                        )
                        perturbed_scores.append(-999.0)

        # Calculate stability metrics
        if not perturbed_scores:
            logger.error("No valid perturbation scores obtained")
            return StabilityResult(
                is_stable=False,
                perturbation_scores=[],
                original_score=original_score,
                mean_perturbed=0.0,
                std_perturbed=0.0,
                coefficient_of_variation=999.0,
                max_degradation=1.0,
            )

        mean_perturbed = np.mean(perturbed_scores)
        std_perturbed = np.std(perturbed_scores)

        # Coefficient of variation (relative stability)
        cv = std_perturbed / abs(mean_perturbed) if mean_perturbed != 0 else 999.0

        # Maximum degradation from original
        max_degradation = (original_score - min(perturbed_scores)) / abs(
            original_score
        )

        # Stability criteria
        is_stable = (
            max_degradation <= self.max_degradation
            and mean_perturbed >= 0.8 * original_score
        )

        logger.info(
            f"Stability test: {'PASSED' if is_stable else 'FAILED'} | "
            f"Original: {original_score:.3f}, Mean: {mean_perturbed:.3f}, "
            f"CV: {cv:.2f}, MaxDeg: {max_degradation:.1%}"
        )

        return StabilityResult(
            is_stable=is_stable,
            perturbation_scores=perturbed_scores,
            original_score=original_score,
            mean_perturbed=mean_perturbed,
            std_perturbed=std_perturbed,
            coefficient_of_variation=cv,
            max_degradation=max_degradation,
        )


@dataclass
class OverfitResult:
    """Results from overfitting detection."""

    is_overfit: bool
    train_score: float
    validation_score: float
    test_score: float
    train_val_ratio: float
    val_test_ratio: float
    degradation_severity: str


class OverfitValidator:
    """
    Detect overfitting through train/validation/test score comparison.

    Robust strategies should maintain performance across time periods
    and market regimes.
    """

    def __init__(
        self,
        max_train_val_ratio: float = 1.20,
        max_val_test_ratio: float = 1.15,
    ):
        """
        Initialize overfit validator.

        Args:
            max_train_val_ratio: Maximum acceptable train/validation score ratio
            max_val_test_ratio: Maximum acceptable validation/test score ratio
        """
        self.max_train_val_ratio = max_train_val_ratio
        self.max_val_test_ratio = max_val_test_ratio

        logger.info(
            f"OverfitValidator: max_train_val={max_train_val_ratio:.2f}, "
            f"max_val_test={max_val_test_ratio:.2f}"
        )

    def detect_overfitting(
        self,
        train_score: float,
        validation_score: float,
        test_score: float,
    ) -> OverfitResult:
        """
        Detect overfitting by comparing performance across periods.

        Args:
            train_score: Fitness on training data
            validation_score: Fitness on validation data
            test_score: Fitness on out-of-sample test data

        Returns:
            OverfitResult with detection outcome
        """
        # Calculate performance ratios
        train_val_ratio = (
            train_score / validation_score if validation_score > 0 else 999.0
        )
        val_test_ratio = (
            validation_score / test_score if test_score > 0 else 999.0
        )

        # Check for overfitting indicators
        train_val_overfit = train_val_ratio > self.max_train_val_ratio
        val_test_overfit = val_test_ratio > self.max_val_test_ratio

        # Overall overfit determination
        is_overfit = train_val_overfit or val_test_overfit

        # Classify severity
        if not is_overfit:
            severity = "NONE"
        elif train_val_ratio > 1.5 or val_test_ratio > 1.5:
            severity = "SEVERE"
        elif train_val_ratio > 1.3 or val_test_ratio > 1.3:
            severity = "MODERATE"
        else:
            severity = "MILD"

        logger.info(
            f"Overfit detection: {'OVERFIT' if is_overfit else 'CLEAN'} "
            f"({severity}) | Train: {train_score:.3f}, "
            f"Val: {validation_score:.3f}, Test: {test_score:.3f} | "
            f"Ratios: {train_val_ratio:.2f}, {val_test_ratio:.2f}"
        )

        return OverfitResult(
            is_overfit=is_overfit,
            train_score=train_score,
            validation_score=validation_score,
            test_score=test_score,
            train_val_ratio=train_val_ratio,
            val_test_ratio=val_test_ratio,
            degradation_severity=severity,
        )

    def cross_regime_validation(
        self,
        regime_scores: Dict[str, float],
        min_regimes: int = 2,
        min_consistency: float = 0.7,
    ) -> Tuple[bool, str]:
        """
        Validate strategy performs consistently across market regimes.

        Args:
            regime_scores: Dictionary mapping regime names to fitness scores
            min_regimes: Minimum number of regimes required
            min_consistency: Minimum consistency ratio (worst/best)

        Returns:
            Tuple of (is_consistent, message)
        """
        if len(regime_scores) < min_regimes:
            msg = (
                f"Insufficient regimes tested: {len(regime_scores)} < "
                f"{min_regimes}"
            )
            logger.warning(msg)
            return False, msg

        # Filter out failed regimes (negative scores)
        valid_scores = {k: v for k, v in regime_scores.items() if v > 0}

        if len(valid_scores) < min_regimes:
            msg = (
                f"Strategy failed in too many regimes: "
                f"{len(valid_scores)}/{len(regime_scores)} passed"
            )
            logger.warning(msg)
            return False, msg

        # Calculate consistency
        scores = list(valid_scores.values())
        consistency_ratio = min(scores) / max(scores) if max(scores) > 0 else 0.0

        is_consistent = consistency_ratio >= min_consistency

        msg = (
            f"{'Consistent' if is_consistent else 'Inconsistent'} across regimes: "
            f"{consistency_ratio:.1%} | Scores: {valid_scores}"
        )

        logger.info(msg)
        return is_consistent, msg
