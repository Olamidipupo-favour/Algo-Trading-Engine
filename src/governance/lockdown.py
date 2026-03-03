"""
Parameter lockdown system for preventing emotional tampering.

Enforces scheduled review periods and prevents ad-hoc parameter changes
that could compromise systematic discipline.
"""

import json
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

from src.utils.logger import Logger

logger = Logger(__name__)


@dataclass
class LockdownConfig:
    """Configuration for parameter lockdown."""

    lock_duration_days: int
    next_review_date: str
    evolution_windows: list  # Scheduled evolution/reoptimization dates
    locked_parameters: Dict[str, Any]
    lock_reason: str
    locked_by: str
    locked_at: str


class ParameterLockdown:
    """
    Enforce parameter stability and prevent unauthorized changes.

    Locks parameters for fixed review intervals and only allows
    changes during scheduled evolution windows.
    """

    def __init__(
        self,
        lock_duration_days: int = 30,
        config_file: str = "governance/lockdown_config.json",
    ):
        """
        Initialize parameter lockdown.

        Args:
            lock_duration_days: Days between allowed parameter changes
            config_file: Path to lockdown configuration file
        """
        self.lock_duration_days = lock_duration_days
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        self.current_lock: Optional[LockdownConfig] = None
        self.load_config()

        logger.info(
            f"ParameterLockdown initialized: duration={lock_duration_days} days"
        )

    def lock_parameters(
        self,
        parameters: Dict[str, Any],
        reason: str = "scheduled_review",
        locked_by: str = "system",
    ) -> bool:
        """
        Lock parameters for the configured duration.

        Args:
            parameters: Parameters to lock
            reason: Reason for locking
            locked_by: Who initiated the lock

        Returns:
            True if successfully locked
        """
        if self.is_locked() and not self.is_evolution_window():
            logger.error("Cannot lock: existing lock active and not in evolution window")
            return False

        next_review = datetime.now() + timedelta(days=self.lock_duration_days)

        # Schedule evolution windows (weekly)
        evolution_windows = []
        for i in range(1, 5):
            window_date = datetime.now() + timedelta(weeks=i)
            evolution_windows.append(window_date.strftime("%Y-%m-%d"))

        lock_config = LockdownConfig(
            lock_duration_days=self.lock_duration_days,
            next_review_date=next_review.strftime("%Y-%m-%d"),
            evolution_windows=evolution_windows,
            locked_parameters=parameters,
            lock_reason=reason,
            locked_by=locked_by,
            locked_at=datetime.now().isoformat(),
        )

        self.current_lock = lock_config
        self.save_config()

        logger.info(
            f"Parameters locked until {next_review.strftime('%Y-%m-%d')} | "
            f"Reason: {reason}"
        )

        return True

    def is_locked(self) -> bool:
        """Check if parameters are currently locked."""
        if not self.current_lock:
            return False

        next_review = datetime.strptime(
            self.current_lock.next_review_date, "%Y-%m-%d"
        )

        return datetime.now() < next_review

    def is_evolution_window(self) -> bool:
        """Check if current date is within an evolution window."""
        if not self.current_lock:
            return True  # No lock, always allowed

        today = datetime.now().strftime("%Y-%m-%d")
        return today in self.current_lock.evolution_windows

    def can_modify_parameters(self) -> Tuple[bool, str]:
        """
        Check if parameter modification is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.is_locked():
            return True, "No active lock"

        if self.is_evolution_window():
            return True, "Within scheduled evolution window"

        days_until_review = (
            datetime.strptime(self.current_lock.next_review_date, "%Y-%m-%d")
            - datetime.now()
        ).days

        return (
            False,
            f"Parameters locked. Next review in {days_until_review} days. "
            f"Evolution windows: {', '.join(self.current_lock.evolution_windows)}",
        )

    def request_override(self, reason: str, requestor: str) -> bool:
        """
        Request emergency override of parameter lock.

        Requires explicit approval and is logged.

        Args:
            reason: Justification for override
            requestor: Who is requesting override

        Returns:
            False (manual approval required)
        """
        logger.critical(
            f"OVERRIDE REQUESTED by {requestor}: {reason} | "
            f"Lock active until {self.current_lock.next_review_date}"
        )

        # Log to audit trail
        self._log_override_request(reason, requestor)

        logger.error(
            "Override request logged. Manual approval required via config file."
        )

        return False

    def unlock(self, reason: str, unlocked_by: str) -> bool:
        """
        Manually unlock parameters (emergency use only).

        Args:
            reason: Reason for unlocking
            unlocked_by: Who approved unlock

        Returns:
            True if unlocked
        """
        if not self.current_lock:
            return True

        logger.warning(
            f"Parameters unlocked by {unlocked_by} | Reason: {reason}"
        )

        self.current_lock = None
        self.save_config()

        return True

    def get_locked_parameters(self) -> Optional[Dict[str, Any]]:
        """Get currently locked parameters."""
        if not self.current_lock:
            return None

        return self.current_lock.locked_parameters.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get lockdown status."""
        if not self.current_lock:
            return {
                "locked": False,
                "message": "No active lock",
            }

        days_until_review = (
            datetime.strptime(self.current_lock.next_review_date, "%Y-%m-%d")
            - datetime.now()
        ).days

        return {
            "locked": self.is_locked(),
            "next_review_date": self.current_lock.next_review_date,
            "days_until_review": days_until_review,
            "evolution_windows": self.current_lock.evolution_windows,
            "lock_reason": self.current_lock.lock_reason,
            "locked_by": self.current_lock.locked_by,
            "locked_at": self.current_lock.locked_at,
            "can_modify": self.can_modify_parameters()[0],
        }

    def load_config(self) -> None:
        """Load lockdown configuration from disk."""
        if not self.config_file.exists():
            logger.info("No existing lockdown config found")
            return

        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
                self.current_lock = LockdownConfig(**data)
            logger.info("Lockdown config loaded")
        except Exception as e:
            logger.error(f"Failed to load lockdown config: {e}")

    def save_config(self) -> None:
        """Save lockdown configuration to disk."""
        try:
            with open(self.config_file, "w") as f:
                if self.current_lock:
                    json.dump(asdict(self.current_lock), f, indent=2)
                else:
                    json.dump({}, f)
            logger.debug("Lockdown config saved")
        except Exception as e:
            logger.error(f"Failed to save lockdown config: {e}")

    def _log_override_request(self, reason: str, requestor: str) -> None:
        """Log override request to audit file."""
        audit_file = self.config_file.parent / "override_requests.log"

        with open(audit_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} | OVERRIDE REQUEST | "
                f"Requestor: {requestor} | Reason: {reason}\n"
            )
