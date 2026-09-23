"""Lightweight failure contract for custom agent functions."""


class InfraAbort(Exception):
    """Discard a sample after an infrastructure failure outside policy control.

    Outcomes the policy can cause must instead receive their normal reward. An
    infrastructure abort contributes no gradient, so using it for a
    policy-dependent failure would teach the policy to escape that failure's
    penalty.
    """

    def __init__(self, exit_status: str, message: str | None = None):
        super().__init__(message or exit_status)
        self.exit_status = exit_status
