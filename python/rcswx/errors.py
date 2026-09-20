"""Public failures; reports never retain models, tensors, or traceback objects."""


class RcswxError(Exception):
    """Base class for expected RCSWX failures."""


class UnsupportedArchitecture(RcswxError, ValueError):
    """The model is outside the declared representation envelope."""


class UnsupportedOperator(UnsupportedArchitecture):
    """An operation has no verified registry entry."""


class UnsupportedEffect(UnsupportedArchitecture):
    """An effect cannot be represented safely."""


class UnsupportedSharing(UnsupportedArchitecture):
    """A module, parameter, or storage is shared."""


class UnsupportedConfiguration(UnsupportedArchitecture):
    """A registered operator configuration is not admissible."""


class AmbiguousRepresentation(UnsupportedArchitecture):
    """More than one normalized grammar representation is possible."""


class InputContractMismatch(RcswxError, ValueError):
    """The parents have different root input contracts."""


class OutputContractMismatch(RcswxError, ValueError):
    """The parents have different root output contracts."""


class StaleEditPath(RcswxError, ValueError):
    """The supplied path does not describe these ordered architectures."""


class NoAdmissibleAlignment(RcswxError):
    """The complete supported search proved there is no legal alignment."""


class InvalidEditSelection(RcswxError, ValueError):
    """The selected positive edits do not form a valid architecture."""


class SamplingExhausted(RcswxError):
    """Every proposal in the finite request budget was shape-invalid."""

    def __init__(self, report: dict[str, object]):
        super().__init__("no shape-admissible proposal within max_attempts")
        self.report = report


class BudgetExceeded(RcswxError):
    """A resource guard stopped work; this is not a negative search result."""

    def __init__(self, resource: str, limit: int | float, observed: int | float):
        super().__init__(f"{resource} exceeded: {observed} > {limit}")
        self.resource = resource
        self.limit = limit
        self.observed = observed


class Cancelled(RcswxError):
    """The request's cancellation token was set."""


class InternalInvariant(RcswxError):
    """A library invariant failed; never treat this as a sampling rejection."""


def _native_call(function, /, *args, **kwargs):
    from . import _core

    try:
        return function(*args, **kwargs)
    except _core.CoreError as exc:
        kind, *payload = exc.args
        if kind == "budget":
            raise BudgetExceeded(*payload) from None
        if kind == "cancelled":
            raise Cancelled("request cancelled") from None
        if kind == "invalid_input":
            raise UnsupportedConfiguration(*payload) from None
        if kind == "invalid_selection":
            raise InvalidEditSelection(*payload) from None
        if kind == "no_alignment":
            raise NoAdmissibleAlignment("complete search found no legal witness") from None
        if kind == "internal":
            raise InternalInvariant(*payload) from None
        raise InternalInvariant("unknown native error category") from exc
