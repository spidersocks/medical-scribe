class ServiceError(Exception):
    """Base class for predictable service-layer exceptions."""


class NotFoundError(ServiceError):
    """Raised when the requested resource does not exist."""


class ValidationError(ServiceError):
    """Raised when business rules are violated."""