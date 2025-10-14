class ServiceError(Exception):
    """Generic service-layer error."""


class ValidationError(ServiceError):
    """Raised when input data fails validation rules."""


class NotFoundError(ServiceError):
    """Raised when an entity cannot be found."""