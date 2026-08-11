import os


class BuddySecretError(RuntimeError):
    """Raised when a required BUDDY secret is unavailable."""


def get_secret(name: str, required: bool = False) -> str | None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Secret name must be a non-empty string.")

    value = os.getenv(name)

    if value is not None:
        value = value.strip()

    if not value:
        if required:
            raise BuddySecretError(
                f"Required environment secret '{name}' is not configured."
            )

        return None

    return value
