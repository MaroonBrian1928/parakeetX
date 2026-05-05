"""ParakeetX API server package."""

from .log_filters import install_noisy_dependency_log_filters

install_noisy_dependency_log_filters()

__all__ = ["__version__"]

__version__ = "0.1.0"
