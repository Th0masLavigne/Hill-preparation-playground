"""Weak-form operators (target: ``fenicsoft.operators``)."""

from . import cell, residual_rows
from .residual_rows import ResidualRows

__all__ = ["cell", "residual_rows", "ResidualRows"]
