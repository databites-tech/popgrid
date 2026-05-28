"""Custom exceptions for popgrid."""

from __future__ import annotations


class PopGridError(Exception):
    """Base exception for all popgrid errors."""


class DataNotFoundError(PopGridError):
    """Raised when required geodata cannot be loaded or downloaded."""


class CountryNotFoundError(PopGridError):
    """Raised when the requested ISO country code yields no regions."""


class GeometryError(PopGridError):
    """Raised when a geometry operation fails unexpectedly."""
