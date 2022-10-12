"""Supports efficiency with skip connections."""
from torchmobius.skip.namespace import Namespace
from torchmobius.skip.skippable import pop, skippable, stash, verify_skippables

__all__ = ['skippable', 'stash', 'pop', 'verify_skippables', 'Namespace']
