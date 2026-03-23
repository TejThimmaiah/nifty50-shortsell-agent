"""
market_scanner.py — DEPRECATED SHIM
The agent now uses Nifty50ShortScanner exclusively.
This file re-exports from there for backwards compatibility.
"""
from agents.nifty50_scanner import (
    Nifty50ShortScanner as MarketScannerAgent,
    ShortCandidate,
)

__all__ = ["MarketScannerAgent", "ShortCandidate"]
