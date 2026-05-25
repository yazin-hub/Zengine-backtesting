"""Built-in strategies. Import and register here for UI auto-discovery."""
from .ifvg import IFVGStrategy
from .ma_cross import MACrossStrategy

REGISTRY: dict = {
    IFVGStrategy.NAME:   IFVGStrategy,
    MACrossStrategy.NAME: MACrossStrategy,
}
