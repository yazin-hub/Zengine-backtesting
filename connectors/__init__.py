"""
connectors — ZEngine Live Execution Connectors
===============================================

Bridges ZEngine strategy signals to live brokers and exchanges.

Available connectors:
    CTraderConnector  — cTrader Open API (Spotware)  [connectors/ctrader.py]

Planned:
    BinanceConnector  — Binance Futures (REST + WebSocket)
    IBKRConnector     — Interactive Brokers TWS API
    OANDAConnector    — OANDA v20 REST API

All connectors implement BaseConnector — the common interface that lets you
write a strategy runner once and deploy to any supported platform.

Quick start:
    from connectors.ctrader import CTraderConnector

    conn = CTraderConnector(
        client_id="...", client_secret="...",
        account_id=12345, access_token="...", env="demo",
    )
    conn.connect()
    df = conn.fetch_ohlcv("BTCUSD", "15m", limit=200)
"""

from .base import BaseConnector, OrderResult, Position

__all__ = ["BaseConnector", "OrderResult", "Position"]
