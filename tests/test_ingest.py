"""실데이터 인제스트 테스트 — 다양한 파일 관례를 표준 OHLCV로 흡수하는지.

거래소/yfinance 관례(에폭 ms, 대문자 컬럼, Date 문자열)를 임시 파일로 만들어 로드→검증.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import ingest


def _write_csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_yfinance_style_date_columns(tmp_path):
    """Date + 대문자 OHLCV 문자열 날짜."""
    df = pd.DataFrame({
        "Date": pd.date_range("2021-01-01", periods=50, freq="1h"),
        "Open": np.linspace(100, 110, 50), "High": np.linspace(101, 111, 50),
        "Low": np.linspace(99, 109, 50), "Close": np.linspace(100, 110, 50),
        "Volume": np.arange(50) + 1000.0,
    })
    out = ingest.load_ohlcv_file(_write_csv(tmp_path, "AAPL.csv", df))
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None and str(out.index.tz) == "UTC"
    assert out.index.name == "time"
    assert len(out) == 50


def test_binance_style_epoch_ms(tmp_path):
    """timestamp가 밀리초 에폭 정수."""
    start_ms = 1609459200000  # 2021-01-01 UTC
    ts = [start_ms + i * 3600_000 for i in range(30)]
    df = pd.DataFrame({
        "timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
        "volume": 10.0,
    })
    out = ingest.load_ohlcv_file(_write_csv(tmp_path, "BTCUSDT.csv", df))
    assert len(out) == 30
    assert out.index[0].year == 2021
    assert out.index.tz is not None


def test_missing_volume_filled(tmp_path):
    df = pd.DataFrame({
        "time": pd.date_range("2021-01-01", periods=10, freq="1h"),
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
    })
    out = ingest.load_ohlcv_file(_write_csv(tmp_path, "x.csv", df))
    assert "volume" in out.columns
    assert (out["volume"] == 0.0).all()


def test_missing_ohlc_raises(tmp_path):
    df = pd.DataFrame({
        "time": pd.date_range("2021-01-01", periods=5, freq="1h"),
        "open": 1.0, "close": 1.5,  # high/low 없음
    })
    with pytest.raises(ValueError, match="필수 OHLC"):
        ingest.load_ohlcv_file(_write_csv(tmp_path, "bad.csv", df))


def test_no_time_column_raises(tmp_path):
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    with pytest.raises(ValueError, match="시간열"):
        ingest.load_ohlcv_file(_write_csv(tmp_path, "notime.csv", df))


def test_parquet_roundtrip_with_datetime_index(tmp_path):
    """parquet 캐시(이미 DatetimeIndex)도 그대로 로드."""
    idx = pd.date_range("2021-01-01", periods=20, freq="1h", tz="UTC", name="time")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0},
                      index=idx)
    p = tmp_path / "S.parquet"
    df.to_parquet(p)
    out = ingest.load_ohlcv_file(p)
    assert len(out) == 20 and list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_load_universe_from_dir_and_backtest(tmp_path):
    """디렉터리 로드 → 포트폴리오까지 흐르는지(엔드투엔드)."""
    for sym in ["A", "B", "C"]:
        df = pd.DataFrame({
            "time": pd.date_range("2021-01-01", periods=400, freq="1h"),
            "open": np.random.default_rng(hash(sym) % 100).normal(100, 1, 400).cumsum() + 500,
            "high": 0.0, "low": 0.0, "close": 0.0, "volume": 1000.0,
        })
        df["high"] = df["open"] + 1
        df["low"] = df["open"] - 1
        df["close"] = df["open"]
        df.to_csv(tmp_path / f"{sym}.csv", index=False)

    data = ingest.load_universe(tmp_path)
    assert set(data) == {"A", "B", "C"}
    for d in data.values():
        assert list(d.columns) == ["open", "high", "low", "close", "volume"]
        assert d.index.tz is not None

    # 포트폴리오까지 실제로 흐르는지.
    from src import portfolio, strategies as strat
    from src.engine.position import Costs
    res = portfolio.run_portfolio(
        strat.get("breakout_donchian"), data,
        Costs(0.001, 0.001, 0.0), {"initial_capital": 9000.0, "position_pct": 1.0},
        24 * 365)
    assert res["diversification"]["n_assets"] == 3
