"""오프라인 freqtrade 실행기 — 거래소 마켓 로딩을 네트워크 없이 우회.

배경
----
이 샌드박스는 binance 를 포함한 모든 거래소 API 를 403 으로 막는다. freqtrade 는
시작 시 `reload_markets()` 로 거래소 마켓(심볼·정밀도·계약크기 등)을 받아오는데,
네트워크가 막혀 백테스트조차 시작되지 않는다.

레버리지 티어는 freqtrade 에 binance_leverage_tiers.json 으로 번들돼 오프라인에서
동작한다. 남은 것은 마켓 dict 뿐 → 여기서 BTC/ETH USDT 무기한선물 2종의 마켓을
수작업으로 구성해 `Exchange.reload_markets` 를 몽키패치한다.

그 후 freqtrade 명령(backtesting / lookahead-analysis / recursive-analysis)을
**같은 프로세스 안에서** 호출한다(서브프로세스면 패치가 전달되지 않음).

사용:
    .venv/bin/python scripts/offline_freqtrade.py backtesting -c freqtrade/config_bollinger.json -s BollingerOpenBreakout --timerange 20240115-
    .venv/bin/python scripts/offline_freqtrade.py lookahead-analysis -c freqtrade/config_bollinger.json -s BollingerOpenBreakout --timerange 20240201-20240601
    .venv/bin/python scripts/offline_freqtrade.py recursive-analysis  -c freqtrade/config_bollinger.json -s BollingerOpenBreakout --timerange 20240201-20240301
"""
from __future__ import annotations

import sys

from freqtrade.enums import TradingMode
from freqtrade.exchange import exchange as ex_mod
from freqtrade.util.datetime_helpers import dt_ts


# 백테스트할 무기한선물(USDT 정산, linear) 페어들.
_PAIRS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
]


def _market(symbol: str) -> dict:
    base, rest = symbol.split("/")
    quote = rest.split(":")[0]
    return {
        "id": f"{base}{quote}",
        "lowercaseId": f"{base}{quote}".lower(),
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "settle": quote,
        "baseId": base,
        "quoteId": quote,
        "settleId": quote,
        "type": "swap",
        "spot": False,
        "margin": False,
        "swap": True,
        "future": False,
        "option": False,
        "index": False,
        "active": True,
        "contract": True,
        "linear": True,
        "inverse": False,
        "subType": "linear",
        "taker": 0.0005,
        "maker": 0.0002,
        "contractSize": 1.0,
        "expiry": None,
        "expiryDatetime": None,
        "strike": None,
        "optionType": None,
        "precision": {"amount": 0.001, "price": 0.01, "base": 1e-8, "quote": 1e-8},
        "limits": {
            "leverage": {"min": 1.0, "max": 125.0},
            "amount": {"min": 0.001, "max": 10000.0},
            "price": {"min": None, "max": None},
            "cost": {"min": 5.0, "max": None},
            "market": {"min": 0.001, "max": 10000.0},
        },
        "marginModes": {"cross": True, "isolated": True},
        "created": None,
        "percentage": True,
        "tierBased": True,
        "feeSide": "get",
        "info": {},
    }


def _install_offline_markets() -> None:
    markets = {s: _market(s) for s in _PAIRS}

    def offline_reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True):
        # ccxt set_markets 가 markets_by_id/currencies 등 파생 구조를 빌드.
        self._api_async.set_markets(markets)
        self._api.set_markets(markets)
        self._markets = self._api_async.markets
        self._last_markets_refresh = dt_ts()
        if load_leverage_tiers and self.trading_mode == TradingMode.FUTURES:
            self.fill_leverage_tiers()

    ex_mod.Exchange.reload_markets = offline_reload_markets
    print("[offline_freqtrade] reload_markets 패치 완료 — 합성 마켓 사용:", _PAIRS)


def main() -> None:
    _install_offline_markets()
    # freqtrade CLI 를 인-프로세스로 호출 (패치 유지).
    from freqtrade.commands import Arguments

    args = Arguments(sys.argv[1:]).get_parsed_arg()
    func = args.get("func")
    if func is None:
        raise SystemExit("freqtrade 서브커맨드를 지정하세요 (backtesting 등).")
    return func(args)


if __name__ == "__main__":
    main()
