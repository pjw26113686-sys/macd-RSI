"""프랍 룰 평가기 — 봉별 equity(고/저 포함)를 걸어 통과/실격을 판정.

일반 백테스트와 프랍 백테스트의 결정적 차이가 이 모듈이다. 전략의 손익곡선이
프랍 계좌 룰(트레일링DD·일일손실·수익목표·일관성)을 만족하는지 시뮬레이션한다.

정확도 핵심: Apex 인트라데이 트레일링DD는 봉 wick(고/저)에서 터질 수 있으므로
종가가 아니라 **봉 고/저 기준 미실현 포함 equity(equity_high/equity_low)** 로
평가한다(엔진이 산출).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.prop.rules import PropRuleSet


@dataclass
class PropResult:
    ruleset: str
    passed: bool
    failed: bool
    fail_reason: str | None
    fail_time: object
    days_to_pass: int | None
    n_days: int
    final_equity: float
    total_pnl: float
    max_trailing_used: float        # 고점 대비 최대 하락폭($)
    worst_day: float
    best_day: float
    consistency_ratio: float | None # best_day / total_pnl
    passed_consistency: bool

    def summary(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.failed else "—")
        parts = [
            f"[{self.ruleset}] {status}",
            f"순손익 ${self.total_pnl:,.0f}",
            f"최대트레일링 ${self.max_trailing_used:,.0f}",
        ]
        if self.passed and self.days_to_pass is not None:
            parts.append(f"소요 {self.days_to_pass}일")
        if self.failed:
            parts.append(f"사유: {self.fail_reason}")
        if self.consistency_ratio is not None:
            ok = "OK" if self.passed_consistency else "위반"
            parts.append(f"일관성 {self.consistency_ratio*100:.0f}% ({ok})")
        return " · ".join(parts)


def _session_dates(index: pd.DatetimeIndex, tz: str, reset_hour: int) -> pd.Series:
    """봉 인덱스를 세션(거래일) 날짜로 라벨링. reset_hour 이후 봉은 다음 세션."""
    idx = index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(tz)
    shifted = local + pd.Timedelta(hours=24 - reset_hour)
    return pd.Series(shifted.normalize().tz_localize(None), index=index)


def evaluate_prop(equity, equity_high, equity_low, rules: PropRuleSet) -> PropResult:
    """봉별 equity 시리즈로 프랍 룰을 시뮬레이션.

    equity/high/low: tz-aware DatetimeIndex(또는 naive=UTC) Series. 동일 인덱스.
    """
    eq = equity
    hi = equity_high if rules.trailing_type == "intraday" else equity
    lo = equity_low if rules.trailing_type == "intraday" else equity

    start = rules.start_balance
    target_balance = start + rules.profit_target

    # 세션(거래일) 라벨
    sessions = _session_dates(eq.index, rules.reset_tz, rules.reset_time)
    unique_days = list(dict.fromkeys(sessions.tolist()))
    day_index = {d: i for i, d in enumerate(unique_days)}

    peak = start
    threshold = start - rules.trailing_drawdown
    locked = False
    max_trailing_used = 0.0

    failed = False
    fail_reason = None
    fail_time = None
    passed = False
    days_to_pass = None

    # 일일 손익(종가 기준) 누적용
    day_start_equity = {}   # 세션 → 직전 세션 종료 잔고
    prev_day_close = start
    cur_day = None

    times = eq.index
    for i in range(len(eq)):
        t = times[i]
        day = sessions.iloc[i]
        if day != cur_day:
            if cur_day is not None:
                # 직전 세션 마감 → 일일손실 한도 검사
                daily_pnl = prev_close_in_day - day_open_equity
                if rules.daily_loss_limit is not None and daily_pnl <= -rules.daily_loss_limit and not failed and not passed:
                    failed = True
                    fail_reason = "DAILY_LOSS"
                    fail_time = prev_t_in_day
            cur_day = day
            day_open_equity = prev_day_close
            day_start_equity[day] = day_open_equity

        # 트레일링DD (인트라데이=봉 고저)
        peak = max(peak, hi.iloc[i])
        if not locked:
            threshold = peak - rules.trailing_drawdown
            if rules.trail_lock_at is not None and threshold >= rules.trail_lock_at:
                threshold = rules.trail_lock_at
                locked = True
        max_trailing_used = max(max_trailing_used, peak - lo.iloc[i])

        if not failed and not passed and lo.iloc[i] <= threshold:
            failed = True
            fail_reason = "TRAILING_DD"
            fail_time = t

        # 수익목표 통과(종가 기준)
        if not passed and not failed and eq.iloc[i] >= target_balance:
            passed = True
            days_to_pass = day_index[day] + 1

        prev_close_in_day = eq.iloc[i]
        prev_t_in_day = t
        # 세션 종료시 갱신을 위해 매 봉 종가를 prev_day_close 후보로
        prev_day_close = eq.iloc[i]

        if failed:
            break

    # 마지막 세션 일일손실 검사(루프 종료 후)
    if not failed and not passed and rules.daily_loss_limit is not None and cur_day is not None:
        daily_pnl = prev_close_in_day - day_start_equity[cur_day]
        if daily_pnl <= -rules.daily_loss_limit:
            failed = True
            fail_reason = "DAILY_LOSS"
            fail_time = prev_t_in_day

    # 일일 손익 집계(전 구간) — worst/best/consistency
    df = pd.DataFrame({"equity": eq.values, "day": sessions.values})
    day_close = df.groupby("day")["equity"].last()
    day_pnl = day_close.diff()
    day_pnl.iloc[0] = day_close.iloc[0] - start
    worst_day = float(day_pnl.min()) if len(day_pnl) else 0.0
    best_day = float(day_pnl.max()) if len(day_pnl) else 0.0

    final_equity = float(eq.iloc[-1])
    total_pnl = final_equity - start

    consistency_ratio = None
    passed_consistency = True
    if rules.consistency_pct is not None and total_pnl > 0:
        consistency_ratio = best_day / total_pnl
        passed_consistency = consistency_ratio <= rules.consistency_pct

    return PropResult(
        ruleset=rules.name, passed=passed, failed=failed, fail_reason=fail_reason,
        fail_time=fail_time, days_to_pass=days_to_pass, n_days=len(unique_days),
        final_equity=final_equity, total_pnl=total_pnl,
        max_trailing_used=max_trailing_used, worst_day=worst_day, best_day=best_day,
        consistency_ratio=consistency_ratio, passed_consistency=passed_consistency,
    )
