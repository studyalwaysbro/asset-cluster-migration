"""Rate-limited FMP API client with caching and bandwidth tracking."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_api_key, get_settings, PROJECT_ROOT

logger = logging.getLogger(__name__)

STABLE_BASE = "https://financialmodelingprep.com/stable"


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API call control."""

    def __init__(self, max_tokens: int, refill_period: float = 60.0):
        self.max_tokens = max_tokens
        self.refill_period = refill_period
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + (elapsed / self.refill_period) * self.max_tokens,
            )
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.max_tokens / self.refill_period)
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class BandwidthTracker:
    """Track cumulative bandwidth usage."""

    def __init__(self, max_gb: float = 140.0):
        self.max_bytes = max_gb * 1e9
        self.total_bytes = 0

    def record(self, size_bytes: int) -> None:
        self.total_bytes += size_bytes
        if self.total_bytes > self.max_bytes * 0.9:
            logger.warning(
                f"Bandwidth at {self.total_bytes / 1e9:.1f}GB / {self.max_bytes / 1e9:.0f}GB"
            )
        if self.total_bytes > self.max_bytes:
            raise RuntimeError("Bandwidth limit exceeded")

    @property
    def used_gb(self) -> float:
        return self.total_bytes / 1e9


def _is_us_market_open() -> bool:
    """Check if US equity market is currently in session (approx)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _last_business_day() -> date:
    """Return the most recent business day (today if weekday, else Friday)."""
    today = date.today()
    offset = max(0, today.weekday() - 4)  # Sat=1, Sun=2
    return today - timedelta(days=offset)


class FMPClient:
    """Async FMP client using the stable API (post-Aug 2025)."""

    def __init__(
        self,
        api_key: str | None = None,
        max_calls_per_min: int | None = None,
        cache_dir: Path | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or get_api_key()
        self.base_url = STABLE_BASE
        max_rpm = max_calls_per_min or settings["fmp"]["max_calls_per_minute"]
        max_bw = settings["fmp"]["max_bandwidth_gb"]

        self.rate_limiter = TokenBucketRateLimiter(max_rpm)
        self.bandwidth = BandwidthTracker(max_bw)
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data" / "raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _cache_path(self, symbol: str, endpoint: str) -> Path:
        return self.cache_dir / f"{symbol}_{endpoint}.parquet"

    def _check_cache(
        self, symbol: str, endpoint: str, max_staleness_days: int = 1
    ) -> pd.DataFrame | None:
        """Return cached data if fresh enough, else None.

        A cache entry is considered stale if its most recent row is older
        than *max_staleness_days* business days.  Pass ``max_staleness_days=0``
        to always re-fetch.
        """
        path = self._cache_path(symbol, endpoint)
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        if df.empty:
            return None

        last_cached = pd.Timestamp(df.index.max()).date()
        target = _last_business_day()
        stale_cutoff = target - timedelta(days=max_staleness_days)
        if last_cached < stale_cutoff:
            logger.info(
                f"Cache stale for {symbol}: last={last_cached}, target={target}"
            )
            return None

        logger.debug(f"Cache hit: {symbol}")
        return df

    def _write_cache(self, df: pd.DataFrame, symbol: str, endpoint: str) -> None:
        path = self._cache_path(symbol, endpoint)
        df.to_parquet(path, index=True)
        logger.debug(f"Cached: {symbol}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def _request_stable(self, endpoint: str, params: dict[str, Any]) -> Any:
        """Make a request to the stable API."""
        await self.rate_limiter.acquire()
        client = await self._get_client()
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        response = await client.get(url, params=params)
        self.bandwidth.record(len(response.content))
        response.raise_for_status()
        return response.json()

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch the real-time (or latest) quote for a symbol."""
        try:
            data = await self._request_stable("quote", {"symbol": symbol})
            if data and isinstance(data, list):
                return data[0]
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.debug(f"Quote failed for {symbol}: {e}")
        return None

    async def get_historical_prices(
        self,
        symbol: str,
        from_date: str,
        to_date: str | None = None,
        extend_to_today: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily dividend-adjusted prices for a symbol.

        Parameters
        ----------
        extend_to_today : bool
            If True and the historical data does not cover today, append
            the latest quote using close as a fallback for adjClose.
        """
        cached = self._check_cache(symbol, "historical")
        if cached is not None:
            return cached

        params: dict[str, Any] = {"symbol": symbol, "from": from_date}
        if to_date:
            params["to"] = to_date

        data = await self._request_stable(
            "historical-price-eod/dividend-adjusted", params
        )

        if not data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if extend_to_today:
            df = await self._maybe_extend_today(df, symbol)

        self._write_cache(df, symbol, "historical")
        return df

    async def _maybe_extend_today(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Append today's quote if historical data does not cover it."""
        if df.empty:
            return df

        today = pd.Timestamp(date.today())
        last_date = df.index.max()

        if last_date >= today:
            return df

        if today.weekday() >= 5:
            return df

        quote = await self.get_quote(symbol)
        if quote is None:
            return df

        price = quote.get("price") or quote.get("previousClose")
        if price is None:
            return df

        new_row = {col: np.nan for col in df.columns}
        new_row["close"] = float(price)
        new_row["adjClose"] = float(price)
        new_row["open"] = float(quote.get("open", price))
        new_row["high"] = float(quote.get("dayHigh", price))
        new_row["low"] = float(quote.get("dayLow", price))
        new_row["volume"] = int(quote.get("volume", 0))

        today_row = pd.DataFrame(
            [new_row], index=pd.DatetimeIndex([today], name="date")
        )
        today_row = today_row.reindex(columns=df.columns)
        df = pd.concat([df, today_row])
        logger.info(f"{symbol}: extended to {today.date()} via quote (close={price})")
        return df

    async def batch_fetch(
        self,
        symbols: list[str],
        from_date: str,
        to_date: str | None = None,
        max_concurrent: int = 10,
        extend_to_today: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical prices for multiple symbols with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results: dict[str, pd.DataFrame] = {}

        async def fetch_one(symbol: str) -> None:
            async with semaphore:
                try:
                    results[symbol] = await self.get_historical_prices(
                        symbol, from_date, to_date, extend_to_today=extend_to_today
                    )
                    logger.info(f"Fetched {symbol}: {len(results[symbol])} rows")
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                    results[symbol] = pd.DataFrame()

        await asyncio.gather(*[fetch_one(s) for s in symbols])
        return results

    async def get_treasury_rates(
        self, from_date: str, to_date: str | None = None
    ) -> pd.DataFrame:
        """Fetch treasury rates across maturities."""
        cached = self._check_cache("treasury", "rates")
        if cached is not None:
            return cached

        params: dict[str, Any] = {"from": from_date}
        if to_date:
            params["to"] = to_date

        data = await self._request_stable("treasury-rates", params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        self._write_cache(df, "treasury", "rates")
        return df

    async def get_economic_indicator(self, name: str) -> pd.DataFrame:
        """Fetch economic indicator (GDP, CPI, inflation, etc.)."""
        cached = self._check_cache(f"econ_{name}", "indicator")
        if cached is not None:
            return cached

        data = await self._request_stable("economic-indicators", {"name": name})

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        self._write_cache(df, f"econ_{name}", "indicator")
        return df
