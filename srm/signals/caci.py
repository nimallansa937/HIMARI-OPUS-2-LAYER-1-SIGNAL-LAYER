"""
Cross-Asset Contagion Index (CACI)

Monitors traditional finance stress indicators for crypto contagion risk.
Crypto markets remain connected to global risk sentiment, and TradFi stress
frequently precedes crypto selloffs as leveraged players reduce risk.

Forensic Basis:
- August 5, 2024: A $4 trillion yen carry trade unwind originated in TradFi.
  VIX spiked above 65, USD/JPY moved 8% in 48 hours, and BTC dropped 15%
  despite no crypto-specific catalyst.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class CACIConfig:
    """Configuration for Cross-Asset Contagion Index calculation."""
    vix_elevated_threshold: float = 25  # VIX above 25 = elevated fear
    vix_crisis_threshold: float = 40  # VIX above 40 = market panic
    usdjpy_move_threshold: float = 0.02  # 2% daily move = carry trade stress
    spx_drawdown_threshold: float = 0.03  # 3% drawdown from recent high
    lookback_days: int = 5  # Window for calculating SPX drawdown
    timeout_seconds: float = 10.0


class CrossAssetContagionIndex:
    """
    Monitors traditional finance stress indicators for crypto contagion risk.
    
    Crypto markets, despite claims of decorrelation, remain connected to
    global risk sentiment. TradFi stress—particularly carry trade unwinds
    and volatility spikes—frequently precedes crypto selloffs as leveraged
    players reduce risk across asset classes.
    """
    
    YAHOO_FINANCE_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def __init__(self, config: Optional[CACIConfig] = None):
        self.config = config or CACIConfig()
        self.spx_history: List[float] = []  # Store recent SPX closes for drawdown calc
        self.usdjpy_history: List[float] = []  # Store recent USD/JPY for velocity calc
    
    async def _fetch_yahoo_data(
        self, 
        session: aiohttp.ClientSession, 
        symbol: str
    ) -> Optional[List[float]]:
        """Fetch historical data from Yahoo Finance."""
        try:
            url = f"{self.YAHOO_FINANCE_BASE}/{symbol}?interval=1d&range=5d"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as resp:
                if resp.status != 200:
                    return None
                result = await resp.json()
                
                # Extract closing prices
                closes = result['chart']['result'][0]['indicators']['quote'][0]['close']
                # Filter out None values
                valid_closes = [c for c in closes if c is not None]
                return valid_closes if valid_closes else None
                
        except Exception as e:
            logger.debug(f"Yahoo Finance fetch failed for {symbol}: {e}")
            return None
    
    async def fetch_tradfi_data(self) -> Dict[str, Optional[float]]:
        """
        Fetch TradFi indicators from free sources.
        
        Uses Yahoo Finance for VIX, USD/JPY, and SPX.
        
        Returns:
            Dict with 'vix', 'usdjpy', 'spx' values
        """
        data = {}
        
        symbols = {
            'vix': '^VIX',
            'usdjpy': 'JPY=X',  # USD/JPY rate
            'spx': '^GSPC'
        }
        
        async with aiohttp.ClientSession() as session:
            for key, symbol in symbols.items():
                closes = await self._fetch_yahoo_data(session, symbol)
                
                if closes:
                    data[key] = closes[-1]  # Most recent close
                    
                    if key == 'spx':
                        self.spx_history = closes
                    elif key == 'usdjpy':
                        self.usdjpy_history = closes
                else:
                    data[key] = None
        
        return data
    
    def calculate(self, tradfi_data: Dict[str, Optional[float]]) -> Tuple[float, dict]:
        """
        Calculate CACI from TradFi indicators.
        
        Args:
            tradfi_data: Dict with 'vix', 'usdjpy', 'spx' values
        
        Returns:
            Tuple of (caci_score, metadata_dict)
        """
        scores = {}
        raw_values = {}
        
        # VIX component (most important based on Aug 5, 2024 analysis)
        vix = tradfi_data.get('vix')
        if vix is not None:
            raw_values['vix'] = vix
            if vix >= self.config.vix_crisis_threshold:
                scores['vix'] = 1.0
            elif vix >= self.config.vix_elevated_threshold:
                # Linear interpolation
                scores['vix'] = (vix - self.config.vix_elevated_threshold) / \
                                (self.config.vix_crisis_threshold - self.config.vix_elevated_threshold)
            else:
                scores['vix'] = 0.0
        
        # USD/JPY velocity component (detect rapid moves indicating carry unwind)
        usdjpy = tradfi_data.get('usdjpy')
        if usdjpy is not None and len(self.usdjpy_history) >= 2:
            raw_values['usdjpy'] = usdjpy
            # Calculate daily move as % change
            yesterday = self.usdjpy_history[-2] if len(self.usdjpy_history) >= 2 else usdjpy
            daily_move = abs(usdjpy - yesterday) / yesterday if yesterday != 0 else 0
            
            if daily_move >= self.config.usdjpy_move_threshold:
                scores['usdjpy'] = min(daily_move / (self.config.usdjpy_move_threshold * 2), 1.0)
            else:
                scores['usdjpy'] = 0.0
            raw_values['usdjpy_daily_move'] = daily_move * 100
        else:
            scores['usdjpy'] = 0.0
        
        # SPX drawdown component
        spx = tradfi_data.get('spx')
        if spx is not None and len(self.spx_history) >= 2:
            raw_values['spx'] = spx
            recent_high = max(self.spx_history)
            drawdown = (recent_high - spx) / recent_high if recent_high != 0 else 0
            
            if drawdown >= self.config.spx_drawdown_threshold:
                scores['spx'] = min(drawdown / (self.config.spx_drawdown_threshold * 2), 1.0)
            else:
                scores['spx'] = 0.0
            raw_values['spx_drawdown'] = drawdown * 100
            raw_values['spx_recent_high'] = recent_high
        else:
            scores['spx'] = 0.0
        
        if not scores or all(v == 0 for v in scores.values()):
            # Check if we have any data at all
            if not tradfi_data.get('vix') and not tradfi_data.get('spx'):
                return 0.0, {'status': 'no_tradfi_data'}
        
        # CACI = weighted average of components
        # VIX is most important signal based on Aug 5, 2024 analysis
        weights = {'vix': 0.5, 'usdjpy': 0.3, 'spx': 0.2}
        
        caci_score = sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)
        
        # Determine stress level
        if caci_score >= 0.7:
            stress_level = 'CRISIS'
        elif caci_score >= 0.5:
            stress_level = 'HIGH'
        elif caci_score >= 0.3:
            stress_level = 'ELEVATED'
        else:
            stress_level = 'LOW'
        
        metadata = {
            'component_scores': scores,
            'weights': weights,
            'raw_values': raw_values,
            'stress_level': stress_level,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return caci_score, metadata
    
    async def refresh_from_api(self) -> Tuple[float, dict]:
        """
        Fetch TradFi data and calculate CACI.
        
        Returns:
            Tuple of (caci_score, metadata)
        """
        tradfi_data = await self.fetch_tradfi_data()
        return self.calculate(tradfi_data)
