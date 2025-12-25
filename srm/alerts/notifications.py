"""
SRM Alert Notifications

Notification system for critical risk events.
Supports Telegram, Discord, and logging.
"""

from typing import Optional
from datetime import datetime
import aiohttp
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Notification manager for SRM critical events.
    
    Supports:
    - Telegram bot notifications
    - Discord webhook notifications
    - Critical logging
    """
    
    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        timeout_seconds: float = 10.0
    ):
        self.telegram_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook_url
        self.timeout = timeout_seconds
        
        self._telegram_enabled = bool(telegram_bot_token and telegram_chat_id)
        self._discord_enabled = bool(discord_webhook_url)
        
        if self._telegram_enabled:
            logger.info("Telegram notifications enabled")
        if self._discord_enabled:
            logger.info("Discord notifications enabled")
    
    async def send_telegram(self, message: str) -> bool:
        """
        Send message via Telegram bot.
        
        Args:
            message: Message text (supports Markdown)
        
        Returns:
            True if successful
        """
        if not self._telegram_enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    return resp.status == 200
                    
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    async def send_discord(self, message: str, color: int = 0xFF0000) -> bool:
        """
        Send message via Discord webhook.
        
        Args:
            message: Message text
            color: Embed color (default: red for alerts)
        
        Returns:
            True if successful
        """
        if not self._discord_enabled:
            return False
        
        try:
            payload = {
                "embeds": [{
                    "title": "🚨 HIMARI SRM Alert",
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    return resp.status in [200, 204]
                    
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "CRITICAL"
    ) -> bool:
        """
        Send alert to all configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Severity level (INFO, WARNING, CRITICAL)
        
        Returns:
            True if at least one channel succeeded
        """
        # Format message
        emoji = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "WARNING" else "ℹ️"
        formatted = f"{emoji} *{title}*\n\n{message}\n\n`{datetime.utcnow().isoformat()}`"
        
        # Log regardless of channel success
        log_func = logger.critical if severity == "CRITICAL" else logger.warning if severity == "WARNING" else logger.info
        log_func(f"[ALERT] {title}: {message}")
        
        # Send to all channels
        results = []
        
        if self._telegram_enabled:
            results.append(await self.send_telegram(formatted))
        
        if self._discord_enabled:
            color = 0xFF0000 if severity == "CRITICAL" else 0xFFFF00 if severity == "WARNING" else 0x00FF00
            results.append(await self.send_discord(message, color))
        
        return any(results) if results else True  # True if no channels configured (log-only)
    
    async def send_emergency_alert(
        self,
        symbol: str,
        score: float,
        regime: str,
        action: str = "HALT"
    ) -> bool:
        """
        Send emergency risk alert.
        
        Args:
            symbol: Trading symbol
            score: Current risk score
            regime: Current market regime
            action: Action being taken
        
        Returns:
            True if sent successfully
        """
        title = f"EMERGENCY {action}: {symbol}"
        message = (
            f"Risk Score: `{score:.3f}` (CRITICAL)\n"
            f"Regime: `{regime}`\n"
            f"Action: `{action}`\n\n"
            f"All positions may be liquidated."
        )
        
        return await self.send_alert(title, message, "CRITICAL")
    
    async def send_regime_change_alert(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
        score: float
    ) -> bool:
        """
        Send regime transition alert.
        
        Args:
            symbol: Trading symbol
            old_regime: Previous regime
            new_regime: New regime
            score: Current risk score
        
        Returns:
            True if sent successfully
        """
        title = f"Regime Change: {symbol}"
        message = (
            f"Transition: `{old_regime}` → `{new_regime}`\n"
            f"Current Score: `{score:.3f}`"
        )
        
        severity = "WARNING" if new_regime != "normal" else "INFO"
        return await self.send_alert(title, message, severity)
    
    async def send_threshold_breach_alert(
        self,
        symbol: str,
        threshold: str,
        score: float,
        action: str
    ) -> bool:
        """
        Send threshold breach alert.
        
        Args:
            symbol: Trading symbol
            threshold: Which threshold was breached
            score: Current risk score
            action: Action being taken
        
        Returns:
            True if sent successfully
        """
        title = f"Threshold Breach: {symbol}"
        message = (
            f"Threshold: `{threshold}`\n"
            f"Score: `{score:.3f}`\n"
            f"Action: `{action}`"
        )
        
        severity = "CRITICAL" if threshold == "HALT" else "WARNING"
        return await self.send_alert(title, message, severity)
