# swim/shared/alerting.py

"""Alerting system — sends email/Slack notifications when critical risk is detected."""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx

from swim.shared.config import get_alerting_config

logger = logging.getLogger(__name__)

RISK_SEVERITY = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


class AlertManager:
    """Sends alerts when pipeline risk exceeds configured threshold."""

    def __init__(self):
        self._config = get_alerting_config()

    @property
    def enabled(self) -> bool:
        return self._config.get("enabled", False)

    @property
    def threshold(self) -> str:
        return self._config.get("risk_threshold", "critical")

    def should_alert(self, risk_level: str) -> bool:
        if not self.enabled:
            return False
        return RISK_SEVERITY.get(risk_level, 0) >= RISK_SEVERITY.get(self.threshold, 3)

    async def process_pipeline_result(self, result: Dict[str, Any]):
        """Check pipeline result and send alerts if needed."""
        risk = result.get("risk_assessment", {})
        level = risk.get("level", "unknown")

        if not self.should_alert(level):
            return

        location = result.get("metadata", {}).get("location", {})
        lake = location.get("name", "Unknown")
        score = risk.get("score", 0)
        recommendation = risk.get("recommendation", "")

        subject = f"SWIM Alert: {level.upper()} risk at {lake}"
        body = (
            f"SWIM Platform Risk Alert\n"
            f"========================\n\n"
            f"Location: {lake}\n"
            f"Risk Level: {level.upper()}\n"
            f"Risk Score: {score:.3f}\n"
            f"Recommendation: {recommendation}\n\n"
            f"Evidence:\n"
        )
        for e in risk.get("evidence", []):
            body += f"  - {e}\n"

        body += f"\nTimestamp: {risk.get('timestamp', '')}\n"

        # Send email
        await self._send_email(subject, body)

        # Send Slack
        slack_url = self._config.get("slack_webhook_url")
        if slack_url:
            await self._send_slack(slack_url, subject, body)

        logger.info("Alert sent for %s risk at %s", level, lake)

    async def _send_email(self, subject: str, body: str):
        smtp_host = self._config.get("smtp_host")
        if not smtp_host:
            logger.debug("No SMTP host configured — skipping email alert")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self._config.get("from_email", "swim@localhost")
            msg["Subject"] = subject
            recipients = self._config.get("recipients", [])
            if not recipients:
                return
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, self._config.get("smtp_port", 587)) as server:
                server.ehlo()
                smtp_user = self._config.get("smtp_user")
                smtp_pass = self._config.get("smtp_password")
                if smtp_user and smtp_pass:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                server.sendmail(msg["From"], recipients, msg.as_string())
            logger.info("Email alert sent to %s", recipients)
        except Exception as exc:
            logger.error("Failed to send email alert: %s", exc)

    async def _send_slack(self, webhook_url: str, title: str, body: str):
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={"text": f"*{title}*\n```{body}```"},
                    timeout=10,
                )
            logger.info("Slack alert sent")
        except Exception as exc:
            logger.error("Failed to send Slack alert: %s", exc)


# Global singleton
alert_manager = AlertManager()
