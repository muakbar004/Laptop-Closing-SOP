#!/usr/bin/env python3
"""
custom_alert_tool.py
Edge safety and notification plugin for NVIDIA NanoLLM / Agent Studio.
"""
import logging
import datetime
from typing import Dict, Any

try:
    from nano_llm import Plugin
except ImportError:
    # Lightweight mock class for local laptop development without Jetson packages
    class Plugin:
        def __init__(self, inputs=None, outputs=None, **kwargs):
            self.inputs = inputs or []
            self.outputs_channels = outputs or []
            self.parameters = {}

        def add_parameters(self, **kwargs):
            self.parameters.update(kwargs)

        def output(self, data, channel=0):
            print(f"[Channel {channel} Output]: {data}")


class SafetyAlertPlugin(Plugin):
    """
    Edge dispatch plugin that provides function-calling capabilities to the LLM.
    When a target condition or safety hazard is identified, this plugin formats
    the announcement, routes it to Piper TTS for audible speech, and writes
    a structured incident record.
    """

    def __init__(
        self,
        alert_prefix: str = "Attention: ",
        log_to_stdout: bool = True,
        **kwargs
    ):
        """
        Args:
            alert_prefix (str): Prefix spoken before all generated audio alerts.
            log_to_stdout (bool): Whether to log triggered events to stdout.
        """
        # Expose two output channels in Agent Studio:
        # Channel 0 -> 'text' (routed into Piper TTS plugin)
        # Channel 1 -> 'log' (routed to file, dashboard, or downstream network sink)
        super().__init__(inputs=['input_text'], outputs=['speech_text', 'event_log'], **kwargs)

        self.add_parameters(alert_prefix=alert_prefix, log_to_stdout=log_to_stdout)
        self.alert_prefix = alert_prefix
        self.log_to_stdout = log_to_stdout
        self.history = []

    def trigger_alert(self, event_description: str, severity: str = "warning") -> str:
        """
        Send a real-time safety or SOP compliance alert and speak it over the speakers.
        Call this function whenever a safety hazard occurs or an SOP is violated 
        (e.g., an unattended open laptop, incorrect ingredient portions, etc.).

        Args:
            event_description (str): A concise description of the detected object, vehicle, or event.
            severity (str): Urgency level. Must be one of 'info', 'warning', or 'critical'.

        Returns:
            str: Confirmation message returned to the LLM indicating the dispatch status.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spoken_text = f"{self.alert_prefix}{event_description}. Alert level is {severity}."

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "severity": severity.lower(),
            "event": event_description,
            "spoken_text": spoken_text
        }
        self.history.append(payload)

        if self.log_to_stdout:
            logging.warning(f"\n[ALERT DISPATCHED] [{severity.upper()}] {timestamp} -> {event_description}")

        # Channel 0: Piper TTS receives this text and generates audio immediately
        self.output(spoken_text, channel=0)

        # Channel 1: Structured event record for persistence or webhooks
        self.output(payload, channel=1)

        return f"Successfully dispatched '{severity.upper()}' alert: {spoken_text}"

    def process(self, input, **kwargs):
        """
        Handles direct inputs if another node passes plain text to this plugin.
        """
        if not input:
            return None

        if isinstance(input, str):
            # If string received directly from another node, trigger an info alert
            return self.trigger_alert(event_description=input, severity="info")

        return input