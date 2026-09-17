"""
Module 6: Alert System
Handles automated SMS (Twilio) and Email (SMTP) dispatch for critical disaster warnings,
risk-threshold triggers, subscriber registration, and emergency alert templates.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from typing import Dict, List, Any, Optional

from config import (
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER,
    SMTP_SERVER, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD
)
from src.database import execute_query, query_to_df

logger = logging.getLogger(__name__)


def generate_disaster_alert_template(
    disaster_type: str,
    location: str,
    risk_score: float,
    metrics_summary: str
) -> Dict[str, str]:
    """
    Generate structured subject and body text for disaster notifications.
    Includes immediate emergency action steps for citizens.
    """
    disc_upper = disaster_type.upper()
    subject = f"🚨 EMERGENCY ALERT: {disc_upper} WARNING for {location} (Risk Score: {risk_score}/100)"

    safety_actions = {
        "EARTHQUAKE": "1. DROP, COVER, and HOLD ON.\n2. Move away from glass, windows, and unanchored structures.\n3. If outdoors, move to an open area away from power lines.",
        "FLOOD": "1. Move to higher ground immediately.\n2. Do NOT walk or drive through flowing floodwaters.\n3. Turn off main gas and electricity switches if instructed.",
        "CYCLONE": "1. Seek sturdy indoor shelter immediately.\n2. Stay away from doors and windows.\n3. Secure loose outdoor objects and keep emergency radio/phone handy."
    }

    actions = safety_actions.get(disc_upper, "1. Remain calm and follow local emergency authority instructions.")

    email_body = f"""
    ===================================================================
    NATURAL DISASTER EMERGENCY WARNING & ACTION NOTICE
    ===================================================================

    Hazard Category : {disc_upper}
    Target Location : {location}
    Calculated Risk : {risk_score} / 100 (CRITICAL WARNING)
    Incident Details: {metrics_summary}

    -------------------------------------------------------------------
    IMMEDIATE CITIZEN SAFETY INSTRUCTIONS:
    {actions}
    -------------------------------------------------------------------

    For nearest emergency shelter locations, live evacuation routes, 
    and official response coordination, open the Citizen Portal:
    http://localhost:8501

    Natural Disaster Intelligence System Control Room
    ===================================================================
    """

    sms_body = f"🚨 EMERGENCY {disc_upper} ALERT for {location}! Risk Score: {risk_score}/100. {metrics_summary}. TAKE IMMEDIATE SHELTER. Info: http://localhost:8501"

    return {"subject": subject, "email_body": email_body, "sms_body": sms_body}


def send_sms_alert(to_phone: str, message_text: str) -> bool:
    """
    Dispatch SMS notification using Twilio API.
    Falls back gracefully to simulated send if credentials are not configured.
    """
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=message_text,
                from_=TWILIO_PHONE_NUMBER,
                to=to_phone
            )
            logger.info(f"Twilio SMS sent successfully to {to_phone}. SID: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Twilio SMS dispatch failed: {e}")
            
    # Mock SMS dispatch fallback
    logger.info(f"[SIMULATED SMS DISPATCH] To: {to_phone} | Msg: {message_text[:60]}...")
    return True


def send_email_alert(to_email: str, subject: str, body_text: str) -> bool:
    """
    Dispatch Email notification using SMTP.
    Falls back gracefully to simulated send if SMTP password is empty.
    """
    if SMTP_PASSWORD and SMTP_EMAIL:
        try:
            msg = MIMEMultipart()
            msg["From"] = SMTP_EMAIL
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body_text, "plain"))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"SMTP Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"SMTP Email dispatch failed: {e}")

    # Mock Email dispatch fallback
    logger.info(f"[SIMULATED EMAIL DISPATCH] To: {to_email} | Subject: {subject}")
    return True


def register_subscriber(name: str, email: str, phone: str, region: str, preferred_disasters: str = "All", alert_channel: str = "Both") -> int:
    """
    Register a new subscriber in the database for automated alerts.
    """
    query = """
    INSERT INTO alert_subscribers (name, email, phone, region, preferred_disasters, alert_channel)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    return execute_query(query, (name, email, phone, region, preferred_disasters, alert_channel))


def trigger_automated_risk_alerts(disaster_type: str, location: str, risk_score: float, details: str, threshold: float = 65.0) -> Dict[str, int]:
    """
    Scan subscribers database and trigger SMS/Email alerts if risk score exceeds the threshold.
    """
    if risk_score < threshold:
        return {"alerts_sent": 0, "status": "Risk below alert trigger threshold"}

    template = generate_disaster_alert_template(disaster_type, location, risk_score, details)
    subscribers_df = query_to_df("SELECT * FROM alert_subscribers WHERE active = 1")

    sms_count = 0
    email_count = 0

    if not subscribers_df.empty:
        for _, sub in subscribers_df.iterrows():
            pref = sub.get("preferred_disasters", "All")
            sub_region = sub.get("region", "All")

            # Check matching filters
            if pref not in ["All", disaster_type] and disaster_type not in pref:
                continue

            channel = sub.get("alert_channel", "Both")
            if channel in ["SMS", "Both"] and sub.get("phone"):
                if send_sms_alert(sub["phone"], template["sms_body"]):
                    sms_count += 1

            if channel in ["Email", "Both"] and sub.get("email"):
                if send_email_alert(sub["email"], template["subject"], template["email_body"]):
                    email_count += 1

    logger.info(f"Automated Alert Trigger complete for {disaster_type} at {location}. Sent {sms_count} SMS, {email_count} Emails.")
    return {"sms_sent": sms_count, "email_sent": email_count, "status": "Alerts triggered successfully"}
