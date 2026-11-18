import json
import os

# Define categories and intent distribution
categories = [
    {
        "department": "shipping", 
        "intents": ["package_status", "missing_delivered_package", "address_change", "damaged_item"]
    },
    {
        "department": "account_support", 
        "intents": ["password_reset", "2fa_lockout", "email_update", "deactivate_account"]
    },
    {
        "department": "billing", 
        "intents": ["double_charge", "subscription_cancel", "invoice_request", "refund_status"]
    },
    {
        "department": "security", 
        "intents": ["unauthorized_login", "suspicious_activity", "api_key_leak"]
    }
]

# Example template to bootstrap raw records
raw_records = [
    {
        "text": "The tracking says my order was delivered to my front porch, but I checked everywhere and nothing is here.",
        "routing": {
            "department": "shipping",
            "priority": "high",
            "intent": "missing_delivered_package",
            "requires_refund": True
        }
    },
    {
        "text": "My authenticator app codes keep getting rejected even though my clock is synced.",
        "routing": {
            "department": "security",
            "priority": "high",
            "intent": "2fa_lockout",
            "requires_refund": False
        }
    }
]

# Expand or load your corpus to reach 1,000+ records, then save to raw JSONL
os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/raw_domain_data.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    # In a full run, you would loop to generate or populate 1000+ items
    for record in raw_records * 500: # Placeholder replication for demonstration
        f.write(json.dumps(record) + "\n")

print(f"[INFO] Generated raw dataset at {output_path}")
