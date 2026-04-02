from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import json


with open("./projects/am2_project/config/secrets.json") as f:
    secrets = json.load(f)

client = WebClient(token=secrets["SlackBotToken"])

def send_message_to_slack(message): 
    try:
        response = client.chat_postMessage(
            channel=secrets["SlackAlertsChannelID"],
            text=message
        )
        print("Message sent:", response["ts"])
    except SlackApiError as e:
        print("Error:", e.response["error"])