from src.slack.utility import send_message_to_slack
from projects.am1_project.api import api_client
import traceback

try:
   response = api_client.client.get("/health")  
   if response['status'] != "ok":
       send_message_to_slack("Error: AM1 FastAPI model health check returns unexpected HTTP code {response.status_code}.")
   else:
       print("Health check ok")
except Exception as e:
    send_message_to_slack("Error: AM1 FastAPI model is not accessible at deployed location.")
    stack_trace = traceback.format_exc()
    print(stack_trace)
    #send_message_to_slack(str(stack_trace))
        