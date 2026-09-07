from src.slack.utility import send_message_to_slack
from projects.am1_project.api import api_client
import traceback
import logging
from src.logging.utility import StructuredMessage, setup_logging
import time

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")

try:
   start_time = time.time() 
   response = api_client.client.get("/health")
   latency = time.time() - start_time
   logger.info(StructuredMessage(message='FastAPI heartbeat check',
        application="am1",
        operation_type="heartbeat check",
        latency=latency
        ))
   if response['status'] != "ok":
       send_message_to_slack("Error: AM1 FastAPI model health check returns unexpected HTTP code {response.status_code}.")
       logger.info(StructuredMessage(message='FastAPI heartbeat check failure',
        application="am1",
        operation_type="am1_heartbeat_check_fail"
        ))
   else:
       print("Health check ok")
except Exception as e:
    send_message_to_slack("Error: AM1 FastAPI model is not accessible at deployed location.")
    stack_trace = traceback.format_exc()
    print(stack_trace)
    #send_message_to_slack(str(stack_trace))
        