import requests
import time
from typing import Any, Dict, Optional
from src.logging.utility import StructuredMessage, setup_logging

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")

class APIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # Default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })

    def _handle_response(self, response: requests.Response) -> Any:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {response.status_code} - {response.text}") from e

        try:
            return response.json()
        except ValueError:
            return response.text

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        return self._handle_response(response)

    def post(self, endpoint: str, data: Optional[Dict] = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, json=data, timeout=self.timeout)
        return response
        #return self._handle_response(response)

    def put(self, endpoint: str, data: Optional[Dict] = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.put(url, json=data, timeout=self.timeout)
        return self._handle_response(response)

    def delete(self, endpoint: str) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.delete(url, timeout=self.timeout)
        return self._handle_response(response)

client = APIClient(
    base_url="http://127.0.0.1:8000"
)

def execute_query(request_body, agency):
    start_time = time.time()
    response = client.post(f"/categorise_items/{agency}", data=request_body)
    latency = time.time() - start_time
    logger.info(StructuredMessage(message='Categorise text http request',
        application="am1",
        operation_type="item_classification_api_request",
        latency=latency,
        http_response_code=response.status_code
        ))
    return response.json()
