import json
import logging

class StructuredMessage:
    def __init__(self, /, **kwargs):
        self.kwargs = kwargs
    def __str__(self):
        return (json.dumps(self.kwargs))
        #return str(self.kwargs)


def setup_logging():
    logger = logging.getLogger("am2_project")
    print("NAME IN UTILITY")
    print(logger.name)
    logger.setLevel(logging.INFO)    
    handler=logging.FileHandler("logs/am2_log.json")
    formatter=logging.Formatter("{\"time\": \"%(asctime)s\", \"level\": [\"%(levelname)s\"], \"message\": %(message)s}")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
