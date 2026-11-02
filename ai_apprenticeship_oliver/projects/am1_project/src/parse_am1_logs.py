import json
from datetime import datetime

LOG_FILE = "./logs/am1_log.json"

# Define the time range
start_time = datetime.strptime(
    "2026-06-08 12:00:00,000",
    "%Y-%m-%d %H:%M:%S,%f"
)

end_time = datetime.strptime(
    "2026-06-07 13:00:00,000",
    "%Y-%m-%d %H:%M:%S,%f"
)

confidence_scores = {}
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        #timestamp = datetime.strptime(
        #    entry["time"],
        #    "%Y-%m-%d %H:%M:%S,%f"
        #)
        timestamp = datetime.fromisoformat(entry["time"]).replace(tzinfo=None)
        #if start_time <= timestamp <= end_time and "confidence" in entry["message"].keys():
        if start_time <= timestamp and "confidence" in entry["message"].keys():
            if not isinstance(entry["message"]["confidence"], list):
               studies = entry["message"]["confidence"].keys()
               for study in studies:
                  if study not in confidence_scores.keys():
                    confidence_scores[study]={}
                    topics = entry["message"]["confidence"][study]
                    for topic in topics:
                        if topic not in confidence_scores[study].keys():
                            confidence_scores[study][topic]=[]
                        confidence_scores[study][topic].extend(
                            entry["message"]["confidence"][study][topic]
                        )

                confidence_scores.extend(
                    entry["message"]["confidence"]
            )

print(confidence_scores)
print(f"Retrieved {len(confidence_scores)} confidence scores.")

# the below code converts from pretty print to one entry per line
import json

decoder = json.JSONDecoder()

with open("./logs/am1_log.json", "r") as fin, open("compact.json", "w") as fout:
    text = fin.read()
    pos = 0
    length = len(text)
    while pos < length:
        # Skip whitespace between objects
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        obj, pos = decoder.raw_decode(text, pos)
        fout.write(json.dumps(obj, separators=(",", ":")) + "\n")