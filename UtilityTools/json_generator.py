import json
import uuid

unique_id = uuid.uuid4()
text = "             I have an apple. I love to eat. I love to eat and I love you. I love peach. I love melon.           "
text = text.strip()

print(text)
print(unique_id)

data = {"uuid": str(unique_id), "text": text}
with open("train.jsonl", "a") as f:
    f.write(json.dumps(data) + "\n")