import json

text = "I have an apple. I love to eat. I love to eat and I love you. I love peach. I love melon."

list = list(filter(None, text.split(".")))

print("len = ", len(list))

for item in list:
    item = item.lstrip()
    item = item + "."
    print(item)
    data = {"text": item}
    with open("train.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")