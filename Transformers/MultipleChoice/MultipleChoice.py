from datasets import load_dataset


swag = load_dataset("swag", "regular")
train_swag_dataset = swag["train"];
valid_swag_dataset = swag["validation"];
test_swag_dataset = swag["test"];

train_swag_dataset = train_swag_dataset.select(range(10000));
print(len(train_swag_dataset))
print(len(valid_swag_dataset))
print(len(test_swag_dataset))

print(train_swag_dataset[0])
