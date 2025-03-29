from transformers import pipeline

llm = pipeline(model="bert-base-uncased")
print(llm.model)
print(llm.model.config)