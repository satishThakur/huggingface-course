# script which lists all the models from huggingface

from huggingface_hub import HfApi

api = HfApi()

print('making request')
models = list(api.list_models(limit=10, sort="downloads"))
print('request made')
print(len(models))
for model in models:
    print("\n" + "="*50)
    print(f"Model ID: {model.id}")
    print(f"Name: {model.modelId}")
    print(f"Last Modified: {model.lastModified}")
    print(f"Tags: {', '.join(model.tags)}")
    print(f"Pipeline Tag: {model.pipeline_tag if model.pipeline_tag else 'None'}")
    print(f"Private: {model.private}")
    print(f"Downloads: {model.downloads}")
    print(f"Likes: {model.likes}")