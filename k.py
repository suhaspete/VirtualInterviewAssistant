from google.generativeai import list_models

models = list(list_models())
for model in models:
    print(model.name)
