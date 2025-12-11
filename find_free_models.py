import requests
import json

try:
    response = requests.get("https://openrouter.ai/api/v1/models")
    data = response.json()
    
    free_models = []
    for model in data['data']:
        try:
            # Check for explicitly free pricing (some might be string '0' or number 0)
            prompt = float(model['pricing']['prompt'])
            completion = float(model['pricing']['completion'])
            
            if prompt == 0 and completion == 0:
                free_models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'context_length': model['context_length']
                })
        except:
            continue
            
    # Sort by context length as a proxy for capability/modernness, or just print all
    print(f"Found {len(free_models)} free models:")
    for m in free_models:
        print(f"- {m['name']} ({m['id']}) [Ctx: {m['context_length']}]")

except Exception as e:
    print(f"Error: {e}")
