import json
import orjson
import requests

endpoint = "http://localhost:8080/v2/models/my-model/versions/v0.0.1/infer"

inputs = [
    {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2,
    },
    {
        'sepal length (cm)': 6.2,
        'sepal width (cm)': 3.4,
        'petal length (cm)': 5.4,
        'petal width (cm)': 2.3,
    },
    {
        'sepal length (cm)': 5.5,
        'sepal width (cm)': 2.5,
        'petal length (cm)': 4.0,
        'petal width (cm)': 1.3,
    },
]
inputs_bytes = json.dumps(inputs)
# inputs_bytes = orjson.dumps(inputs).decode('utf-8')
print(f"full request:")
print(inputs_bytes)

inference_request = {
    "inputs": [
        {
            "name": "echo_request",
            "shape": [len(inputs_bytes)],
            "datatype": "BYTES",
            "data": [inputs_bytes],
        }
    ]
}

# response = requests.post(endpoint, data=inference_request, headers={"Content-Type": "application/json"})
response = requests.post(endpoint, json=inference_request)

print()
print(f"full response:")
print(response.text)
