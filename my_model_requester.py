import json
import requests

model = "http://localhost:8080/v2/models/my-model/versions/v0.0.1"

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

# response = requests.post(f'http://localhost:8080/v2/repository/index', json={})

# exit(0)

inputs_bytes = json.dumps(inputs)

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
print(f"full request:")
print(inference_request)

response = requests.post(f'{model}/infer', json=inference_request)

print()
print(f"full response:")
print(response.text)
