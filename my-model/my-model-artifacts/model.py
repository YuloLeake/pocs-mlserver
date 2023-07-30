from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd


InputData = List[Dict[str, float]]
OutputData = List[Dict[str, str]]

file_dir = Path(__file__).resolve().parent
iris = joblib.load(file_dir / "artifacts/iris.joblib")
classes = ['setosa', 'versicolor', 'virginica']


def infer(data: InputData):
    data: pd.DataFrame = pd.DataFrame.from_records(data)
    data: np.ndarray = iris.predict(data.to_numpy())
    data: OutputData = [{'class': classes[c]} for c in data.tolist()]
    return data


if __name__ == '__main__':
    data = [
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
    output = infer(data)
    print(output)
