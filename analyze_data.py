import pandas as pd
import json

df = pd.read_csv('data/ai_job_market_insights.csv')

result = {
    "shape": list(df.shape),
    "columns": df.columns.tolist(),
    "dtypes": {col: str(df[col].dtype) for col in df.columns},
    "sample": df.head(3).to_dict('records'),
    "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
    "unique_values": {col: df[col].unique().tolist() for col in df.columns if df[col].nunique() <= 15}
}

with open('dataset_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

print("Analysis saved to dataset_analysis.json")
