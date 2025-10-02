# Law & Order
Benchmark Dataset for Evaluating Large Language Models in Policing

## Overview of Datasets

* We provide English samples for all task datasets.

* All data are in JSON file format and follow the structure shown below. Each key labeled with `conv_num` corresponds to a single data instance.  



<pre> data = { 
            "conv_number": { "input" :  ```, "output" : '''},
            "conv_number": { "input" :  ```, "output" : '''},
            ....
              }
</pre>

* To read the JSON files, please use the sample code provided below.  

```python
import json

# Read JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# Store keys in a list
keys = list(data.keys())
print("All keys list:", keys)

# Access data for each key
for key in keys:
    print(f"Data for {key}:", data[key])
```


## Link to Dataset
https://drive.google.com/file/d/16RTs00zWr1H8vFOU0Wgt0IklXAoDyXNN/view?usp=sharing

## Benchmarks Evaluation Results

| LLM as                | Task                                | Metric            | GPT4o | Gemini 2.0 | EEVE 10.8B | SOLAR 10.7B | Llama 3.1-8B | Llama 3.2-1B |
|-----------------------|-------------------------------------|-------------------|--------|--------------|--------------|---------------|----------------|----------------|
| Police Officer        | Operational QA                      | LLM-as-a-Judge    | 0.69   | 0.66         | 0.87         | 0.85          | 0.88           | 0.64           |
|                       | Offense Detection                   | ACC               | 0.86   | 0.86         | 0.87         | 0.98          | 0.50           | 0.21           |
|                       |                                     | F1                | 0.90   | 0.93         | 0.95         | 0.99          | 0.77           | 0.61           |
| Intelligence Analyst  | Fraudulent Scenario Detection       | ACC               | 0.97   | 0.87         | 0.99         | 0.99          | 0.86           | 0.63           |
|                       |                                     | F1                | 0.97   | 0.88         | 0.99         | 0.99          | 0.85           | 0.58           |
|                       | Fraudulent Scenario Completion      | LLM-as-a-Judge    | 0.70   | 0.66         | 0.67         | 0.71          | 0.71           | 0.64           |
|                       | Fraudulent Intention Interpretation | ACC               | 0.11   | 0.16         | 0.19         | 0.14          | 0.14           | 0.04           |
|                       |                                     | F1 (micro)        | 0.51   | 0.79         | 0.64         | 0.47          | 0.56           | 0.27           |
|                       | Deceptive Message Analysis          | ACC               | 0.88   | 0.93         | 0.97         | 0.99          | 0.97           | 0.88           |
|                       |                                     | F1 (macro)        | 0.70   | 0.76         | 0.91         | 0.98          | 0.95           | 0.73           |
|                       |                                     | F1 (micro)        | 0.88   | 0.93         | 0.97         | 0.99          | 0.97           | 0.88           |
|                       | Case Analysis NER                   | Precision         | 0.17   | 0.14         | 0.31         | 0.52          | 0.17           | 0.08           |
|                       |                                     | F1 (macro)        | 0.17   | 0.14         | 0.22         | 0.29          | 0.16           | 0.06           |
|                       |                                     | F1 (micro)        | 0.46   | 0.44         | 0.06         | 0.11          | 0.04           | 0.03           |
|                       |                                     | F1 (weighted avg) | 0.51   | 0.49         | 0.26         | 0.42          | 0.22           | 0.11           |
| Patrol Officer        | Emergency Reports Summarization     | LLM-as-a-Judge    | 0.89   | 0.75         | 0.62         | 0.56          | 0.51           | 0.20           |
| Criminal Investigator | Criminal Hypothesis                 | ACC               | 0.73   | 0.62         | 0.74         | 0.62          | 0.62           | 0.62           |
|                       |                                     | F1                | 0.79   | 0.77         | 0.79         | 0.77          | 0.77           | 0.77           |
|                       | Statute Mapping                     | ACC               | 0.43   | 0.40         | 0.86         | 0.88          | 0.19           | 0.07           |
|                       |                                     | F1                | 0.65   | 0.69         | 0.92         | 0.95          | 0.35           | 0.12           |
|                       | Element Analysis                    | ACC               | 0.67   | 0.81         | 0.66         | 0.71          | 0.64           | 0.12           |
|                       |                                     | F1                | 0.84   | 0.93         | 0.81         | 0.88          | 0.82           | 0.24           |

# Licensing Information
Licensed under the CC BY-NC 4.0
