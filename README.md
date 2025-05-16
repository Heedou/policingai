# Law & Order
Benchmark Dataset for Evaluating Large Language Models in Policing

## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Heedou Kim</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, South Korea</td>
		<td>heedou123@korea.ac.kr</td>
	</tr>
  <tr>
		<td>Mogan Gim</td>		
		<td>Department of Biomedical Engineering,<br>Hankuk University of Foreign Studies, South Korea</td>
		<td>gimmogan@hufs.ac.kr</td>
	</tr>
 	<tr>
		<td>Donghee Choi</td>		
		<td>Department of Metabolism, Digestion and Reproduction, <br>Imperial College London, United Kingdom</td>
		<td>donghee.choi@imperial.ac.uk</td>
	</tr>
   	<tr>
		<td>Soonil Bae</td>		
		<td>Police Science Institute, <br>Korea National Police University, South Korea</td>
		<td>soonil.bae@police.go.kr</td>
	</tr>
   	<tr>
		<td>Miyoung Kim*</td>		
		<td>Department of Computing Science, <br>University of Alberta, Canada</td>
		<td>miyoung2@ualberta.ca</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>
</table>

- &ast;: *Corresponding Author*

# How to Use Dataset

```python
from datasets import load_dataset

# Criminal Hypothesis
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="CI_Criminal_Hypothesis")

print(ds["train"][0])     
print(ds["validation"][0])  
print(ds["test"][0])        

# Statute_Mapping 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="CI_Statute_Mapping")

# Element_Analysis 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="CI_Element_Analysis")

# Fradulent_Intention_Interpretation 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="IA_Fradulent_Intention_Interpretation")

# Fradulent_Scenario_Completion 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="IA_Fradulent_Scenario_Completion")

# Case_Analysis_NER 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="IA_Case_Analysis_NER")

# Deceptive_Message_Analysis 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="IA_Deceptive_Message_Analysis")

# Offense_Detection 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="PO_Offense_Detection")

# Operational_QA 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="PO_Operational_QA")

# Emergency_Reports_Summarization 
ds = load_dataset("PSI-PAIRC/Law_and_Order", name="PT_Emergency_Reports_Summarization")

```


## Link to Dataset
https://huggingface.co/datasets/PSI-PAIRC/Law_and_Order

# Benchmarks

| LLM as                | Task                                | Metric            | GPT4o | Gemini 2.0 | EEVE 10.8B | SOLAR 10.7B | Llama 3.1-8B | Llama 3.2-1B |
|-----------------------|-------------------------------------|-------------------|--------|--------------|--------------|---------------|----------------|----------------|
| Police Officer        | Operational QA                      | LLM-as-a-Judge    | 0.69   | 0.66         | 0.87         | 0.85          | 0.88           | 0.64           |
|                       | Offense Detection                   | ACC               | 0.88   | 0.88         | 0.83         | 0.95          | 0.50           | 0.21           |
|                       |                                     | F1                | 0.96   | 0.96         | 0.96         | 0.98          | 0.77           | 0.61           |
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
