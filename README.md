# LAW & ORDER : Benchmark Dataset for Evaluating Large Language Models in Policing

## Overview

Our dataset is Law&Order, the first role-based benchmark for evaluating large language models (LLMs) in real-world policing. 

It defines 11 tasks across four policing roles (Police Officer, Intelligence Analyst, Patrol Officer, Criminal Investigator) under the MKT (Mission–Knowledge–Thinking) framework, with over 210,000 instances annotated by 39 experts.

## Datasets

### Since our dataset is originally constructed in Korean, we provide English samples for all task datasets.

#### 1. Operational QA task
#### Answer procedural or legal questions across all police domains. This task reflects the foundational need for officers to respond accurately to a wide range of operational queries.

<pre>```json
{
"conv_993": {
  "manual": "79. Investigation Manual Focused on Elements of Criminal Offenses - Step 2 Merged (Seongnam Sujeong Police, Kim Seong-taek).xlsx",
  "section_outline": "<Theoretical Part> > VI. Crimes Against Property and Special Provisions for Family Members",
  "content": "VI. Special Provisions for Crimes Against Property Committed by Family Members\n\n1. Definition\nThe special legal provision for crimes among close family members is based on the principle that the law should refrain from intervening in such cases as much as possible. It is applied mutatis mutandis to property crimes (except for damage and robbery) as stipulated under the Criminal Act’s provision on obstruction of rights. Because this provision may result in either a judgment of exemption from punishment (a personal grounds for exclusion of criminal liability) or change the prosecutorial requirements (i.e., requiring a relative’s complaint), it is necessary to check for family relationships when reviewing complaints or statements related to property crimes.\n\n2. Scope of Family Relationship\n- According to case law, the provision applies only when both the owner and the possessor of the property are relatives. For example, if a grandson transfers funds from his grandfather’s bank account to his own via an ATM, the bank—being the victim—is not a relative and thus the provision does not apply.\n- The family relationship must exist at the time of the criminal act. Even if the relationship ends later, it does not affect applicability. Also, when a child born out of wedlock is later recognized (acknowledged) by the biological parent, the recognition applies retroactively and the provision can apply even to crimes committed before the recognition.\n\n※Relevant Laws:\n- Civil Act Article 855 (Acknowledgment of Illegitimate Children)\n- Civil Act Article 860 (Retroactive Effect of Acknowledgment)\n\n3. Legal Effect\nThis provision may lead to a judgment of exemption from punishment or alter the requirements for prosecution (e.g., requiring a complaint from the victim).\n\n4. Special Provision for Crimes of Receiving Stolen Property (Criminal Act Article 365)\n- If the perpetrator and victim of a receiving offense are in a family relationship as defined in Article 328(1)(2), the same rules apply.\n- If the perpetrator and principal offender are family, the punishment may be mitigated or exempted. However, co-offenders who are not family are excluded from this benefit.\n\n5. Application to Fraud and Extortion\nIn the case of litigation fraud, although the deceived party (i.e., the court) is not a relative, if the actor and the actual economic victim are relatives, the provision applies (Case 75Do781).\n※ In embezzlement or breach of trust cases, since the trustor is also a victim, the actor must be related to both the owner and the trustor for the provision to apply.\n\n6. Civil Code Provisions (with inline explanations)\nIncludes definitions and scope of relatives and in-laws, rules for calculating kinship degree, and provisions on termination of family relations due to divorce, annulment, or remarriage.\n\n- Civil Act Article 767: Definition of Relatives\n- Article 768: Definition of Blood Relatives\n- Article 769: Definition of Affinity (In-Laws)\n- Article 770: Calculation of Degree of Blood Relation\n- Article 771: Calculation of Degree of Affinity\n- Article 775: Termination of Affinity\n- Article 777: Scope of Relatives (legal effect limited to 8th-degree blood relatives, 4th-degree in-laws, and spouses)\n- Article 779: Scope of Family (nuclear and extended family depending on cohabitation)\n- Article 855 & 860: Acknowledgment and retroactive recognition of illegitimate children",
  "document_topic": "Special Legal Provisions for Property Crimes Committed by Family Members",
  "question": "How does the special legal provision for family members apply in cases of fraud and extortion?",
  "answer": "In fraud and extortion cases, the special legal provision for family members applies if there is a family relationship between the actor and the actual economic victim. In litigation fraud, even though the deceived entity is the court (not a relative), the provision still applies if the actor and the economic victim are relatives."
},
```</pre>

#### 2. Offense Detection task
#### Predict the offense category from a description of criminal behavior. This task is motivated by the need for police officers to classify crimes and determine whether certain behaviors fall under punishable legal offenses.

<pre>```json
{
"conv_112824": {
  "conversation": "On January 7, 2013, at approximately 18:35, the defendant drove a C Sonata sedan to Dongdae-dong, Dongdae Bridge, and the riverside parking lot in Boryeong City. There, the defendant collided with vehicles owned by D (vehicle E) and F (vehicle G), which were parked in the area. Upon receiving a report, police officers responded to the scene and found the defendant showing visible signs of intoxication—red eyes, a flushed face, and unsteady gait—providing reasonable grounds to suspect that the defendant had been driving under the influence of alcohol. From 19:14 to 19:35 that same day, at the emergency room of H Hospital in Boryeong City, the defendant was asked three times by Officer J of the I Police Substation, Boryeong Police Station, to undergo a breathalyzer test. However, the defendant refused to comply without a valid reason.",
  "output": "Based on the analysis of the conduct described in the alleged criminal facts, the key criminal element is identified as 'Refusal to comply with a police officer's request for a breathalyzer test.'\nAccordingly, the offense can be classified under the crime name 'Violation of the Road Traffic Act (Refusal of Breathalyzer Test).'\nThe relevant legal provisions are Article 148-2(2) and Article 44(2) of the Road Traffic Act.",
  "law": [
    "Article 148-2 (2) of the Road Traffic Act",
    "Article 44 (2) of the Road Traffic Act"
  ],
  "crime_name": "Violation of the Road Traffic Act (Refusal of Breathalyzer Test)",
  "component": [
    "Refusal to comply with a police officer’s request for a breathalyzer test"
  ]
},
```</pre>

#### 3. Fradulent Scenario Detection task
#### Detect whether a given conversation is related to voice phishing. This task supports officers in distinguishing fraudulent content from general civilian interactions.

<pre>```json
{
"conv_7113": {
  "conversation": "Hello, do you happen to know someone named Kim Tae-yoon? No, I don’t think so. He’s a 19-year-old male from Gwangju, Jeolla Province. Have you ever heard of him through acquaintances? The reason I’m calling is that we recently arrested Kim Tae-yoon and 28 others for fraud, and during the arrest, we found a copy of an ID under your name, Mr. Yoon Young-sik, so we’re contacting you for verification. I’m not involved in anything like that. Have you ever submitted a copy of your ID to a credit card company, insurance company, or any financial institution? No, I haven’t. Have you ever lost your ID or had it stolen? No, I haven’t. The reason I ask is because using a copy of your ID, a Meritz Securities account was opened on February 9, 2022, at the Ingye-dong branch in Suwon, Gyeonggi Province. Did you open that account? When did you say? February 9, 2022. A Meritz Securities account. No, that wasn’t me. Hello? That doesn’t sound like me. Aren’t you Mr. Yoon Young-sik, born August 5th? Yes, that’s me. According to the bank, the account is under your name, so are you sure you didn’t open it? Do you know the account number? It starts with 1002. I think I opened that a long time ago. Well, this account was newly opened on February 9, 2022, at the Ingye-dong branch in Suwon—are you certain you didn’t open it? Yes, I’m sure. This was opened a long time ago. I see. We’ll have to investigate further, but the more serious issue is that this account under your name has been used by suspect Kim Tae-yoon to defraud 72 victims for a total of 64.7 million KRW. Using my account? But I checked my transaction history, and there’s nothing. It shows no activity in June. No, the Meritz Securities account opened on February 9, 2022, at the Ingye-dong branch is now frozen. No, I’m still using my account—I don’t know what you’re talking about. You’re saying you’re using the one opened at Ingye-dong? No, I’m using a different one. Oh. That one was opened in 2016. Right. So the one opened on February 9, 2022, is now frozen. I can’t even find that account—so I really don’t understand what you’re saying. This doesn’t make sense.",
  "time": "UTC 2023-02-08 03:37:36",
  "source": "voice",
  "output": "vishing"
},
```</pre>

#### 4. Fradulent Intention Interpretation task
#### Classify the intentions behind a scammer’s speech and explain why the dialogue qualifies as voice phishing. This task supports the need to identify deceptive intent in dialogue patterns common to vishing scams.

<pre>```json
{
"35": {
  "conversation": "So you don’t have one.\nJust to confirm, you're saying you’re not the one who opened that account, correct?\n",
  "intention": [
    "3. Case Introduction - ⑦ Confirming whether the suspicious (borrowed-name) bank account was directly opened by the citizen"
  ],
  "file": "[Seoul Nat'l Univ.]NR1463.m4a.txt",
  "case": null,
  "time": "UTC 2019-05-14 01:16:25"
},
```</pre>

#### 5. Fradulent Scenario Completion task
#### Predict the next likely utterance from a scammer in an ongoing fraud attempt dialogue. This task is designed to support intelligence analysts in understanding and modeling the sequential flow of scripted conversations commonly used in fraud attempts.

<pre>```json
{
"139": {
    "conversation": "During the investigation of the confiscated items, bank accounts under your name from [Bank Name] and [Bank Name] have been discovered.\n",
    "output": "I see. (Do you know anything about these two accounts?)\nUh, upon checking the passbooks, it was confirmed that they were opened around August 2018 at [Address].\nDid you personally receive the issuance of these accounts?",
    "file": "[Seoul National University]NR_0011114_.txt",
    "case": "NR_001111",
    "time": "UTC 2019-04-09 04:11:32"
},
```</pre>

#### 6. Deceptive Message Analysis task
#### Classify the fraudulent tactics used in scam messages by analyzing given SMS content. This task addresses the need to differentiate between types of scam attempts based on SMS messages.

<pre>```json
{
"conv_9476": {
    "conversation": "[International Call][The National Health Insurance] Health checkup results have been sent. Please check the details at https://c07.mt2s.best_x000D_\nSMS/-",
    "output": "Report of spam impersonating an institution"
},
```</pre>

#### 7. Case Analysis NER task
#### Extract named entities from crime-related texts for investigative analysis. This task supports the identification of key investigative elements such as phone numbers or suspect’s names

<pre>```json
{
"conv_13534": {
    "conversation": "At the scene of an arrest involving a financial fraud ring led by a primary suspect, two bankbooks were discovered under the name of Kim Soo-ha : one from Hyundai Motor Securities and one from Lotte Card. We are currently investigating and contacted you during the process. Upon checking the date of account opening, we found that they were opened this year—specifically on August 22nd in Seoul. Did you, Kim Soo-ha, open these accounts yourself? So are you saying that you are unaware of this matter? You were born in 1989, correct? These were identified as 'stress accounts,' which cannot be checked through mobile banking systems. Because of this, more than half of the dozens of people connected to this case are unaware of their involvement. As I mentioned, the accounts were confirmed to have been opened on August 22nd. There have been many victims, and these accounts were used for crimes related to fake (borrowed-name) bankbooks. So, what is the requirement in this case?",
    "source": "voice",
    "output": "At the scene of an arrest involving a financial fraud ring led by a primary suspect, two bankbooks were discovered under the name of KimSoo-ha[B-ps_name]: one from HyundaiMotorSecurities[B-og_organization] and one from LotteCard[B-og_organization]. We are currently investigating and contacted you during the process. Upon checking the date of account opening, we found that they were opened this year—specifically on August 22nd in Seoul. Did you, KimSoo-ha[B-ps_name], open these accounts yourself? So are you saying that you are unaware of this matter? You were born in 1989[B-qt_birth], correct? These were identified as 'stress accounts,' which cannot be checked through mobile banking systems. Because of this, more than half of the dozens of people connected to this case are unaware of their involvement. As I mentioned, the accounts were confirmed to have been opened on August 22nd. There have been many victims, and these accounts were used for crimes related to fake (borrowed-name) bankbooks. So, what is the requirement in this case?"
},
```</pre>

#### 8. Emergency Report Summarization task
#### Summarize emergency call conversations from both the caller’s and the police officer’s perspectives. This task supports rapid understanding of urgent situations for appropriate response planning.

<pre>```json
{
"conv_72": {
  "conversation": "Yes, thank you. I reported a stalking incident yesterday. After that, I went to the police station, and then I received a call and a message from the person saying, 'I'm going to die.' I even received a message saying the person died at [time]. I'm worried because I don’t know whether they are just trying to scare me or if it’s real. I can’t call and have no way to verify the situation. Has the police station received any death reports? We can only verify report history by phone number, but I haven't received any related death report so far. I don’t know what to do in this situation. I was going to meet the person outside, but they messaged saying they would die, saying they’d 'hang themselves to die.' And then I received a message from their number saying they had died. I don’t know their home address, but I do know they went to [store/business name]. The police will be arriving soon. Wait a moment, I need to sort this out. I had contact with the stalker again, and they sent a message saying they would hang themselves, and then a message came saying they were dead at [time]. I’m reporting this now because I’m worried—it might be real or just a threat. What I know for now is that 'the stalker is the same person as before,' and that 'no death report has been received yet.' Let me organize this a little more. Yes, this is the Seoul Metropolitan Police Agency. A report came in from a stalking victim. The perpetrator sent a suicide-threatening message and then a message claiming they died. We are checking the number; no related report exists as of now. The stalker contacted the victim and sent a message threatening suicide, followed by one claiming they had died. The message came from the perpetrator’s number, but they are now unreachable. We haven’t tried calling yet. It's an urgent situation, and we are trying to locate the perpetrator. The person is an acquaintance of the reporter and sent suicidal messages. No report has been filed yet. The reporter's phone number is [phone number]. The stalker was a friend and ex-lover; there had been dating violence. Yes, they were in a past relationship, and now we’re considering a third-party location information request. The person’s name is [name], and the phone number is [phone number]. We’ll now proceed with the location request, but I must give you a legal notice: location results won’t be shared with the requester, and false reports may result in a fine of up to 10 million won. The person is in their 50s, born in 1970, and an investigation is underway. An officer will arrive shortly, so please wait.",
  "output": "[Reporter’s Situation]\n1. The reporter had previously filed a stalking complaint two days ago. The stalker is an ex-boyfriend.\n2. While at the police station, the stalker called and messaged the reporter, expressing remorse and stating intent to commit suicide.\n3. The final message claimed the stalker had died, prompting the reporter to seek confirmation out of concern.\n\n[Police Response]\n1. Initiated procedures to locate the suspect by requesting location information.\n2. Dispatched officers to respond to the reporter’s location."
},
```</pre>

#### 9. Criminal Hypothesis Evaluation task
#### Determine the truth value of a criminal investigation hypothesis and justify it using criminal law and investigative method This task helps investigators assess the legality of actions in case scenarios.

<pre>```json
{
"hypothesis_5062": {
  "hypothesis_id": "5062",
  "year": "2021",
  "subject": "criminal investigation",
  "hypothesis": "A confrontation between a witness and a suspect involves placing them face-to-face so that the witness can confirm whether the suspect is indeed the perpetrator.",
  "hypothesis_answer": "True",
  "premise": "\npremise 1: Types of perpetrator identification procedures (confrontation between witness and suspect)\nIt is possible to place the witness and the suspect face-to-face to confirm whether the suspect is the perpetrator. At this time, the witness is informed that the person they are confronting may or may not be the perpetrator. Before the confrontation, the suspect is notified of the witness's description of the perpetrator, and if the witness or suspect does not wish to proceed, they are not forced to face each other (U.K. practice).\npremise 2: Factors affecting identification accuracy (appropriate homogeneity)\nThis relates to how the control group in a lineup is selected. Usually, people who resemble the suspect are chosen as the control group. However, there are two different strategies here. One is to select individuals who resemble the suspect identified by investigators. In this case, the witness may be biased to identify the suspect most similar to their own description. The second strategy is to select individuals who resemble the description of the perpetrator given by the witness, which can improve identification accuracy compared to the first strategy.\npremise 3: To ensure the credibility of a witness's testimony in perpetrator identification procedures, the witness's statements or descriptions regarding the suspect’s appearance should be recorded in detail beforehand. Then, the suspect should be presented along with others of similar appearance for simultaneous confrontation with the witness. Additionally, the suspect, witness, and other individuals involved should not have any prior contact. To allow later evaluation of evidentiary value, the confrontation process and results should be documented in writing or photographs. The same principles apply to photo-based identification procedures. (Supreme Court 2003Do7033)\npremise 4: In general, showing a witness a single suspect in person or a single photograph to confirm identification can lower the credibility of the witness’s statement due to limitations and inaccuracies of human memory and the implicit suggestion that the person presented is the perpetrator. Unless there are additional circumstances—such as prior acquaintance between the suspect and the victim, or corroborating evidence—the credibility of the witness is considered low. Thus, to enhance credibility, the witness's statements should be recorded in detail beforehand, and multiple individuals with similar appearance, including the suspect, should be simultaneously presented to the witness. Contact between the parties should be avoided, and the process and results must be documented. However, in cases where the witness's memory is fresh immediately after the crime and the procedure is conducted on-site or nearby, one-on-one confrontations may be permitted due to the likelihood of accurate identification and the urgency of solving the crime. (Supreme Court 2008Do12111)\npremise 5: Showing a witness a single suspect or a single photograph in perpetrator identification procedures can suggest, consciously or unconsciously, that the person is the perpetrator, lowering the credibility of the witness’s statement—unless other supporting circumstances exist. Therefore, the witness's testimony should be carefully recorded in advance, suspects and similarly appearing individuals should be presented together, and prior contact must be prevented. The confrontation and results should be documented. These principles apply not only to photo-based procedures but also to video presentation, street identification, and any post-identification procedures. (Supreme Court 2007Do5201)"
},
```</pre>

#### 10. Statute Mapping task
#### Predict the specific criminal law article that applies to a described behavior. This task is formulated as a multilabel multiclass classification problem to support criminal investigators in mapping factual descriptions of unlawful
behavior to the correct provisions in criminal law, aiding in the accurate drafting of reports.

<pre>```json
{
"conv_112824": {
  "conversation": "On January 7, 2013, at approximately 18:35, the defendant drove a C Sonata sedan to Dongdae-dong, Dongdae Bridge, and the riverside parking lot in Boryeong City. There, the defendant collided with vehicles owned by D (vehicle E) and F (vehicle G), which were parked in the area. Upon receiving a report, police officers responded to the scene and found the defendant showing visible signs of intoxication—red eyes, a flushed face, and unsteady gait—providing reasonable grounds to suspect that the defendant had been driving under the influence of alcohol. From 19:14 to 19:35 that same day, at the emergency room of H Hospital in Boryeong City, the defendant was asked three times by Officer J of the I Police Substation, Boryeong Police Station, to undergo a breathalyzer test. However, the defendant refused to comply without a valid reason.",
  "output": "Based on the analysis of the conduct described in the alleged criminal facts, the key criminal element is identified as 'Refusal to comply with a police officer's request for a breathalyzer test.'\nAccordingly, the offense can be classified under the crime name 'Violation of the Road Traffic Act (Refusal of Breathalyzer Test).'\nThe relevant legal provisions are Article 148-2(2) and Article 44(2) of the Road Traffic Act.",
  "law": [
    "Article 148-2 (2) of the Road Traffic Act",
    "Article 44 (2) of the Road Traffic Act"
  ],
  "crime_name": "Violation of the Road Traffic Act (Refusal of Breathalyzer Test)",
  "component": [
    "Refusal to comply with a police officer’s request for a breathalyzer test"
  ]
},
```</pre>

#### 11. Element Analysis task
#### Identify the act element of an offense as recognized in the given factual description of a crime. This task is designed to support investigators in determining the presence of legal elements necessary for criminal liability

<pre>```json
{
"conv_112824": {
  "conversation": "On January 7, 2013, at approximately 18:35, the defendant drove a C Sonata sedan to Dongdae-dong, Dongdae Bridge, and the riverside parking lot in Boryeong City. There, the defendant collided with vehicles owned by D (vehicle E) and F (vehicle G), which were parked in the area. Upon receiving a report, police officers responded to the scene and found the defendant showing visible signs of intoxication—red eyes, a flushed face, and unsteady gait—providing reasonable grounds to suspect that the defendant had been driving under the influence of alcohol. From 19:14 to 19:35 that same day, at the emergency room of H Hospital in Boryeong City, the defendant was asked three times by Officer J of the I Police Substation, Boryeong Police Station, to undergo a breathalyzer test. However, the defendant refused to comply without a valid reason.",
  "output": "Based on the analysis of the conduct described in the alleged criminal facts, the key criminal element is identified as 'Refusal to comply with a police officer's request for a breathalyzer test.'\nAccordingly, the offense can be classified under the crime name 'Violation of the Road Traffic Act (Refusal of Breathalyzer Test).'\nThe relevant legal provisions are Article 148-2(2) and Article 44(2) of the Road Traffic Act.",
  "law": [
    "Article 148-2 (2) of the Road Traffic Act",
    "Article 44 (2) of the Road Traffic Act"
  ],
  "crime_name": "Violation of the Road Traffic Act (Refusal of Breathalyzer Test)",
  "component": [
    "Refusal to comply with a police officer’s request for a breathalyzer test"
  ]
},
```</pre>

## Data Structure
### All data are in JSON file format and follow the structure shown below. Each key labeled with `conv_num` corresponds to a single data instance.  



<pre> data = { 
            "conv_number": { "input" :  ```, "output" : '''},
            "conv_number": { "input" :  ```, "output" : '''},
            ....
              }
</pre>

## How to read Json file
### To read the JSON files, please use the sample code provided below.  

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

## Training
### To train smaller Language Models, you can use the code provided below. Refer to the example in prepare_train.yaml and configure the settings for the model you want to train.
```python
python train_laworer.py -oc [model name]
```

## Evaluation
### For the evaluation of models using each task dataset, we also provide English Instruction Templates. The templates for each task are presented in Appendix D of the paper.
```The evaluation template in Korean is provided in dataset.py.```

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
