# Medprompt
An optimisable implementation of Medprompt using Dspy.

This repository implements a Multiple Choice Question Answering (QA) system using the MedPrompt prompting strategy and the `dspy` library. The system leverages techniques such as KNN FewShot and ensemble learning to answer medical-related questions accurately with single-letter options.

## Prerequisites

- Python 3.7 or later
- DSPy
- FAISS
- TQDM

## Installation
1. Ensure Python 3.7 or later is installed on your system.
2. Install required packages using pip:
   ```
   pip install openai faiss tqdm
   ```
   Install dspy directly dspy GitHub repository or use the command:
   ```
   pip install git+https://github.com/stanfordnlp/dspy.git
   ```

1. Clone the repository:

```
git clone https://github.com/Technoculture/Medprompt.git
```

## Setup
1. Obtain an API key from OpenAI.
2. Prepare or identify a dataset for training and evaluation.


## Usage
Run the script with the necessary arguments:

```
python3 example.py --model model_name --api_key YOUR_API_KEY --dataset dataset_name --shots number_of_shots
```
Default parameters:
```
Model: gpt-3.5-turbo
Dataset: bigbio/med_qa
Shots: 5
```

## Development

```sh
python medprompt.py --help
```

```sh
# Use the virtual environment
poetry shell

# Format the code
ruff format

# Lint the code
ruff check . --fix
```

## Asking Questions

The system will prompt you to ask questions, and you can provide options.
Type 'exit' to end the questioning session.

## Example
Run command.
```
python3 example.py --api_key API_KEY --dataset "GBaker/MedQA-USMLE-4-options"
```
Input the questions and the options.

Example question 1:
```
Ask me a question (type 'exit' to end):
A 23-year-old woman comes to the physician because she is embarrassed about the appearance of her nails. She has no history of serious illness and takes no medications. She appears well. A photograph of the nails is shown. Which of the following additional findings is most likely in this patient?
Please provide the options.
"A": "Silvery plaques on extensor surfaces", "B": "Flesh-colored papules in the lumbosacral region", "C": "Erosions of the dental enamel", "D": "Holosystolic murmur                               
A
```
Example question 2:
```
Ask me a question (type 'exit' to end):
A 72-year-old man comes to the physician because of a 2-month history of fatigue and worsening abdominal pain. During this period, he also has excessive night sweats and shortness of breath on exertion. Over the past 3 months, he has had a 5.6-kg (12-lb) weight loss. He had a myocardial infarction 3 years ago. He has hypertension, diabetes mellitus, and chronic bronchitis. His medications include insulin, aspirin, lisinopril, and an albuterol inhaler. He has smoked half a pack of cigarettes for the past 45 years. Vital signs are within normal limits. The spleen is palpated 6 cm below the left costal margin. Laboratory studies show: Hemoglobin 6.4 g/dL Mean corpuscular volume 85 μm3 Leukocyte count 5,200/mm3 Platelet count 96,000/mm3 A blood smear is shown. Bone marrow aspiration shows extensive fibrosis and a few scattered plasma cells. A JAK 2 assay is positive. Which of the following is the most appropriate next step in management? 
Please provide the options.
"A": "Cladribine", "B": "Prednisone", "C": "Imatinib", "D": "Ruxolitinib"
D
```
To end program type exit.
```
Ask me a question (type 'exit' to end):
exit
```

## Script Functionality
model_setting: Configures the dspy model with the specified API key and model name.

MultipleChoiceQA and MultipleChoiceQA1: Classes for handling multiple-choice questions.

store_correct_cot: Stores correctly predicted chain-of-thought responses.

MultipleQABot: A dspy module for predicting answers to multiple-choice questions.

Ensemble: Implements an ensemble of models for improved prediction accuracy.

ask_questions: Function to interact with the user, asking for questions and options.

main: The main function is setting up the model, processing the dataset, and starting the question-answering interaction.

## Acknowledgments
DSPy for their various modules.
Hugging Face for the datasets library.
Feel free to reach out if you have any questions or issues!
