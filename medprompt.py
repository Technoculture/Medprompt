import argparse
from datasets import DatasetDict, load_dataset
import dspy
import openai
import os
import faiss
import tqdm
import random
import joblib
from dspy.teleprompt import KNNFewShot
from dspy.predict.knn import KNN
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate import Evaluate


def model_setting(API_KEY, model_name):

    api_key = os.environ.get(API_KEY)

    model=dspy.OpenAI(model=model, api_key=api_key)
    dspy.settings.configure(lm=model)

def formatting_options(qoptions):
  return [' '.join(f"{option['key']} {option['value']}" for option in options) for options in qoptions]

# We are expecting a single choice answer so signature accordingly.
class MultipleChoiceQA(dspy.Signature):
    """Answer questions with single letter answers."""

    question = dspy.InputField(desc="The multiple-choice question.")
    options = dspy.InputField(desc="The set of options in the format : A option1 B option2 C option3 D option4 E option5 where A corresponds to option1, B to option2 and so on.")
    context = dspy.InputField(desc="may contain relevant facts")
    answer = dspy.OutputField(desc="A single-letter answer corresponding to the selected option.")

class MultipleQABot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MultipleChoiceQA)

    def forward(self, question, options):
        answer = self.generate_answer(question=question,options=options)

        return answer

generate_answer = dspy.ChainOfThought(MultipleChoiceQA)

def store_correct_cot(questions: list[str], option_sets: list[str], answers: list[str]) -> list[str]:
    train_set = []
    for question, options, answer in zip(questions, option_sets, answers):
        pred_response = generate_answer(question=question, options=options)
        if pred_response.answer == answer:
          example = dspy.Example(
            question=question,
            options=options,
            context=pred_response.rationale.split('.', 1)[1].strip(),
            answer=answer
        ).with_inputs("question", "options")

          train_set.append(example)

    return train_set



class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

        return EnsembledProgram()


def ask_questions(program):
    while True:
        q = input("Ask me a question (type 'exit' to end):\n")
        if q.lower() == 'exit':
            break

        o = input("Provide the options in the format A option1 B option2 C option3 and so on.\n")
        pred = program(question=q, options=o)
        print(pred.answer)   

def main(args):
    model_setting(args.api_key,args.model)
    
    #Loading the dataset
    dataset: DatasetDict = (load_dataset(args.dataset))

    #Processing datset
    train_subset = dataset.get("Train",[])
    train_questions = train_subset.get("question",[])
    train_options = train_subset.get("options",[])
    train_answers_id = train_subset.get("answer_idx",[])
    formatted_train_options = formatting_options(train_options)

    #Generating chain of thought
    trainset = store_correct_cot(train_questions, formatted_train_options, train_answers_id)

    #KNN Fewshot
    knn_teleprompter = KNNFewShot(KNN, args.shots, trainset)
    compiled_knn = knn_teleprompter.compile(MultipleQABot(), trainset=trainset)

    #Ensemble
    programs = [compiled_knn]
    ensembled_program = Ensemble(reduce_fn=dspy.majority).compile(programs)

    ask_questions(ensembled_program)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, default="gpt-3.5-turbo", help="Model to be used.")
    parser.add_argument("--dataset", type=str, default="bigbio/med_qa")
    parser.add_argument("--api_key", type=str, help="YOUR_API_KEY")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of shots for knn fewshot.")

    args = parser.parse_args()
    main(args)