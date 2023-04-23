import transformers
import datasets
import json
import numpy as np
import pdb
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from nltk import sent_tokenize
import re
from string import punctuation

zero_shot_instruction = "Answer the following question by reasoning step-by-step.\n" #Write the answer as a separate sentence starting with 'The answer is'.\n"
few_shot_prefix = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n
"""
# Examples that won't fit in context from CoT paper
# """
# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does
# he have now?
# A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n
# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
# A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n
# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n
# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n"""

def apply_zs_prompt(example, instruction=zero_shot_instruction):
    return instruction + '\nQ: ' + example['question'] + '\n' + 'A:'

def apply_fs_prompt(example, instruction=zero_shot_instruction):
    return few_shot_prefix + '\n' + instruction + '\nQ: ' + example['question'] + '\n' + 'A:'

def generate_math_prompt(prefix, question, instruction):
    prompt = ''
    prompt += prefix
    if len(instruction):
        prompt += instruction + '\n'
    prompt += question
    return prompt

def extract_answer(answer_text, answer_prefix='answer is ', no_prefix=False, overfit=False):
    sentences = sent_tokenize(answer_text)
    ans_candidate = sentences[-1]
    found = True
    if not no_prefix and answer_prefix in ans_candidate:
        answer = ans_candidate.partition(answer_prefix)[2].strip(punctuation)
        try:
            return float(answer)
        except: found = False
    else: found = False
    if no_prefix:
        answers = re.findall(r'\d+', ans_candidate)
        if len(answers):
            return float(answers[-1])
        else: found = False
    if not found:
        if not overfit: return None
        else:
            answer = ans_candidate.partition('=')[2].strip(punctuation)
            try:
                return float(answer)
            except:
                return None

def extract_gold_answer(ans):
    return float(ans.partition('###')[2].replace(',', '').strip(punctuation))


dataset = load_dataset("gsm8k", 'main', split='test')

dataset = dataset.map(lambda example: {'gold': extract_gold_answer(example['answer'])})
dataset = dataset.map(lambda example: {'input_prompt': apply_zs_prompt(example)})
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

num_gen_tokens = 128

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 
if torch.cuda.is_available():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map="auto") 
else:
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
   

count = 0
correct = 0

for batch in tqdm(data_loader):
    input_text = batch['input_prompt']
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    else: 
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=num_gen_tokens)
    op_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    op_answers = [extract_answer(op, no_prefix=True, overfit=True) for op in op_texts]
    correct += sum(np.array(op_answers) == np.array(batch['gold']))
    count += len(batch['gold'])
print('0-shot')
print(round(100*correct/count, 2))

