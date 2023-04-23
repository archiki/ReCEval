import graphviz
from graphviz import Digraph
import json
import re
import random
from copy import deepcopy
from checklist.perturb import Perturb
import spacy
nlp = spacy.load('en_core_web_sm')
import os
import jsonlines
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

gold_path = 'entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/test.jsonl'
more_path = 'entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_2/test.jsonl'
preds_path = 'entailment_bank/data/processed_data/predictions/emnlp_2021/task1/T5_11B/test.16K_steps.predictions.tsv'

gold_examples = [json.loads(line) for line in open(gold_path, 'r')]
more_examples = [json.loads(line) for line in open(more_path, 'r')]
pred_examples = [line.split('=')[-1].strip() for line in open(preds_path, 'r')]

gold_examples = [g for g in gold_examples if g['proof'].count(';') > 1]
more_examples = [m for m in more_examples if m['proof'].count(';') > 1]

para_model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()
# Can replace with alternate paraphrase models

def get_response(input_text,num_return_sequences,num_beams):
  batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def obtain_paraphrase(phrase):
    num_beams = 5
    num_return_sequences = 5
    paraphrases = get_response(phrase, num_return_sequences, num_beams)
    paraphrase = np.random.choice(paraphrases, 1)[0] 
    paraphrase = paraphrase.strip('.')
    return paraphrase

def process(proof, alt_triples, hypothesis):
    proof = proof.rstrip('; ')
    steps = proof.split(';')
    step_list = []
    triples = deepcopy(alt_triples)
    for step in steps:
        [parents, leaf] = step.split('->')
        parents = parents.split('&')
        parents = [p.strip(' ') for p in parents]
        leaf = leaf.strip()
        leafId = leaf.split(':')[0]
        if 'int' in leafId: 
            leaf_sent = leaf.split(':')[-1].strip()
            triples[leafId] = leaf_sent
        elif leafId == 'hypothesis':
            leaf_sent = hypothesis
            triples[leafId] = leaf_sent
        step_list.append({'parents':parents, 'child': leafId})
    proof = proof + '; '
    return step_list, triples

def reconstruct_proof(steps, sentences):
    proof = ''
    for step in steps:
        proof += " & ".join(step['parents']) + ' -> ' + step['child']
        if 'int' in step['child']:
            proof += ': ' + sentences[step['child']]
        proof += '; '
    return proof

def repeat_steps(in_steps, in_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    repeated_node = steps[idx]['child']
    print(idx, repeated_node)
    key = 'int' + str(len(int_idxs) + 1)
    sentences[key] = sentences[repeated_node]
    steps[idx]['child'] = key
    steps.insert(idx + 1, {'parents': [key], 'child': repeated_node})
    return steps, sentences

def delete_steps(in_steps, in_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    del_node = steps[idx]['child']
    del_parents = steps[idx]['parents']
    del steps[idx]
    for step in steps:
        if del_node in step['parents']:
            step['parents'].extend(del_parents)
            step['parents'] = [p for p in step['parents'] if p != del_node]
    return steps, sentences

def swapped_steps(in_steps, in_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    swap_node = deepcopy(steps[idx]['child'])
    swap_parent = random.choice(steps[idx]['parents'])
    print(swap_node, swap_parent)
    alt_parents = [p for p in steps[idx]['parents'] if p!= swap_parent]
    alt_parents.append(swap_node)
    steps[idx]['parents'] = alt_parents
    steps[idx]['child'] = swap_parent
    for step in steps[idx + 1:]:
        if swap_node in step['parents']:
            step['parents'].append(swap_parent)
            step['parents'] = [p for p in step['parents'] if p != swap_node]
    return steps, sentences

def negate_step(in_steps, in_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    negated_node = steps[idx]['child']
    print(negated_node, sentences[negated_node])
    sentences[negated_node] = Perturb.add_negation(nlp(sentences[negated_node]))
    return steps, sentences

def hallucinate_step(in_steps, in_sentences, extra_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    hallucinate_node = steps[idx]['child']
    print(hallucinate_node, sentences[hallucinate_node])
    potential_sents = [v for (k,v) in extra_sentences.items() if v not in in_sentences.values()]
    sentences[hallucinate_node] = random.choice(potential_sents)
    print(sentences[hallucinate_node])
    return steps, sentences

def paraphrase_steps(in_steps, in_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    repeated_node = steps[idx]['child']
    print(idx, repeated_node)
    key = 'int' + str(len(int_idxs) + 1)
    sentences[key] = obtain_paraphrase(sentences[repeated_node])
    steps[idx]['child'] = key
    steps.insert(idx + 1, {'parents': [key], 'child': repeated_node})
    return steps, sentences

def redundant_steps(in_steps, in_sentences, extra_sentences):
    steps = deepcopy(in_steps)
    sentences = deepcopy(in_sentences)
    int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
    idx = random.choice(int_idxs)
    assert idx < len(steps) - 1
    repeated_node = steps[idx]['child']
    print(idx, repeated_node)
    key = 'int' + str(len(int_idxs) + 1)
    potential_sents = [v for (k,v) in extra_sentences.items() if v not in in_sentences.values()]
    sentences[key] = random.choice(potential_sents)
    steps[idx]['child'] = key
    steps.insert(idx + 1, {'parents': [key], 'child': repeated_node})
    return steps, sentences
    


tree_dest_path = 'perturbed_trees'
if not os.path.exists(tree_dest_path): os.makedirs(tree_dest_path)
random.seed(0)

unperturbed_path = 'ParlAI/projects/roscoe/roscoe_data/unperturbed_ids.json'
custom_unperturbed_ids = {'entailment_bank_synthetic':{}}

perturbation_functions = {'DuplicateOneStep': repeat_steps, 'ParaphraseOneStep': paraphrase_steps, 'RedundantOneStep': redundant_steps, 'RemoveOneStep': delete_steps, 'SwapOneStep': swapped_steps, 'NegateStep': negate_step, 'ExtrinsicHallucinatedStep': hallucinate_step}

for perturb_type, perturb_func in tqdm(perturbation_functions.items()):
    type_unperturbed_ids = random.choices(range(len(gold_examples)), k=126) # unperturbed_ids[perturb_type + "_test.jsonl"]
    custom_unperturbed_ids[perturb_type + "_test.jsonl"] = type_unperturbed_ids
    fname = '50%_' + perturb_type + '_test.jsonl'
    revised_entry = []
    tree_entry = []
    for id, gold_example, more_example in zip(range(len(gold_examples)), gold_examples, more_examples):
        steps, sentences = process(gold_example['proof'], gold_example['meta']['triples'], gold_example['hypothesis'])
        question = ". ".join(list(gold_example['meta']['triples'].values()) + [gold_example['question']]) 
        answer = gold_example['answer']
        int_idxs = [s for s in range(len(steps)) if 'int' in steps[s]['child']]
        if not len(int_idxs): continue

        if id in type_unperturbed_ids:
            perturbed_steps = steps
            perturbed_sentences = sentences
            perturbed = False
        else:
            more_sentences = more_example['meta']['triples']
            perturbed_steps, perturbed_sentences = perturb_func(steps, sentences, more_sentences)
            if steps == perturbed_steps and sentences == perturbed_sentences:
                perturbed = False
                custom_unperturbed_ids[perturb_type + "_test.jsonl"].append(id)
            else: perturbed = True
        tree_entry.append({'perturbed': perturbed, 'perturbations': perturb_type, 'steps':{'original': steps, 'perturbed': perturbed_steps}, 'sentences':{'original': sentences, 'perturbed': perturbed_sentences}, 'written':{'original': original_written_steps, 'perturbed': written_steps}, 'question': question, 'answer': answer})
    with jsonlines.open(os.path.join(tree_dest_path, fname), 'w') as writer:
        writer.write_all(tree_entry)




