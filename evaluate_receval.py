from supar import Parser
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import pdb
import json
import tqdm
def tqdm_replacement(iterable_object,*args,**kwargs):
    return iterable_object
tqdm_copy = tqdm.tqdm
tqdm.tqdm = tqdm_replacement
import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from scipy.stats import somersd
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import re, string
import logging
import logging.config
from datasets import load_dataset, load_metric, Dataset
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
import pandas as pd


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
disable_progress_bar()
random.seed(1)


srl_predictor = predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ent_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ent_tokenizer = AutoTokenizer.from_pretrained(ent_model_name)
ent_model = AutoModelForSequenceClassification.from_pretrained(ent_model_name).to(device)

# Intra-Step PVI arguments
inp_model_dir = 'PVI/inp_models/'
no_inp_model_dir = 'PVI/noinp_models/'
# Infor-gain PVI arguments
info_gain_model_dir = 'PVI/infogain_models/'

max_input_length = 512
max_target_length = 64
padding = "max_length"
model_name = "t5-large"
label_pad_token_id = -100
pad_token = '<pad>'
prefix = 'Generate entailed sentence: '

inp_tokenizer = AutoTokenizer.from_pretrained(inp_model_dir)
inp_config = AutoConfig.from_pretrained(inp_model_dir)
inp_model = AutoModelForSeq2SeqLM.from_pretrained(inp_model_dir, config=inp_config)
inp_model.cuda().eval()
no_inp_tokenizer = AutoTokenizer.from_pretrained(no_inp_model_dir)
no_inp_config = AutoConfig.from_pretrained(no_inp_model_dir)
no_inp_model = AutoModelForSeq2SeqLM.from_pretrained(no_inp_model_dir, config=no_inp_config)
no_inp_model.cuda().eval()

info_gain_tokenizer = AutoTokenizer.from_pretrained(info_gain_model_dir)
info_gain_config = AutoConfig.from_pretrained(info_gain_model_dir)
info_gain_model = AutoModelForSeq2SeqLM.from_pretrained(info_gain_model_dir, config=info_gain_config)
info_gain_model.cuda().eval()
info_gain_mname = 'gpt2'


def init_gpt2():
    ll_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    ll_model = AutoModelForCausalLM.from_pretrained('gpt2-xl')
    ll_model.eval().cuda()
    ll_tokenizer.padding_side = "left"
    ll_tokenizer.pad_token = ll_tokenizer.eos_token
    ll_model.config.pad_token_id = ll_model.config.eos_token_id
    return ll_model, ll_tokenizer

def inti_t5():
    ll_tokenizer = AutoTokenizer.from_pretrained("t5-large")
    config = AutoConfig.from_pretrained("t5-large")
    ll_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large", config=config)
    ll_model.eval().cuda()
    ll_tokenizer.pad_token = pad_token
    ll_model.config.pad_token_id = pad_token
    return ll_model, ll_tokenizer

# For Info-Gain PVI
if info_gain_mname == 'gpt2': ll_model, ll_tokenizer = init_gpt2()
elif info_gain_mname == 't5-large': ll_model, ll_tokenizer = inti_t5()

def obtain_entailment_scores(premise, hypothesis):
    input = ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = ent_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    return prediction['entailment']

def obtain_contradiction_scores(premise, hypothesis):
    input = ent_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = ent_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    return prediction['contradiction']

def obtain_unit_entailment_score(prem_units, conc_units):
    if len(prem_units):
        premise = ' and '.join(prem_units)
        hypothesis = ' and '.join(conc_units)
        score = obtain_entailment_scores(premise, hypothesis)
    else:
        score = 1
    return score

def obtain_contradiction_score(prem_units, conc_units):
    pair_scores = []
    hypothesis = ' and '.join(conc_units)
    for premise in prem_units:
        pair_scores.append(obtain_contradiction_scores(premise, hypothesis))
    if len(pair_scores):
        score = 1 - max(pair_scores)
    else:
        score = 1
       
    return score


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

def verb_modifiers(desc):
    filtered_mods = []
    mods = re.findall(r"\[ARGM.*?\]", desc)
    if not len(mods): return filtered_mods
    for mod in mods:
        phrase = mod.split(': ')[1].rstrip(']')
        verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(phrase))]
        if sum(verb_match) and len(phrase.split()) > 2: filtered_mods.append(phrase) # put in a length criteria
    return filtered_mods
    
def remove_modifiers(sent, modifiers):
    if not len(modifiers): return sent
    for mod in modifiers:
        sent = sent.replace(mod, "")
        sent = re.sub(' +', ' ', sent) # remove any double spaces
        sent = sent.strip(string.punctuation + ' ') # remove stray punctuations
    return sent

def extract_frame(tags, words, desc):
    prev = 'O'
    start, end = None, None
    if len(set(tags)) == 1: return ''
    tags = [t if 'C-ARG' not in t else 'O' for t in tags] #check if the modifier is a verb phrase
    for w in range(len(words)):
        if 'B-' in tags[w] and start is None: start = w
        if tags[len(words) - w -1]!='O' and end is None: end = len(words) - w -1 
    
    if end is None: end = start
    sent = detokenize(words[start: end + 1]).rstrip('.')
    return sent


def get_phrases(sent):
    # Simple RCU extractor without conjunction check for premises
    phrases = []
    history = ''
    srl_out = predictor.predict(sent) 
    words = srl_out['words']  
    frames = [s['tags'] for s in srl_out['verbs']]
    descs = [s['description'] for s in srl_out['verbs']]
    mod_sent = detokenize(words).rstrip('.')
    for frame, desc in zip(frames, descs):
        phrase = extract_frame(frame, words, desc)
        if phrase == mod_sent: phrase = remove_modifiers(phrase, verb_modifiers(desc))
        phrases.append(phrase)
    phrases.sort(key=lambda s: len(s), reverse=True)
    filtered_phrases = []
    for p in phrases: 
        if p not in history:  
            history += ' ' + p
            filtered_phrases.append(p)
    if len(filtered_phrases): 
        filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        left = mod_sent
        mod_filt = False
        for fp in filtered_phrases: left = left.replace(fp, '#').strip(string.punctuation + ' ')
        for l in left.split('#'): 
            l = l.strip(string.punctuation + ' ')
            if len(l.split()) >=4 and l not in " ".join(filtered_phrases): 
                verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(l))]
                if sum(verb_match):
                    filtered_phrases.append(l)
                    mod_filt = True
        if mod_filt: filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        return filtered_phrases
    else: return [sent.rstrip('.')]

def get_sent_phrases(para):
    sentences = sent_tokenize(para)
    phrases = []
    for sent in sentences:
        phrases.extend(get_phrases(sent))
    return phrases

def get_reasoning_chain_text(steps, sentences):
    # If using the reasoning trees directly
    step_texts = []
    covered_nodes = []
    for step in steps:
        parent_text = " and ".join([sentences[p] for p in step['parents'] if p not in covered_nodes])
        if len(parent_text): step_text = parent_text + ', so ' + sentences[step['child']] + "."
        else: step_text =  'so ' + sentences[step['child']] + '.' 
        covered_nodes.extend(step['parents']); covered_nodes.append(step['child'])
        step_texts.append(step_text)
    return step_texts


def preprocess_and_convert(premise_units, conc_units): 
    data = {'inputs': [], 'labels': []} 
    parent_text = " & ".join(premise_units) + ' ->'
    child_text = " " + conc_units[0] # assume just one conc unit
    data['inputs'].append(parent_text)
    data['labels'].append(child_text)
    return data

def postprocess_test_data(examples):
    inputs = [prefix + text for text in examples['inputs']]
    model_inputs = inp_tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with inp_tokenizer.as_target_tokenizer():
        targets = [pad_token + label for label in examples['labels']]
        labels = inp_tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def noinp_postprocess_test_data(examples):
    inputs = [prefix + 'None ->' for text in examples['inputs']]
    model_inputs = no_inp_tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with no_inp_tokenizer.as_target_tokenizer():
        targets = [pad_token + label for label in examples['labels']]
        labels = no_inp_tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def obtain_log_prob(predict_dataset, model, tokenizer):
    logits = model(input_ids=torch.Tensor(predict_dataset['input_ids']).long().cuda(), attention_mask=torch.Tensor(predict_dataset['attention_mask']).long().cuda(),  
                    decoder_input_ids=torch.Tensor(predict_dataset['decoder_input_ids']).long().cuda(), decoder_attention_mask = torch.Tensor(predict_dataset['decoder_attention_mask']).long().cuda()).logits
    all_logprobs = torch.log(torch.softmax(logits, dim=-1))
    labels = tokenizer(predict_dataset['labels'], max_length=max_target_length).input_ids
    filter_sums = []
    for row, label in zip(all_logprobs, labels):
        label.pop()
        row = row[:len(label), :].detach().cpu().numpy()
        vocab_size = row.shape[-1]
        loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
        try: summed_logprob = np.sum(row, where = loc)
        except: import pdb; pdb.set_trace()
        filter_sums.append(summed_logprob/len(label))
    return np.array(filter_sums)

def obtain_unit_pvi_score(premise_units, conc_units):
    dataset = Dataset.from_dict(preprocess_and_convert(premise_units, conc_units))
    inp_dataset = dataset.map(postprocess_test_data, batched=True, remove_columns=['inputs'])
    no_inp_dataset = dataset.map(noinp_postprocess_test_data, batched=True, remove_columns=['inputs'])
    inp_logprob = obtain_log_prob(inp_dataset, inp_model, inp_tokenizer)[0]
    no_inp_logprob = obtain_log_prob(no_inp_dataset, no_inp_model, no_inp_tokenizer)[0]
    return inp_logprob - no_inp_logprob
    

def slice_select_logits(all_logprobs, label):
    filter_sums = []
    if info_gain_mname == 'gpt2': row = all_logprobs[-len(label):, :].detach().cpu().numpy()
    elif info_gain_mname == 't5-large': row = all_logprobs[:len(label), :].detach().cpu().numpy()
    vocab_size = row.shape[-1]
    loc = F.one_hot(torch.tensor(label), num_classes=vocab_size).numpy().astype(bool)
    try: summed_logprob = np.sum(row, where = loc)
    except: import pdb; pdb.set_trace()
    filter_sums.append(summed_logprob/len(label))
    return np.array(filter_sums)

def obtain_info_gain_score(prev_steps, current_step, current_conc, target, info_model, info_tokenizer):
    if info_gain_mname == 't5-large':
        target = pad_token + target
        input = " ".join(prev_steps + [current_step]) + ' Therefore,' + target
        if len(prev_steps): ref_input =  " ".join(prev_steps) + ' Therefore,' + target
        else: ref_input = 'Therefore,' + target

        inputs = info_tokenizer(input, return_tensors="pt")
        ref = info_tokenizer(ref_input, return_tensors="pt")
        labels = info_tokenizer(target, return_tensors="pt")
        inputs["decoder_input_ids"] = labels['input_ids']
        inputs['decoder_attention_mask'] = labels['attention_mask']
        ref["decoder_input_ids"] = labels['input_ids']
        ref['decoder_attention_mask'] = labels['attention_mask']

        for i in inputs: inputs[i] = inputs[i].cuda()
        for i in ref: ref[i] = ref[i].cuda()
        with torch.no_grad():
            inp_logits = info_model.forward(**inputs).logits.detach().cpu()
            ref_logits = info_model.forward(**ref).logits.detach().cpu()
        all_inp_logprobs = torch.log(torch.softmax(inp_logits, dim=-1))
        all_ref_logprobs = torch.log(torch.softmax(ref_logits, dim=-1))
        labels = labels.input_ids.detach().cpu().tolist()[0][1:]
        filtered_inp_logprobs = slice_select_logits(all_inp_logprobs[0,:,:], labels)[0]
        filtered_ref_logprobs = slice_select_logits(all_ref_logprobs[0,:,:], labels)[0]

    elif info_gain_mname == 'gpt2':
        target = " " + target
        input = " " + " ".join(prev_steps + [current_step]) + ' Therefore,' + target
        if len(prev_steps): ref_input = " " + " ".join(prev_steps) + ' Therefore,' + target
        else: ref_input = ' Therefore,' + target
        labels = info_tokenizer(target).input_ids
        input_ids = info_tokenizer(input, return_tensors="pt").input_ids.cuda()
        ref_input_ids = info_tokenizer(ref_input, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            inp_logits = info_model.forward(input_ids=input_ids, return_dict=True).logits.detach().cpu()
            ref_logits = info_model.forward(input_ids=ref_input_ids, return_dict=True).logits.detach().cpu()
        all_inp_logprobs = torch.log(torch.softmax(inp_logits, dim=-1))
        all_ref_logprobs = torch.log(torch.softmax(ref_logits, dim=-1))
        filtered_inp_logprobs = slice_select_logits(all_inp_logprobs[0,:-1,:], labels)[0] #shift probability since at idx i produce distribution of tokens at i+1
        filtered_ref_logprobs = slice_select_logits(all_ref_logprobs[0,:-1,:], labels)[0]
    return (filtered_inp_logprobs - filtered_ref_logprobs)
    

source_path = 'perturbed_trees'
error_types = os.listdir(source_path)
score_keys = ['entail', 'pvi', 'contradict', 'll-info', 'pvi-info']
score_keys = ['ll-info']
errors_correl = {k:{'somersd':{}, 'pearson':{}} for k in score_keys}
K = 0 # Set how many past steps to look at.

for error in error_types:
    print(error)
    epath = os.path.join(source_path, error)
    tree_entry = [json.loads(line) for line in open(epath, 'r')]
    local_ent_scores = []
    alt_local_ent_scores = []
    local_pvi_scores = []
    global_contradict_scores = []
    info_ll_scores, info_pvi_scores = [], [], []
    for t, entry in tqdm_copy(enumerate(tree_entry)):
        # Tree-based Evaluation from EB
        # Otherwise, directly iterate over reasoning problems and directly get sentences
        # Hypothesis for GSM-8K is concat question and answer
        input_context = entry['question']
        input_context_sentences = sent_tokenize(input_context)
        steps, sentences = entry['steps']['perturbed'], entry['sentences']['perturbed']
        reasoning_steps = get_reasoning_chain_text(steps, sentences)
        # reasoning_steps = sent_tokenize(entry['steps'])
        # Needed keys are: question (input_context), steps, hypothesis
        step_ent_scores = []
        alt_step_ent_scores = []
        step_pvi_scores = []
        step_contradict_scores = []
        step_redundancy_scores = []
        step_ll_scores = []
        step_pviinfo_scores = []
        running_conc = []
        for sid, step in enumerate(reasoning_steps):
            units = get_phrases(step)
            premise_units, conc_units = [], []
            premise_units.extend(units[:-1])
            conc_units.append(units[-1])
        
            # Entail Step Calculation
            if 'entail' in score_keys:
                alt_step_ent_scores.append(obtain_unit_entailment_score(premise_units + running_conc[-1*K:], conc_units))
            # Intra-Step PVI Calculation
            if 'pvi' in score_keys:
                step_pvi_scores.append(obtain_unit_pvi_score(premise_units + running_conc[-1*K:], conc_units))
            # Global Contradiction Check
            if 'contradict' in score_keys:
                step_contradict_scores.append(obtain_contradiction_score(input_context_sentences + running_conc, conc_units))

            # LL Informativeness Check
            if 'll-info' in score_keys:
                step_ll_scores.append(obtain_info_gain_score(reasoning_steps[:sid], step, conc_units, sentences['hypothesis'], ll_model, ll_tokenizer))
            
            # PVI Informativeness Check
            if 'pvi-info' in score_keys:
                step_pviinfo_scores.append(obtain_info_gain_score(reasoning_steps[:sid], step, conc_units, sentences['hypothesis'], info_gain_model, info_gain_tokenizer))
            running_conc.extend(conc_units)
        
        if 'entail' in score_keys: alt_local_ent_scores.append(min(alt_step_ent_scores))
        if 'pvi' in score_keys: local_pvi_scores.append(min(step_pvi_scores))
        if 'contradict' in score_keys: global_contradict_scores.append(min(step_contradict_scores))
        if 'll-info' in score_keys: info_ll_scores.append(min(step_ll_scores))
        if 'pvi-info' in score_keys: info_pvi_scores.append(min(step_pviinfo_scores))
    perturbed_ids = [1 - int(e['perturbed']) for e in tree_entry]

    if 'entail' in score_keys: print(somersd(perturbed_ids, alt_local_ent_scores).statistic)
    if 'pvi' in score_keys: print(somersd(perturbed_ids, local_pvi_scores).statistic)
    if 'contradict' in score_keys: print(somersd(perturbed_ids, global_contradict_scores).statistic)
    if 'll-info' in score_keys: print(somersd(perturbed_ids, info_ll_scores).statistic)
    if 'pvi-info' in score_keys: print(somersd(perturbed_ids, info_pvi_scores).statistic)
    
    if 'entail' in score_keys:
        errors_correl['entail']['pearson'][error] = np.corrcoef(alt_local_ent_scores, perturbed_ids)[0][1]
        errors_correl['entail']['somersd'][error] = somersd(perturbed_ids, alt_local_ent_scores).statistic
    if 'pvi' in score_keys:
        errors_correl['pvi']['pearson'][error] = np.corrcoef(local_pvi_scores, perturbed_ids)[0][1]
        errors_correl['pvi']['somersd'][error] = somersd(perturbed_ids, local_pvi_scores).statistic
    if 'contradict' in score_keys:
        errors_correl['contradict']['pearson'][error] = np.corrcoef(global_contradict_scores, perturbed_ids)[0][1]
        errors_correl['contradict']['somersd'][error] = somersd(perturbed_ids, global_contradict_scores).statistic
    if 'pvi-info' in score_keys:
        errors_correl['redundancy']['pearson'][error] = np.corrcoef(info_pvi_scores, perturbed_ids)[0][1]
        errors_correl['redundancy']['somersd'][error] = somersd(perturbed_ids, info_pvi_scores).statistic
    if 'll-info' in score_keys:
        errors_correl['ll-info']['pearson'][error] = np.corrcoef(info_ll_scores, perturbed_ids)[0][1]
        errors_correl['ll-info']['somersd'][error] = somersd(perturbed_ids, info_ll_scores).statistic
    
f = open('ResultLogs/correlations.json', 'w+')  
json.dump(errors_correl, f, indent=4)
