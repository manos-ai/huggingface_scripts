# -*- coding: utf-8 -*-
"""
calculate the probability of a completion for a huggingface LLM
"""


#%% imports

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


#%% load llm and tokenizer

# Load the tokenizer and model from Hugging Face
model_name = "gpt2"  # You can choose other models like "gpt2-medium", "gpt-neo", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = "../HF")
model.eval()  # Set the model to evaluation mode

# Ensure that the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
#%% set up prompt and completion

# Define the prompt and the completion with EOS token
prompt = "The quick brown fox jumps"
completion = " over the lazy dog" + "{tokenizer.eos_token}"

# Encode the prompt and completion
input_ids = tokenizer.encode(prompt, return_tensors='pt')
completion_ids = tokenizer.encode(completion, return_tensors='pt')

# Concatenate prompt and completion
input_ids_full = torch.cat([input_ids, completion_ids], dim=1)


#%% get model log-probs for the entire seq

with torch.no_grad():
    # Get the logits for the entire sequence
    outputs = model(input_ids_full)
    logits = outputs.logits
# end with    

#%% manipulate probs, get what we need

# Shift logits and labels for next-token prediction
# We want to calculate P(completion | prompt), including the EOS token
prompt_length = input_ids.shape[1]
completion_length = completion_ids.shape[1]

# Extract logits for the completion tokens
# For each token in the completion, its probability is based on the tokens up to that point (including prompt and previous completion tokens)
# We include the EOS token in the calculation
completion_logits = logits[0, prompt_length-1:prompt_length + completion_length - 1, :]

# The target tokens are the completion tokens except the first one
target_ids = completion_ids[0, :]

# Calculate log probabilities
log_probs = torch.log_softmax(completion_logits, dim=-1)

# Gather the log probabilities of the target tokens
selected_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

# Sum the log probabilities to get the total log-probability of the completion
total_log_prob = selected_log_probs.sum().item()

# Calculate the probability by exponentiating the log-probability
total_prob = torch.exp(selected_log_probs).prod().item()

# Alternatively, calculate perplexity
perplexity = torch.exp(-selected_log_probs.mean()).item()


#%% Display the results

print(f"Prompt: {prompt}")
print(f"Completion: {completion.replace(tokenizer.eos_token, '<EOS>')}")
print("Detailed Token Probabilities:")
for i, log_prob in enumerate(selected_log_probs):
    token = tokenizer.decode(completion_ids[0, i])
    prob = torch.exp(log_prob).item()
    print(f"  Token {i + 1}: '{token}' - Log-Prob: {log_prob.item():.4f}, Prob: {prob:.6f}")
print(f"Total Log-Probability: {total_log_prob:.4f}")
print(f"Total Probability: {total_prob:.6e}")
print(f"Perplexity: {perplexity:.4f}")


#%% define a function for it

def get_completion_prob(model, tokenizer, prompt, completion, assume_eos = True):
    '''
    given a HuggingFace LLM, a prompt and a completion, calculates the 
    probability P(completion|prompt), and related quantities such as perplexity

    Parameters
    ----------
    model : transformers.model
        a huggingface LLM
    tokenizer : transformers.tokenizer
        a huggingface tokenizer
    prompt : string
        the prompt
    completion : string
        the required completion
    assume_eos : bool
        if True, we assume that the model must output EOS afterwards and append it to the 
        completion
        EOS: End of Sentance token

    Returns
    -------
    total_prob: float
        the prob of the completion
    total_log_prob: float
        the log prob of the completion
    perplexity: float
        the perplexity of the completion
    '''
    
    # attach an EOS token to the completion
    if assume_eos:
        completion += "{tokenizer.eos_token}"
    
    # Encode the prompt and completion
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    completion_ids = tokenizer.encode(completion, return_tensors='pt')

    # Concatenate prompt and completion
    input_ids_full = torch.cat([input_ids, completion_ids], dim=1)
    
    with torch.no_grad():
        # Get the logits for the entire sequence
        outputs = model(input_ids_full)
        logits = outputs.logits
    # end with  
    
    # Shift logits and labels for next-token prediction
    # We want to calculate P(completion | prompt), including the EOS token
    prompt_length = input_ids.shape[1]
    completion_length = completion_ids.shape[1]

    # Extract logits for the completion tokens
    # For each token in the completion, its probability is based on the tokens up to that point (including prompt and previous completion tokens)
    # We include the EOS token in the calculation
    completion_logits = logits[0, prompt_length-1:prompt_length + completion_length - 1, :]

    # The target tokens are the completion tokens except the first one
    target_ids = completion_ids[0, :]

    # Calculate log probabilities
    log_probs = torch.log_softmax(completion_logits, dim=-1)

    # Gather the log probabilities of the target tokens
    selected_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Sum the log probabilities to get the total log-probability of the completion
    total_log_prob = selected_log_probs.sum().item()

    # Calculate the probability by exponentiating the log-probability
    total_prob = torch.exp(selected_log_probs).prod().item()

    # Alternatively, calculate perplexity
    perplexity = torch.exp(-selected_log_probs.mean()).item()
    
    # ready
    return total_prob, total_log_prob, perplexity
# end func


#%% test function

prompt = '2 + 3 ='
completion = ' 5'

total_prob, total_log_prob, perplexity = get_completion_prob(model, tokenizer, prompt, completion, False)

print(f"Prompt: {prompt}")
print(f"Completion: {completion.replace(tokenizer.eos_token, '<EOS>')}")
print(f"Total Log-Probability: {total_log_prob:.4f}")
print(f"Total Probability: {total_prob:.6e}")
print(f"Perplexity: {perplexity:.4f}")
    


