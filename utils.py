import re

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence_with_values(model, sentence, english, flutter, device, max_length=50):
    dot_values = []
    for dot_val in re.findall("\.[a-z]+", sentence):
        dot_values.append(dot_val)
        sentence = sentence.replace(dot_val, ".value")
    str_values = []

    pattern = r'"([A-Za-z0-9 ]*)"'
    for str_val in re.findall(pattern, sentence):
        str_values.append(str_val)
        sentence = sentence.replace(str_val, "value")

    code = ''.join(translate_sentence(model, sentence, english, flutter, device, max_length=max_length)).replace('<eos>',
                                                                                                         '').replace(
        'utf-8', '')
    print(code)
    for idx, dot_val in enumerate(re.findall(".value", code)):
        if len(dot_values) > idx:
            code = code.replace(".value", dot_values[idx])

    for idx, str_val in enumerate(re.findall("value", code)):
        if len(str_values) > idx:
            code = code.replace("value", f'"{str_values[idx]}"')


def translate_sentence(model, sentence, english, flutter, device, max_length=50):
    # Load English tokenizer
    spacy_en = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_en(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    # Go through each english token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [flutter.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == flutter.vocab.stoi["<eos>"]:
            break

    translated_sentence = [flutter.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, english, flutter, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["Input"]
        trg = vars(example)["Output"]

        prediction = translate_sentence(model, src, english, flutter, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
