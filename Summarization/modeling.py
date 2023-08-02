import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

def summarize_text(text):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

    text = text.replace('\n', ' ')

    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
    data = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return data


if __name__ == "__main__":
    print(summarize_text(input()))
    #summarized_text = summarize_text(original_text)
    #print("Original Text:", original_text)
    #print("Summarized Text:", summarized_text)