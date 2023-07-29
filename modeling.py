from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration,BartTokenizer
import torch
def summarize_text(text):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")

    # 토크나이징
    input_ids = tokenizer.encode(text)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([input_ids])

    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_length=128,
        min_length=32,
        num_beams=4,
    )
    data = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    return data


if __name__ == "__main__":
    print(summarize_text(input()))
    #summarized_text = summarize_text(original_text)
    #print("Original Text:", original_text)
    #print("Summarized Text:", summarized_text)
