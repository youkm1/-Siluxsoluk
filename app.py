from flask import Flask, redirect, url_for, request,render_template,session
import torch
import transformers #import BartTokenizer, BartForMaskedLM


app=Flask(__name__)

#model
model_name = 'digit82/kobart-summarization'
#"bert-base-multilingual-cased"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
#tokenizer=BertTokenizer.from_pretrained(model_name)
#model = BertForMaskedLM.from_pretrained(model_name)

#요약 기능 함수
def summary_txt(text):
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.route("/", methods=['GET','POST'])
def index():
    summary = None
    if request.method == 'POST':
        text = request.form["text"]
        if text:
            summary = summary_txt(text)
    return render_template("index.html", summary=summary)

if __name__=='__main__':
    app.run(debug=True)
