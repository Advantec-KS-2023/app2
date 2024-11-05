from flask import Flask, render_template, request, jsonify
import io, os, torch, gc,math
from PyPDF2 import PdfReader
from transformers import pipeline

app = Flask(__name__)

MODEL_PATH = "tsmatz/mt5_summarize_japanese"


seq2seq = pipeline("summarization", model = MODEL_PATH, max_length = 5000, min_length = 100)


def extract_the_text(file_path):
    reader= PdfReader(file_path)
    text=" "
    for page in reader.pages:
        cs = page.extract_text()
        cs2 = cs.replace("\n", ' ' )
        text +=cs2
    x = math.ceil(len(text)/5)
    y = math.ceil(len(text)/10)
    return text.strip(), x, y


@app.route('/', methods=['GET', 'POST'] )
def index():
    if request.method == 'POST':
        file = request.files['pdf']
        pdf_path = os.path.join('uploads', file.filename)
        file.save(pdf_path)

        text, x, y = extract_the_text(pdf_path)
        print(len(text))
        print(x)
        print(y)
        summary = seq2seq(text, max_length=x, min_length=y)[0]
        summary_text = summary.get('summary_text')
        os.remove(pdf_path)
        return render_template('summary.html', summaries = summary_text)
    gc.collect()
    return render_template('index.html')


if __name__ =='__main__':
    app.run(debug = True)
    
