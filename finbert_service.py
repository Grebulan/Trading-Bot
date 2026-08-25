import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
torch.set_num_threads(1)

from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

_finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert',
                     tokenizer='ProsusAI/finbert', truncation=True, max_length=512)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/score', methods=['POST'])
def score():
    data = request.get_json(force=True) or {}
    headlines = data.get('headlines', [])
    if not headlines:
        return jsonify({'scores': []})
    results = _finbert(headlines, truncation=True, max_length=512, batch_size=8)
    return jsonify({'scores': results})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051, threaded=False)
