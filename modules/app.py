import sys
from typing import List
import traceback
import os
import base64

import logging
logging.basicConfig(level=logging.INFO)
import modules.cloud_logging

import tokenizers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pprint

# needs to be imported *before* transformers
if os.path.exists('debug'):
    BIG_MODEL = False
    CUDA = False
    logging.info('debug BIG_MODEL false')
else:
    BIG_MODEL = True
    CUDA = True
    logging.info('debug BIG_MODEL true')

# from flask import Flask, request, render_template
# from flask_cors import CORS
# app = Flask(__name__, static_folder='static')
# app.config['TEMPLATES_AUTO_RELOAD'] = Tru
# CORS(app, resources= {
#     r"/generate": {"origins": origins},
#     r"/infill": {"origins": origins},
# })
# origins=[f"http://localhost:{PORT}", "https://huggingface.co", "https://hf.space"]

PORT = 7860
VERBOSE = False

if os.path.exists('unlock'):
    MAX_LENGTH = 2048
else:
    MAX_LENGTH = 256+64
TRUNCATION_MESSAGE = f'warning: This demo is limited to {MAX_LENGTH} tokens in the document for efficiency.'

if BIG_MODEL:
    model_name = "facebook/incoder-6B"
    kwargs = dict(
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
else:
    model_name = "facebook/incoder-1B"
    kwargs = dict()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")


logging.info("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
logging.info("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
logging.info("loading complete")

if CUDA:
    model = model.half().cuda()

BOS = "<|endoftext|>"
EOM = "<|endofmask|>"

def make_sentinel(i):
    return f"<|mask:{i}|>"

SPECIAL_TOKENS = [make_sentinel(i) for i in range(256)] + [EOM]

def generate(input, length_limit=None, temperature=None):
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if CUDA:
        input_ids = input_ids.cuda()
    current_length = input_ids.flatten().size(0)
    max_length = length_limit + current_length
    truncated = False
    if max_length > MAX_LENGTH:
        max_length = MAX_LENGTH
        truncated = True
    if max_length == current_length:
        return input, True
    output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
    detok_hypo_str = tokenizer.decode(output.flatten())
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str, truncated

def infill(parts: List[str], length_limit=None, temperature=None, extra_sentinel=False, max_retries=1):
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False


    while (not done) and (retries_attempted < max_retries):
        any_truncated = False
        retries_attempted += 1
        if VERBOSE:
            logging.info(f"retry {retries_attempted}")
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
            
            # prompt += TokenizerWrapper.make_sentinel(0)
        
        infills = []
        complete = []

        done = True

        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            completion, this_truncated = generate(prompt, length_limit, temperature)
            any_truncated |= this_truncated
            completion = completion[len(prompt):]
            if EOM not in completion:
                if VERBOSE:
                    logging.info(f"warning: {EOM} not found")
                completion += EOM
                # TODO: break inner loop here
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        logging.info("generated text:")
        logging.info(prompt)
        logging.info()
        logging.info("parts:")
        logging.info(parts)
        logging.info()
        logging.info("infills:")
        logging.info(infills)
        logging.info()
        logging.info("restitched text:")
        logging.info(text)
        logging.info()
    
    return {
        'text': text,
        'parts': parts,
        'infills': infills,
        'retries_attempted': retries_attempted,
        'truncated': any_truncated,
    } 


@app.head("/")
@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="static/playground.html", media_type="text/html")

@app.head("/idx")
@app.get("/idx")
def index() -> FileResponse:
    return FileResponse(path="static/index.html", media_type="text/html")

@app.get('/generate')
# async def generate_maybe(request: Request):
async def generate_maybe(info: str):
    # form = await info.json()
    # form = await request.json()
    # info is a base64-encoded, url-escaped json string (since GET doesn't support a body, and POST leads to CORS issues)
    # fix padding, following https://stackoverflow.com/a/9956217/1319683
    info = base64.urlsafe_b64decode(info + '=' * (4 - len(info) % 4)).decode('utf-8')
    form = json.loads(info)
    # print(form)
    prompt = form['prompt']
    length_limit = int(form['length'])
    temperature = float(form['temperature'])
    logging.info(json.dumps({
        'length': length_limit,
        'temperature': temperature,
        'prompt': prompt,
    }))
    try:
        generation, truncated = generate(prompt, length_limit, temperature)
        if truncated:
            message = TRUNCATION_MESSAGE 
        else:
            message = ''
        return {'result': 'success', 'type': 'generate', 'prompt': prompt, 'text': generation, 'message': message}
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        logging.error(e)
        return {'result': 'error', 'type': 'generate', 'prompt': prompt, 'message': f'Error: {e}.'}

@app.get('/infill')
# async def infill_maybe(request: Request):
async def infill_maybe(info: str):
    # form = await info.json()
    # form = await request.json()
    # info is a base64-encoded, url-escaped json string (since GET doesn't support a body, and POST leads to CORS issues)
    # fix padding, following https://stackoverflow.com/a/9956217/1319683
    info = base64.urlsafe_b64decode(info + '=' * (4 - len(info) % 4)).decode('utf-8')
    form = json.loads(info)
    length_limit = int(form['length'])
    temperature = float(form['temperature'])
    max_retries = 1
    extra_sentinel = True
    logging.info(json.dumps({
        'length': length_limit,
        'temperature': temperature,
        'parts_joined': '<infill>'.join(form['parts']),
    }))
    try:
        if len(form['parts']) > 4:
            return {'result': 'error', 'text': ''.join(form['parts']), 'type': 'infill', 'message': f"error: Can't use more than 3 <infill> tokens in this demo (for efficiency)."}
        generation = infill(form['parts'], length_limit, temperature, extra_sentinel=extra_sentinel, max_retries=max_retries)
        generation['result'] = 'success'
        generation['type'] = 'infill'
        if generation['truncated']:
            generation['message'] = TRUNCATION_MESSAGE
        else:
            generation['message'] = ''
        return generation
        # return {'result': 'success', 'prefix': prefix, 'suffix': suffix,  'text': generation['text']}
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        logging.error(e)
        return {'result': 'error', 'type': 'infill', 'message': f'Error: {e}.'}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, threaded=False)
