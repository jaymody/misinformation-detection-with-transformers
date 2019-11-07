#### Setup ####
# Python Standard Library
import logging
import collections

# Third Party Packages
import numpy as np
from simpletransformers.model import TransformerModel

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]


#### Load Model ####
def basic_config(output_dir, cache_dir, fp16=False, max_seq_length=256, train_batch_size=8, nproc=1, ngpu=0):
    return {
        "output_dir": output_dir,
        "cache_dir": cache_dir,

        "fp16": fp16,
        "fp16_opt_level": "O1",
        "max_seq_length": max_seq_length,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 8,
        "num_train_epochs": 1,
        "weight_decay": 0,
        "learning_rate": 2e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,

        "logging_steps": 100,
        "save_steps": 2000,

        "overwrite_output_dir": False,
        "reprocess_input_data": False,

        "process_count": nproc,
        "n_gpu": ngpu,
    }

def load_model(model_dir, model_args):
    logger.info("... loading model ...")
    return TransformerModel('xlnet', model_dir, num_labels=3, args=model_args)
  
def predict_proba(examples_text, claim_ids, model):
    logger.info("... predicting ...")
    model_outputs = model.predict(examples_text)
    
    probs = collections.defaultdict(list)
    for claim_id, prob in zip(claim_ids, model_outputs[:][1]):
        probs[claim_id].append(prob)
        
    averaged_probs = {k: np.mean(v, axis=0) for k,v in probs.items()}
    return averaged_probs

def predict(examples_text, claim_ids, model):
    probs = predict_proba(examples_text, claim_ids, model)
    return {k: np.argmax(v) for k,v in probs.items()}
