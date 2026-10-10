"""Shared checkpoint loading for offline inspection and validation tools."""
from pathlib import Path
import json
from training_runtime import load_object, load_inference_model, resume_directory, require_training_device


def checkpoint_config(checkpoint, config_path=None):
    checkpoint = Path(checkpoint).resolve()
    candidates = ([Path(config_path)] if config_path else [
        resume_directory(checkpoint) / 'config.json',
        checkpoint.parent / 'config.json', checkpoint.parent.parent / 'config.json'])
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    raise ValueError('checkpoint configuration not found; provide --config')


def load_policy(checkpoint, config, device=None):
    device = require_training_device(device or config['device'])
    model = load_inference_model(load_object(config['algorithm']), checkpoint, device)
    model.policy.set_training_mode(False)
    return model
