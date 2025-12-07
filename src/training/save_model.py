import os
import json

def save_model_and_history(model, history, model_filename="model.keras", history_filename="history.keras"):

    model_dir = os.path.dirname(model_filename)
    history_dir = os.path.dirname(history_filename)
    
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
        
    if history_dir and not os.path.exists(history_dir):
        os.makedirs(history_dir)
        print(f"Created directory: {history_dir}")
    
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    with open(history_filename, 'w') as history_file:
        json.dump(history, history_file)
    print(f"Training history saved to {history_filename}")
