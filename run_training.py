import torch
import os
import sys
from src import config
from src.data.dataset import get_dataloaders
from src.training.models import setup_model_and_optimizer
from src.training.trainer import train_epoch, validate_epoch

def main():
    train_dataloader, val_dataloader = get_dataloaders()
    pipe, controlnet, optimizer, scheduler = setup_model_and_optimizer()
    
    patience_counter = 0
    best_eval_loss = config.best_eval_loss 
    
    for epoch in range(config.num_epochs):
        avg_train_loss = train_epoch(controlnet, pipe, train_dataloader, optimizer)
        print(f"\nEpoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = validate_epoch(controlnet, pipe, val_dataloader)
        print(f"Epoch {epoch}, Avg Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_eval_loss:
            best_eval_loss = avg_val_loss
            patience_counter = 0
            controlnet.save_pretrained(config.train_best_model_path)
            print(f"Saved best model at: {config.train_best_model_path}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter} / {config.patience}")
            if patience_counter >= config.patience:
                print(f"Early stopping after {config.patience} epochs.")
                break 
        
        scheduler.step()

    controlnet.save_pretrained(config.train_latest_model_path)
    print(f"Saved best model (Eval Loss: {best_eval_loss:.4f}) at: {config.train_best_model_path}")
    print(f"Saved final model at: {config.train_latest_model_path}")

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    main()