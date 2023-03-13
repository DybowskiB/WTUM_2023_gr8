from model import LSTMModel
import pytorch_lightning as pl


def main():
    print("Initializing model...")
    model = LSTMModel()
    print("Training model...")
    trainer = pl.Trainer(max_epochs=20, accelerator="cpu", log_every_n_steps=2)
    trainer.fit(model)
        
    
if __name__ == "__main__":
   main()