import datamodule
import h5torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics as m

# Fix for: You are using a CUDA device ('NVIDIA GeForce RTX 3070 Laptop GPU') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')

dataset = h5torch.Dataset("GNPS.h5t")

batch_size = 128

data = datamodule.GNPSDataModule("GNPS.h5t", batch_size)
data.setup(0)

# LightningModule that receives a PyTorch model as input
class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
    
        self.accuracy = m.Accuracy(task="multilabel", num_labels=model.output_size)
        self.precision = m.Precision(task="multilabel", num_labels=model.output_size)
        self.recall = m.Recall(task="multilabel", num_labels=model.output_size)
        self.f1score = m.F1Score(task="multilabel", num_labels=model.output_size)
        self.jaccard = m.JaccardIndex(task="multilabel", num_labels=model.output_size)
        self.dice = m.Dice()
        #self.cosine_sim = m.CosineSimilarity(reduction = 'mean')

    def forward(self, x):
        return self.model(x)
        
    def calculate_metrics(self, batch, prefix=''):
        X = batch['spectrum']
        Y = batch['fingerprint'].float()
        
        logits = self(X)
        
        # calculate loss (Try different loss functions)
        loss = F.binary_cross_entropy_with_logits(logits, Y)
        
        # Apply sigmoid to get probabilities
        predicted_probs = F.sigmoid(logits)
        # Convert probabilities to binary predictions (threshold = 0.5)
        predicted_labels = (predicted_probs > 0.5)
        
        accuracy = self.accuracy(predicted_labels, Y)
        precision = self.precision(predicted_labels, Y)
        recall = self.recall(predicted_labels, Y)
        f1score = self.f1score(predicted_labels, Y)
        jaccard = self.jaccard(predicted_labels, Y.int())
        dice = self.dice(logits, Y.int())
        #cosine_sim = self.cosine_sim(logits, Y)
        
        return {
            prefix+'loss': loss,
            prefix+'accuracy': accuracy,
            prefix+'precision': precision,
            prefix+'recall': recall,
            prefix+'f1score': f1score,
            prefix+'jaccardIndex': jaccard,
            prefix+'dice': dice,
            #prefix+'cosine_sim': cosine_sim
        }
        
    
    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, prefix='train_')
        self.log_dict(metrics)
        return metrics['train_loss']  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, prefix='val_')
        self.log_dict(metrics, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, prefix='test_')
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class myMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()

        self.output_size = output_size
        
        # init layers (default Linear + Relu)
        layers = [input_size] + hidden_sizes + [output_size]
        l = []
        for i in range(len(layers)-1):
            l.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                l.append(nn.ReLU())
            else:
                break
                
        self.layers = torch.nn.Sequential(*l)

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()

# Assuming DataLoader train_loader is provided, and has input_size and output_size
input_size = train_loader.dataset[0]['spectrum'].shape[0]  # Assuming first element gives the shape
output_size = train_loader.dataset[0]['fingerprint'].shape[0]  # Assuming labels are bitvectors

MLP_archs = [[2048, 8192]]#, [8192, 2048], [2048, 2048]]#[2048, 4096, 8192]
EPOCHS = 50
LRs = [.005]#[.05, .01, .005]
model_name = "MLP_loss_func_test"

for hs in MLP_archs:
    for LR in LRs:
        print(hs, LR)
        pytorch_model = myMLP(input_size, output_size, hs)
        lightning_model = LightningModel(model=pytorch_model, learning_rate=LR)
        logger = L.pytorch.loggers.TensorBoardLogger("lightning_logs", name=model_name, version=f"H{hs}LR{LR:.0e}E{EPOCHS}")

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices="auto",  # Uses all available GPUs if applicable
            logger=logger
        )

        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        