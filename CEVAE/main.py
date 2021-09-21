import argparse
from CEVAE.data_loader import IHDPDataset, IHDPDataLoader
from CEVAE.inference import Inference
import torch
from pyro.optim import Adam
import numpy as np
import os

def main():
    # Command line
    parser = argparse.ArgumentParser(description="CEVAE-Pyro")
    parser.add_argument('cevae', type=str, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--z-dim', type=int, default=20)
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=200)
    args = parser.parse_args()

    # Data
    path = os.path.dirname(os.path.abspath(__file__))
    data_loaders = []
    for i in range(10):
        data = np.loadtxt(
            f"{path}/data/IHDP/ihdp_npci_{i+1}.csv", delimiter=',')
        dataset = IHDPDataset(data)
        binary_indices, continuous_indices = dataset.indices_each_features()
        data_loader = IHDPDataLoader(dataset, validation_split=0.2)
        train_loader, test_loader = data_loader.loaders(
            batch_size=args.batch_size)
        data_loaders.append((train_loader, test_loader))

    # CEVAE
    cuda = torch.cuda.is_available()
    print(f"CUDA: {cuda}")
    optimizer = Adam({
        "lr": args.learning_rate, "weight_decay": args.weight_decay
    })
    activation = torch.nn.functional.elu
    inference = Inference(len(binary_indices), len(continuous_indices), args.z_dim,
                          args.hidden_dim, args.hidden_layers, optimizer, activation, cuda)

    # Training
    for i, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"## replication {i+1}/10 ##")
        train_elbo = []
        test_elbo = []
        for epoch in range(args.epochs):
            total_epoch_loss_train = inference.train(train_loader)
            train_elbo.append(-total_epoch_loss_train)
            # print(
            #     f"[epoch {epoch:03d}] average training loss: {total_epoch_loss_train:.4f}")
            if epoch % 5 == 0:
                total_epoch_loss_test = inference.evaluate(test_loader)
                test_elbo.append(-total_epoch_loss_test)
                # print(
                #     f"[epoch {epoch:03d}] average test loss: {total_epoch_loss_test:.4f}")

                (ITE, ATE, PEHE), (RMSE_factual,
                                   RMSE_counterfactual) = inference.train_statistics(L=1, y_error=True)
                print(f"[epoch {epoch:03d}] #TRAIN# ITE: {ITE:0.3f}, ATE: {ATE:0.3f}, PEHE: {PEHE:0.3f}, Factual RMSE: {RMSE_factual:0.3f}, Counterfactual RMSE: {RMSE_counterfactual:0.3f}")

                ITE_test, ATE_test, PEHE_test = inference.test_statistics(L=1)
                print(
                    f"[epoch {epoch:03d}] #TEST# ITE: {ITE_test:0.3f}, ATE: {ATE_test:0.3f}, PEHE: {PEHE_test:0.3f}")

        score = inference.train_statistics(L=100)
        print(
            f"[replication {i+1}/10] #TRAIN# ITE: {score[0]:0.3f}, ATE: {score[1]:0.3f}, PEHE: {score[2]:0.3f}")
        score_test = inference.test_statistics(L=100)
        print(
            f"[replication {i+1}/10] #TEST# ITE: {score_test[0]:0.3f}, ATE: {score_test[1]:0.3f}, PEHE: {score_test[2]:0.3f}")

        inference.initialize_statistics()
