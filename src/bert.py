import os
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from Modules import utils
from Modules.utils import EventTimer
from Modules.bert.dataset import RetrievalDataset
from Modules.bert.model import BertModel

def main():
    args = parseArguments()

    device=torch.device(f'cuda:{args.gpuID}')

    with EventTimer('Create Dataset'):
        corpusDir = os.path.join(args.dataDir, 'corpus')
        trainDataset = RetrievalDataset(corpusDir, args.docIDFile, args.trainDir)
        trainDataloader = DataLoader(trainDataset, batch_size=args.batchSize)

        validDataset = RetrievalDataset(corpusDir, args.docIDFile, args.validDir)
        validDataloader = DataLoader(validDataset, batch_size=args.batchSize)

    with EventTimer('Build model'):
        bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

        model = BertModel(bert, args.clfHiddenDim).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = CrossEntropyLoss()

    with EventTimer('Train Model'):
        def runEpoch(dataloader, train=True):
            losses, accuracies = [], []
            with (torch.enable_grad() if train else torch.no_grad()):
                for doc, query, rel in tqdm(dataloader):
                    docTokens = pad_sequence([torch.LongTensor(tokenizer.encode(d)[:512]) for d in doc], batch_first=True).to(device)
                    queryTokens = pad_sequence([torch.LongTensor(tokenizer.encode(q)[:512]) for q in query], batch_first=True).to(device)

                    print(docTokens.shape)

                    output = model(docTokens, queryTokens).cpu()
                    loss = criterion(output, rel)

                    if train:
                        loss.backward()
                        optimizer.step()

                    losses.append(loss)
                    accuracies.append(np.mean(output.round() == rel))

            return np.mean(losses), np.mean(accuracies)

        for epoch in range(1, args.epochs + 1):
            trainLoss, trainAccu = runEpoch(trainDataloader)
            validLoss, validAccu = runEpoch(validDataloader)

            print(f'> Epoch: {epoch:2d} | Train: {validLoss:.5f} / {validAccu:.3f} | Validation: {validLoss:.5f} / {validAccu:.3f}')

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('--trainDir', default='data/partial/train')
    parser.add_argument('--validDir', default='data/partial/dev')
    parser.add_argument('--modelDir', default='models/bert/')
    parser.add_argument('--clfHiddenDim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--epochStepSize', type=int, default=500000)
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--gpuID', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main()
