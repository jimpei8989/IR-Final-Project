import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from Modules import utils
from Modules.utils import EventTimer
from Modules.bert.dataset import RetrievalDataset
from Modules.bert.model import BertModel

def main():
    args = parseArguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
    device=torch.device('cuda')

    with EventTimer('Create Dataset'):
        corpusDir = os.path.join(args.dataDir, 'corpus')
        trainDataset = RetrievalDataset(corpusDir, args.docIDFile, args.trainDir)
        trainDataloader = DataLoader(trainDataset, batch_size=args.batchSize, num_workers=1)

        validDataset = RetrievalDataset(corpusDir, args.docIDFile, args.validDir)
        validDataloader = DataLoader(validDataset, batch_size=args.batchSize, num_workers=1)

    with EventTimer('Build model'):
        bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

        model = BertModel(bert, args.clfHiddenDim).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        criterion = BCELoss()

    with EventTimer('Train Model'):
        def runEpoch(dataloader, valid_dataloader=None, validationInterval=None, train=True):
            losses, accuracies = [], []
            for i, (doc, query, rel) in enumerate(dataloader):
                with (torch.enable_grad() if train else torch.no_grad()):
                    docTokens = pad_sequence([torch.LongTensor(tokenizer.encode(d)[:512]) for d in doc], batch_first=True).to(device)
                    queryTokens = pad_sequence([torch.LongTensor(tokenizer.encode(q)[:512]) for q in query], batch_first=True).to(device)

                    output = model(docTokens, queryTokens).cpu()
                    rel = rel.unsqueeze(1)
                    loss = criterion(output, rel.float())

                    if train:
                        loss.backward()
                        optimizer.step()

                    losses.append(abs(loss.item()))
                    accuracies.append(np.mean((output.round().int() == rel.int()).numpy()))
                print(f'> [{i} / {len(dataloader)}] Epoch: {epoch:2d} | loss: {losses[-1]:.05f}, acc: {accuracies[-1]:.03f}', end='\r')
                
                
                if validationInterval is not None and (i + 1) % validationInterval == 0:
                    val_loss, val_acc = runEpoch(valid_dataloader, train=False)
                    print(f'> Epoch: {epoch:2d} | loss: {losses[-1]:.05f}, acc: {accuracies[-1]:.03f} | val_loss: {val_loss:.05f}, val_acc: {val_acc:.03f}', end='\r')

            return np.mean(losses), np.mean(accuracies)

        for epoch in range(1, args.epochs + 1):
            trainLoss, trainAccu = runEpoch(trainDataloader, validDataloader, args.validationInterval)
            validLoss, validAccu = runEpoch(validDataloader, train=False)

            print(f'> Epoch: {epoch:2d} | loss: {trainLoss:.05f}, acc: {trainAccu:.03f} | val_loss: {validLoss:.05f}, val_acc: {validAccu:.03f}')

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataDir', default='data/')
    parser.add_argument('-di', '--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('-t', '--trainDir', default='data/partial/train')
    parser.add_argument('-v', '--validDir', default='data/partial/dev')
    parser.add_argument('-m', '--modelDir', default='models/bert/')
    parser.add_argument('-hd', '--clfHiddenDim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batchSize', type=int, default=32)
    parser.add_argument('-vi', '--validationInterval', type=int, default=None)
    parser.add_argument('--topK', type=int, default=1000)
    # parser.add_argument('-n', '--numWorkers', type=int, default=8)
    parser.add_argument('-g', '--gpuID', default='0')
    return parser.parse_args()

if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    main()
