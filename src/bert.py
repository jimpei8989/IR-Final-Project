import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

from Modules import utils
from Modules.utils import EventTimer
from Modules.bert.dataset import RetrievalDataset, DocRetrievalDataset, QueryRetrievalDataset
from Modules.bert.model import BertModel

def scheduler(model, epoch):
    model.freezeBert()
    model.unfreezeBert(7 - epoch)

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

        docDataset = DocRetrievalDataset(corpusDir, args.docIDFile, args.testDir)
        docDataloader = DataLoader(docDataset, batch_size=args.batchSize, num_workers=1)

        queryDataset = QueryRetrievalDataset(corpusDir, args.docIDFile, args.testDir)
        queryDataloader = DataLoader(queryDataset, batch_size=args.batchSize, num_workers=1)

    with EventTimer('Build model'):
        bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

        model = BertModel(bert, args.clfHiddenDim).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=4, verbose=True, min_lr=1e-6)
        criterion = BCELoss()

    with EventTimer('Train Model'):
        def runEpoch(epoch, dataloader, valid_dataloader=None, validationInterval=None, train=True):
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
                    print('\n', end='\r')
                    val_loss, val_acc = runEpoch(epoch, valid_dataloader, train=False)
                    print(f'> Epoch: {epoch:2d} | loss: {losses[-1]:.05f}, acc: {accuracies[-1]:.03f} | val_loss: {val_loss:.05f}, val_acc: {val_acc:.03f}', end='\n')

            return np.mean(losses), np.mean(accuracies)

        def runPredict(docloader, queryloader):
            model.eval()
            docEmb = []
            queryEmb = []
            allMAP = []
            with torch.no_grad():
                for i, doc in enumerate(tqdm(docloader)):
                    docTokens = pad_sequence([torch.LongTensor(tokenizer.encode(d)[:512]) for d in doc], batch_first=True).to(device)
                    docEmb.append(model(docTokens, None, 1).cpu())
                docEmb = torch.cat(docEmb, dim=0)
                for i, query in enumerate(tqdm(queryloader)):
                    queryTokens = pad_sequence([torch.LongTensor(tokenizer.encode(d)[:512]) for q in query], batch_first=True).to(device)
                    queryEmb.append(model(queryTokens, None, 1).cpu())
                queryEmb = torch.cat(queryEmb, dim=0)
                
                with open(args.docIDFile) as F:
                    docIDs = [line.strip() for line in F.readlines()]
                for i in tqdm(range(1000)):#tqdm(range(len(queryEmb))):
                    score = []
                    while True:
                        idx = 0
                        output = model(torch.stack([queryEmb[i]]*args.batchSize).to(device), docEmb[idx:idx+args.batchSize if idx+args.batchSize <= 1000000 else None].to(device), 2)
                        score.append(output)
                        idx += args.batchSize
                        if idx >= 1000000:    break
                    score = torch.cat(score, dim=0)
                    pred = torch.topk(score.squeeze(),args.topK)[1]
                    predDoc = [docIDs[doc.item()] for doc in pred]
                    with open(os.path.join(args.testDir, 'topK.csv')) as F:    truth = [set(line.strip().split(',')[1].split()) for line in F.readlines()]
                    allMAP.append(utils.MAP(truth, predDoc))
                    print(f'Query {(i+1):3d} -- MAP: {allMAP[-1]:.4f}')

            return np.mean(allMAP)


        if args.predict:
            checkpoint = torch.load(args.predict)
            model.load_state_dict(checkpoint['model'])
            validAccu = checkpoint['val_acc']
            aveMAP = runPredict(docDataloader, queryDataloader)
            print(f'Valid Accuracy: {validAccu:.03f} | Average MAP: {aveMAP:.05f}')
        else:
            for epoch in range(1, args.epochs + 1):
                scheduler(model, epoch)
                trainDataloader = DataLoader(trainDataset, batch_size=8+4*epoch, num_workers=1)
                validDataloader = DataLoader(validDataset, batch_size=args.batchSize, num_workers=1)
                trainLoss, trainAccu = runEpoch(epoch, trainDataloader, validDataloader, args.validationInterval)
                validLoss, validAccu = runEpoch(epoch, validDataloader, train=False)
                lr_scheduler.step(validLoss)

                checkpoint = {}
                checkpoint['model'] = model.state_dict()
                checkpoint['optim'] = optimizer.state_dict()
                checkpoint['val_acc'] = validAccu
                torch.save(checkpoint, os.path.join(args.modelDir, f'{epoch}'))

                print(f'> Epoch: {epoch:2d} | loss: {trainLoss:.05f}, acc: {trainAccu:.03f} | val_loss: {validLoss:.05f}, val_acc: {validAccu:.03f}')
    

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataDir', default='data/')
    parser.add_argument('-di', '--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('-t', '--trainDir', default='data/partial/train')
    parser.add_argument('-v', '--validDir', default='data/partial/dev')
    parser.add_argument('-te', '--testDir', default='data/partial/test')
    parser.add_argument('-m', '--modelDir', default='models/bert/')
    parser.add_argument('-hd', '--clfHiddenDim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batchSize', type=int, default=32)
    parser.add_argument('-vi', '--validationInterval', type=int, default=None)
    parser.add_argument('--topK', type=int, default=1000)
    # parser.add_argument('-n', '--numWorkers', type=int, default=8)
    parser.add_argument('-g', '--gpuID', default='0')
    parser.add_argument('-p', '--predict', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    main()
