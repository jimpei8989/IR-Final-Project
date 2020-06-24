import os
from argparse import ArgumentParser

from Modules.vsm.model import Model

def main():
    args = parseeArguments()

    # Create Model
    if args.train:
        print(f'- Train model')
        model = Model(maxFeatures=10 ** 5, minDF=1000, maxDF=0.5)
        model.build(os.path.join(args.dataDir, 'corpus'), args.docIDFile)
        model.save(args.modelDir)
    else:
        print(f'- Load model from {args.modelDir}')
        model = Model()
        model.load(args.modelDir)

    print(model.model.get_feature_names())


def parseeArguments():
    parser = ArgumentParser()
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--modelDir', default='models/vsm/')
    parser.add_argument('--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('--train', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
