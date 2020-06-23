import os
from argparse import ArgumentParser

from Modules.vsm.model import Model

def main():
    args = parseeArguments()

    model = Model(maxFeatures=10 ** 6, maxDF=0.5)
    model.build(os.path.join(args.dataDir, 'corpus'), args.docIDFile)
    model.save(args.modelDir)


def parseeArguments():
    parser = ArgumentParser()
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--modelDir', default='models/vsm/')
    parser.add_argument('--docIDFile', default='data/partial/corpus/docIDs')
    return parser.parse_args()

if __name__ == '__main__':
    main()
