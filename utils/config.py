import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='DaGuanCup')
parser.add_argument('-input', '--input_size', help='Input size', required=False, default=300)
parser.add_argument('-hdd','--hidden_size', help='Hidden size', required=False, default=300)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, default=50)
parser.add_argument('-lr','--lr', help='Learning Rate', required=False, default=1e-3)
parser.add_argument('-dr','--dropout', help='Drop Out', required=False, default=0.25)
parser.add_argument('-layer','--n_layers', help='Layer Number', required=False, default=1)
parser.add_argument('-split', '--data_split', help='Data split', required=False, default=0.8)
parser.add_argument('-clip', '--clip', help='Clip gradient', required=False, default=10.0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
args = vars(parser.parse_args())