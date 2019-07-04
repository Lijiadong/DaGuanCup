
from utils.config import *
from model.BLSTM_CRF import *
from utils.process_data import *
import logging
from tqdm import tqdm

train, dev, vocab_size, labels_dict = prepare_data(args['batch'])
test = prepare_test_data()

model = BLSTM_CRF(vocab_size, args['input_size'], args['hidden_size'], len(labels_dict), args['n_layers'],
                  args['lr'], args['dropout'])

for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        data, data_lengths, labels, label_lengths = data
        model.train_batch(data, data_lengths, labels, label_lengths, args['clip'], i == 0)
        pbar.set_description(model.print_loss())

    if (epoch + 1) % int(args['evalp']) == 0:
        model.eval()
        dev_score = model.evaluate(dev)
        if dev_score > model.max_score:
            model.inference(test, labels_dict, './save/', dev_score)
            model.max_score = dev_score
        model.train()
