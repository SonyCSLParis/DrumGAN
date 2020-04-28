from torch import nn
from data.preprocessing import AudioProcessor
from data.loaders import get_data_loader
from tqdm import tqdm, trange
from utils.utils import mkdir_in_path, GPU_is_available
from torch.utils.data import DataLoader

import torch
import os

from os.path import dirname, realpath, join

from datetime import datetime
from gans.ac_criterion import ACGANCriterion
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from utils.utils import read_json
import torch.nn.functional as F
from datetime import datetime
import ipdb
from .inception_network import SpectrogramInception3
import numpy as np
import logging


def train_inception_model(name: str, path: str, labels: list, config: str, batch_size: int=50, n_epoch=100):

    output_path = mkdir_in_path(path, 'inception_models')
    output_file = join(output_path, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.pt")
    output_log = join(output_path, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
    logging.basicConfig(filename=output_log, level=logging.INFO)

    assert os.path.exists(config), f"Path to config {config} does not exist"
    config = read_json(config)

    loader_config = config['loader_config']
    transform_config = config['transform_config']
    transform = transform_config['transform']
    dbname = loader_config.pop('dbname')
    loader_module = get_data_loader(dbname)

    processor = AudioProcessor(**transform_config)
    loader = loader_module(dbname=dbname + '_' + transform, preprocessing=processor, **loader_config)

    val_data, val_labels = loader.get_validation_set()
    val_data = val_data[:, 0:1]

    att_dict = loader.header['attributes']
    att_classes = att_dict.keys()
    num_classes = sum(len(att_dict[k]['values']) for k in att_classes)
  
    data_loader = DataLoader(loader,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    

    device = "cuda" if GPU_is_available() else "cpu"
    sm = nn.Softmax(dim=1)

    inception_model = nn.DataParallel(
            SpectrogramInception3(num_classes, aux_logits=False))
    inception_model.to(device)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, inception_model.parameters()),
                       betas=[0, 0.99], lr=0.001)

    criterion = ACGANCriterion(att_dict)
    epochs = trange(n_epoch, desc='train-loop') 

    for i in epochs:
        data_iter = iter(data_loader)
        iter_bar = trange(len(data_iter), desc='epoch-loop')
        inception_model.train()
        for j in iter_bar:
            input, target = data_iter.next()

            input.requires_grad = True

            # take magnitude
            mag_input = F.interpolate(input[:, 0:1], (299, 299))
            optim.zero_grad()

            
            output = inception_model(mag_input.float().to(device))
            loss = criterion.getCriterion(output, target.to(device))

            loss.backward()
            state_msg = f'Iter: {j}; loss: {loss:0.2f} '
            iter_bar.set_description(state_msg)
            optim.step()

        # SAVE CHECK-POINT
        if i % 10 == 0:
            if isinstance(inception_model, torch.nn.DataParallel):
                torch.save(inception_model.module.state_dict(), output_file)
            else:
                torch.save(inception_model.state_dict(), output_file)

        # EVALUATION
        with torch.no_grad():
            inception_model.eval()

            val_i = int(np.ceil(len(val_data) / batch_size))
            vloss = 0
            prec = 0
            y_pred = []
            y_true = []
            prec = {k: 0 for k in att_classes}
            for k in range(val_i):
                vlabels = val_labels[k*batch_size:batch_size * (k+1)]
                vdata = val_data[k*batch_size:batch_size * (k+1)]
                vdata = F.interpolate(vdata, (299, 299))

                vpred = inception_model(vdata.to(device))
                vloss += criterion.getCriterion(vpred, vlabels.to(device)).item()
                vlabels_pred, _ = criterion.getPredictionLabels(vpred)
                y_pred.append(vlabels_pred)
                # y_true += list(vlabels)

            y_pred = torch.cat(y_pred)
            for i, c in enumerate(att_classes):
                
                # if class is xentroopy...
                logging.info(c)
                pred = [att_dict[c]['values'][int(l)] for l in y_pred.t()[i]]
                true = [att_dict[c]['values'][int(l)] for l in val_labels.t()[i]]
                cm = confusion_matrix(
                    true, pred,
                    labels=att_dict[c]['values'])
                print("")
                print("Confusion Matrix")
                print(cm)
                logging.info(cm)
                print("")
                crep = classification_report(true, pred, target_names=[str(v) for v in att_dict[c]['values']])
                logging.info(crep)
                print(crep)
            state_msg2 = f'epoch {i}; val_loss: {vloss / val_i: 0.2f}'
            logging.info(state_msg2)
            epochs.set_description(state_msg2)



if __name__=='__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--name', dest='name', type=str, default="default_inception_model",
                     help="Name of the output inception model")

    argparser.add_argument('-c', '--config', dest='config', type=str, default="default_inception_model",
                     help="Name of the output inception model")

    argparser.add_argument('-p', '--path', dest='path', type=str,
                     default=dirname(realpath(__file__)))
    argparser.add_argument('--batch-size', dest='batch_size', type=int, default=100,
                     help="Name of the output inception model")
    argparser.add_argument('-l', '--labels', dest='labels', nargs='+', help='Labels to train on')
    argparser.add_argument('-e', '--epochs', dest='n_epoch', type=int, default=100,
                           help='Labels to train on')

    args = argparser.parse_args()

    train_inception_model(**vars(args))

