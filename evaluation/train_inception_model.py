from torch import nn
from data.preprocessing import AudioPreprocessor
from tqdm import tqdm, trange
from utils.utils import mkdir_in_path, GPU_is_available
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from os.path import dirname, realpath, join

from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionC, \
    InceptionE, InceptionB, InceptionAux, InceptionD

from datetime import datetime


class SpectrogramInception3(nn.Module):
    def __init__(self, num_classes=128, aux_logits=True, transform_input=False):
        super(SpectrogramInception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        # self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        # self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        # 299 x 299 x 1
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=(2, 30))
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

def bce_loss(pi, x):
    sm = nn.Softmax()

    s = 1e-7
    pi = (1-2*s) * pi + s
    # loss = -(x * torch.log(pi) + (1-x) * torch.log(1-pi))
    loss = -x * torch.log(pi)

    #loss[torch.isnan(loss)] = 0
    loss = torch.mean(loss)
    assert not torch.isnan(loss), "bce loss is nan!"
    return loss

def train_inception_model(output_file,
                          att_cls="pitch",
                          dbsize=100000, 
                          labels=["mallet"],
                          batch_size=50):
    # path_out = mkdir_in_path(".", "inception_data")
    # path_out = "/ldaphome/jnistal/sandbox"
    path_out = "/home/javier/developer/inception_test"
    path_to_raw = "/home/javier/developer/datasets/nsynth-train/audio"
    att_dict_path = "/home/javier/developer/datasets/nsynth-train/examples.json"
    # path_to_raw = "/ldaphome/jnistal/data/nsynth-train/audio/"
    data_manager = AudioPreprocessor(data_path=path_to_raw,
                                     output_path=path_out,
                                     dbname='nsynth',
                                     sample_rate=16000,
                                     audio_len=16000,
                                     data_type='audio',
                                     transform='specgrams',
                                     db_size=dbsize,
                                     labels=labels,
                                     transformConfig=dict(
                                         n_frames=64,
                                         n_bins=128,
                                         fade_out=True,
                                         fft_size=1024,
                                         win_size=1024,
                                         hop_size=256,
                                         n_mel=256
                                     ),
                                     load_metadata=True,
                                     loaderConfig=dict(
                                                      size=dbsize,
                                                      instrument_labels=labels,
                                                      pitch_range=[44, 70],
                                                      filter_keys=['acoustic'],
                                                      attribute_list=[att_cls],
                                                      att_dict_path=att_dict_path
                                                      ))
    data_loader = data_manager.get_loader()

    val_data, val_labels = data_loader.train_val_split()
    val_data = val_data[:, 0:1]
    att_index = data_loader.getKeyOrders()[att_cls]['order']
    att_classes = data_loader.att_classes[att_index]
    num_classes = len(att_classes)
  
    data_loader = DataLoader(data_loader,
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
    # optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, inception_model.parameters()),
    #                    alpha=1.0, lr=0.045, weight_decay=0.9)
    # criterion = nn.BCEWithLogitsLoss()

    criterion = nn.CrossEntropyLoss()
    epoch_bar = trange(5000,
                  desc='train-loop') 

    for i in epoch_bar:
        data_iter = iter(data_loader)
        iter_bar = trange(len(data_iter), desc='epoch-loop')
        inception_model.train()
        for j in iter_bar:
            data = data_iter.next()
            inputs_real = data[0]
            inputs_real.requires_grad = True
            target = data[1][:, att_index]

            # take magnitude cqt
            mag_input = F.interpolate(inputs_real[:, 0:1], (299, 299))
            # mag_input = inputs_real
            optim.zero_grad()

            output = inception_model(mag_input.to(device))
            loss = criterion(output, target.to(device))

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
            import numpy as np
            inception_model.eval()

            val_i = int(np.ceil(len(val_data) / batch_size))
            val_loss = 0
            prec = 0
            y_pred = []
            y_true = []
            for k in range(val_i):
                vlabels = val_labels[k*batch_size:batch_size * (k+1)][:, att_index]
               
                val_output = inception_model(F.interpolate(val_data[k*batch_size:batch_size * (k+1)], (299, 299)))
                val_loss += criterion(val_output, vlabels.long()).item()

                val_p = sm(val_output).detach().to(device)
                val_out = list(map(lambda x: x.argmax(), val_p))
                y_pred += val_out
                y_true += list(vlabels)
                # val_str = midi2str([v.item() for v in val_out])
                # val_freq = midi2freq([v.item() for v in val_out])

                # confusion_matrix(val_output, )
                prec += (torch.stack(val_out) == vlabels.long()).sum() * 100 / len(vlabels)
            cm = confusion_matrix([att_classes[i.int()] for i in y_pred], [att_classes[i.int()] for i in y_true], labels=att_classes)
            print(cm)
            print(classification_report(y_true, y_pred, labels=np.arange(num_classes), target_names=att_classes))
            state_msg2 = f'm_precision: {prec / val_i: 0.2f} %; epoch {i}; m_val_loss: {val_loss / val_i: 0.2f}'
            epoch_bar.set_description(state_msg2)


            



if __name__=='__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--name', dest='name', type=str, default="default_inception_model",
                     help="Name of the output inception model")
    argparser.add_argument('-o', '--output', dest='output', type=str,
                     default=dirname(realpath(__file__)))
    argparser.add_argument('--dbsize', dest='dbsize', type=int, default=10000,
                     help="Name of the output inception model")
    
    argparser.add_argument('--batch-size', dest='batch_size', type=int, default=10000,
                     help="Name of the output inception model")
    argparser.add_argument('-l', '--labels', dest='labels', nargs='+', help='Labels to train on')
    argparser.add_argument('-a', '--att', dest='att_cls', type=str, default="pitch", help='Labels to train on')
    args = argparser.parse_args()

    output_path = mkdir_in_path(args.output, 'inception_models')
    output_file = join(output_path, f"{args.name}_{datetime.now().strftime('%Y-%m-%d')}.pt")


    train_inception_model(output_file, 
                          dbsize=args.dbsize,
                          batch_size=args.batch_size, 
                          labels=args.labels,
                          att_cls=args.att_cls)

