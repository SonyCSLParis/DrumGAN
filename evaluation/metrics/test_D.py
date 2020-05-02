import os
import torch

from datetime import datetime

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, get_device
from data.preprocessing import AudioProcessor
from gans.ac_criterion import ACGANCriterion
from data.loaders import get_data_loader
import ipdb
from torch.utils.data import DataLoader

from tqdm import trange


def test(parser):
    parser.add_argument('--size', dest='size', default=100, type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=50, type=int)
    args = parser.parse_args()
    kargs = vars(args)
    model, config, model_name = load_model_checkp(**kargs)

    transform_config = config['transform_config']
    loader_config = config['loader_config']

    d_net = model.getOriginalD()
    d_net.eval()
    # We load a dummy data loader for post-processing
    processor = AudioProcessor(**transform_config)

    dbname = loader_config['dbname']

    loader_config["criteria"]["size"] = args.size
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    att_dict = loader.header['attributes']
    criterion = ACGANCriterion(att_dict)

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"tests_D")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    batch_size = min(args.batch_size, len(loader))
    data_loader = DataLoader(loader,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    device = get_device()

    data_iter = iter(data_loader)
    iter_bar = trange(len(data_iter), desc='epoch-loop')

    D_loss = []
    data = []
    for j in iter_bar:
        input, target = data_iter.next()

        with torch.no_grad():
            pred = d_net(input.float().to(device)).cpu()
            clf_loss = criterion.getCriterion(pred, target.to(device))
            # get D loss
            D_loss.append(pred[:, -1])
            data.append(input)
            state_msg = f'Iter: {j}; avg D_nloss: {sum(pred[:, -1])/len(pred[:, -1]):0.3f}, classif_loss: {clf_loss:0.3f}'
            iter_bar.set_description(state_msg)
    # Create evaluation manager
    D_loss = torch.cat(D_loss)
    data = torch.cat(data)
    D_loss, idx = D_loss.sort()


    audio_out = loader.postprocess(data[idx])
    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='test_D_loss',
                   sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")