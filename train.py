import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.measure import compare_ssim as ssim
import time
from models import ConvLSTM, PhyCell, EncoderRNN
from dataloader_blackWhite import VideoFramesDataloader
from constrain_moments import K2M
import argparse
from torchvision.utils import save_image
from torchvision import models
import os, sys
from utilsPhyDnet import get_features, compute_style_loss, compute_content_loss

sys.path.append('RAFT/core')
from raft import RAFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='folder for dataset')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--n_epochs', type=int, default=700, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=5, help='')
parser.add_argument('--out_dir', type=str, default='Test10_RAFT_3Ball128Mass_diffOrder7_128x128_blackWhite_gradually',
                    help='')

parser.add_argument('--model', default='RAFT/models/raft-sintel.pth', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

num_input = 5
num_output = 25
differential_order = 7

image_size = 128

RES_DIR = 'output'
LOGS_DIR = 'logs'
OUTPUT_DIR = args.out_dir

OUTPUT_FILE_IN = RES_DIR + '/' + OUTPUT_DIR + '/' + 'input_iteration_%s_sequence_%s.png'
OUTPUT_FILE_TARGET = RES_DIR + '/' + OUTPUT_DIR + '/' + 'target_iteration_%s_sequence_%s.png'
OUTPUT_FILE_PRED = RES_DIR + '/' + OUTPUT_DIR + '/' + 'prediction_iteration_%s_sequence_%s.png'

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if not os.path.exists(RES_DIR + '/' + OUTPUT_DIR):
    os.makedirs(RES_DIR + '/' + OUTPUT_DIR)


def log(message):
    print(message)
    with open(LOGS_DIR + '/' + args.out_dir + '.txt', 'a') as log_file:
        log_file.write(message + '\n')


vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

flow_correction_loss_weight = 0.000005
topology_aware_weight = 0.00001

RAFT_model = torch.nn.DataParallel(RAFT(args))
RAFT_model.load_state_dict(torch.load(args.model))
RAFT_model = RAFT_model.module
RAFT_model = RAFT_model.cuda()
RAFT_model.eval()

dataset_path_train = './datasets/train_data'
dataset_path_val = './datasets/val_data'

PRETRAINED_MODEL_NAME = 'Test10_RAFT_3Ball128Mass_diffOrder7_128x128_blackWhite_gradually'

mm = VideoFramesDataloader(args.root, dataset_path_train, is_train=True, n_frames_input=num_input,
                           n_frames_output=num_output, sequence_len=30, image_size=image_size)
train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, num_workers=0)

mm = VideoFramesDataloader(args.root, dataset_path_val, is_train=False, n_frames_input=num_input,
                           n_frames_output=num_output, sequence_len=30, image_size=image_size)
test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=1, shuffle=False, num_workers=0)

constraints = torch.zeros((differential_order * differential_order, differential_order, differential_order)).to(device)
ind = 0
for i in range(0, differential_order):
    for j in range(0, differential_order):
        ind += 1


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, flow_loss=0):
    encoder_optimizer.zero_grad()
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    total_mse_loss = 0
    total_perceptual_loss = 0
    total_structural_correction_loss = 0
    total_moment_loss = 0
    total_topology_aware = 0

    loss = 0

    for ei in range(input_length - 1):
        encoder_output, encoder_hidden, output_image, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
        mse_loss = criterion(output_image, input_tensor[:, ei + 1, :, :, :])
        topology_aware_loss = 0
        total_mse_loss += mse_loss
        loss += mse_loss

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence

    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
        decoder_input = output_image
        target = target_tensor[:, di, :, :, :]
        mse_loss = criterion(output_image, target)
        topology_aware_loss = 0
        loss += mse_loss
        total_mse_loss += mse_loss

        perc_loss = perceptual_loss(output_image, target)
        loss += perc_loss
        total_perceptual_loss += perc_loss

        if di < flow_loss:
            flow_low, flow_up = RAFT_model(target.detach(), output_image, iters=10, test_mode=True)
            flow_correction_loss = torch.abs(flow_up).sum()  # exponent to increase high values
            structural_correction_loss = flow_correction_loss * flow_correction_loss_weight
            loss += structural_correction_loss
            total_structural_correction_loss += structural_correction_loss

    k2m = K2M([differential_order, differential_order]).to(device)
    for b in range(0, encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)

        m = k2m(filters.double())
        m = m.float()
        moment_loss = criterion(m, constraints)
        loss += moment_loss  # constrains is a precomputed matrix
        total_moment_loss += moment_loss

    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length, total_mse_loss, total_perceptual_loss, total_moment_loss, total_structural_correction_loss, total_topology_aware


def trainIters(encoder, n_epochs, print_every, eval_every):
    train_losses = []
    flow_loss = 30

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    for epoch in range(0, n_epochs):
        t0 = time.time()
        loss_epoch = 0

        for i, out in enumerate(train_loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)

            loss, total_mse_loss, total_perceptual_loss, total_moment_loss, total_structural_correction_loss, total_topology_aware = \
                train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, flow_loss)

            loss_epoch += loss

            if (i % 100) == 0:
                log("%s/%s [total_mse_loss=%s total_perceptual_loss=%s total_moment_loss=%s total_structural_correction_loss=%s total_topology_aware=%s] total_loss=%s" % (
                i, len(train_loader), total_mse_loss, total_perceptual_loss, total_moment_loss,
                total_structural_correction_loss, total_topology_aware, loss))
                save_output(encoder, test_loader)

        train_losses.append(loss_epoch)
        if (epoch + 1) % print_every == 0:
            log("%s - epoch %s loss %s epoch time %s" % (OUTPUT_DIR, epoch, loss_epoch, time.time() - t0))

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, test_loader)
            scheduler_enc.step(mse)

        try:
            torch.save(encoder.state_dict(), RES_DIR + '/' + OUTPUT_DIR + '.pth')
        except:
            log("Failed to save model")

    return train_losses


def save_output(encoder, loader):
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)

            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            try:
                for j, image in enumerate(input_tensor[0]):
                    save_image(image, OUTPUT_FILE_IN % (i, j))

                for j, image in enumerate(target_tensor[0]):
                    save_image(image, OUTPUT_FILE_TARGET % (i, j))

                for j, image in enumerate(predictions):
                    save_image(image, OUTPUT_FILE_PRED % (i, j))
                break
            except:
                print("Failed to save images")
                break


def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim, total_bce = 0, 0, 0, 0
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)

            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)
            predictions = predictions.swapaxes(0, 1)

            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0,], predictions[a, b, 0,]) / (target.shape[0] * target.shape[1])

            cross_entropy = -target * np.log(predictions) - (1 - target) * np.log(1 - predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size * target_length)
            total_bce += cross_entropy

    print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ',
          total_ssim / len(loader), ' eval bce ', total_bce / len(loader))
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)


# orig perceptual loss style_weight=0.05, content_weight=0.1
def perceptual_loss(output_image, target, style_weight=0.05, content_weight=0.1):
    output_image_features = get_features(output_image, vgg)
    target_features = get_features(target, vgg)
    content_loss = compute_content_loss(target_features['conv4_2'], output_image_features['conv4_2']) * content_weight
    style_loss = compute_style_loss(output_image_features, target_features) * style_weight
    return content_loss + style_loss


log('BEGIN TRAIN')
phycell = PhyCell(input_shape=(16, 16), input_dim=image_size, F_hidden_dims=[differential_order * differential_order],
                  n_layers=1, kernel_size=(differential_order, differential_order), device=device)
convlstm = ConvLSTM(input_shape=(16, 16), input_dim=image_size,
                    hidden_dims=[image_size * 2, image_size * 2, image_size], n_layers=3, kernel_size=(3, 3),
                    device=device)
encoder = EncoderRNN(phycell, convlstm, device, image_size)

# encoder.load_state_dict(torch.load('output/' + PRETRAINED_MODEL_NAME + '.pth'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


log(torch.__version__)
log("phycell %s" % count_parameters(phycell))
log("convlstm %s" % count_parameters(convlstm))
log("opticalFlowNetwork %s" % count_parameters(encoder))

plot_losses = trainIters(encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
print(plot_losses)
