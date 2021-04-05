from argparse import ArgumentParser

# Process command line arguments
parser = ArgumentParser()

parser.add_argument('--train', type=int, default=1,
                    help='Train network (default=1 (True)')

parser.add_argument('--test', type=int, default=1,
                    help='Test network (default=1 (True)')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size of the training and test set (default=32)')

parser.add_argument('--lr', type=float, default=1e-04,
                    help='Main learning rate of the adam optimizer (default=1e-04)')

parser.add_argument('--channel_factor', type=float, default=1.0,
                    help='Channel factor adopts the number of channels utilized in the U-Net (default=1)')

parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use (default=cuda)')

parser.add_argument('--gpus_to_use', type=str, default='0',
                    help='Indexes of the GPUs to be use (default=0)')

parser.add_argument('--use_data_parallel', type=int, default=0,
                    help='Use multiple GPUs (default=0 (False))')

parser.add_argument('--load_generator_network', type=str, default=None,
                    help='Name of the generator network the be loaded from model file (.pt) (default=None)')

parser.add_argument('--load_discriminator_network', type=str, default=None,
                    help='Name of the discriminator network the be loaded from model file (.pt) (default=None)')

# Updated to retrained classifier Jamie 24/01 08:57
# Updated after correctly saving classifier model as model file and not state dict - Jamie 24/01 09:19
parser.add_argument('--load_pretrained_vgg16', type=str, default='pre_trained_models/VGG16_P10_50ep.pt',
                    help='Name of the pretrained (places365) vgg16 network the be loaded from model file (.pt)')

# Updated to new training data - Jamie 24/01 08:57
parser.add_argument('--path_to_places365', type=str, default='../Training Data/Places_Nature10',
                    help='Path to places365 dataset.')

parser.add_argument('--epochs', type=int, default=50,
                    help='Epochs to perform while training (default=100)')

parser.add_argument('--model_wrapper', type=int, default=0,
                    help='Select a valid model wrapper option:\n\t0: Original SGP'
                         '\n\t1: Self-Supervised SGP w/ Aux Rotation')

parser.add_argument('--visualise', type=int, default=0,
                    help='Select a valid option:\n\t0: Skip Visualisation'
                         '\n\t1: Perform Visualisation')

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_to_use

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import Generator, Discriminator, DiscriminatorAuxRotation, VGG16
from model_wrapper import ModelWrapper
from aux_rotation_model_wrapper import AuxRotationModelWrapper
import data

if __name__ == '__main__':
    # Init models
    if args.load_generator_network is None:
        generator = Generator(channels_factor=args.channel_factor)
    else:
        generator = torch.load(args.load_generator_network)
        if isinstance(generator, nn.DataParallel):
            generator = generator.module
    if args.load_discriminator_network is None:
        discriminator = Discriminator(channel_factor=args.channel_factor)
    else:
        discriminator = torch.load(args.load_discriminator_network)
        if isinstance(discriminator, nn.DataParallel):
            discriminator = discriminator.module
    vgg16 = VGG16(args.load_pretrained_vgg16)

    # Init data parallel
    if bool(args.use_data_parallel):
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        vgg16 = nn.DataParallel(vgg16)

    # Init optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-05)#, betas=adam_betas)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-07)#, betas=adam_betas)
    # Print number of network parameters
    print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
    print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))

    # Init dataset
    # Changed num of workers to 2 instead of batch size - Jamie 25/01/21 21:04
    training_dataset = DataLoader(
        data.Places365(path_to_index_file=args.path_to_places365, index_file_name='train.txt'),
        batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True,
        collate_fn=data.image_label_list_of_masks_collate_function)

    # Changed max_length from 6000 to 200 - Jamie TODO: Find optimal max_length value
    # Back to 50, believe this is causing an error i.e. CUDA out of memory
    validation_dataset_fid = DataLoader(
        data.Places365(path_to_index_file=args.path_to_places365, index_file_name='val.txt',
                       max_length=50, validation=True),
        batch_size=args.batch_size, num_workers=2, shuffle=False,
        collate_fn=data.image_label_list_of_masks_collate_function)
    validation_dataset = data.Places365(path_to_index_file=args.path_to_places365, index_file_name='val.txt',
                                        test=True)

    # Initialises chosen model wrapper for training - Jamie 12/02 15:06
    if args.model_wrapper == 0:
        model_wrapper = ModelWrapper(generator=generator,
                                     discriminator=discriminator,
                                     vgg16=vgg16,
                                     training_dataset=training_dataset,
                                     validation_dataset=validation_dataset,
                                     validation_dataset_fid=validation_dataset_fid,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    elif args.model_wrapper == 1:
        discriminator2 = DiscriminatorAuxRotation(channel_factor=args.channel_factor)
        model_wrapper = AuxRotationModelWrapper(generator=generator,
                                                discriminator=discriminator2,
                                                vgg16=vgg16,
                                                training_dataset=training_dataset,
                                                validation_dataset=validation_dataset,
                                                validation_dataset_fid=validation_dataset_fid,
                                                generator_optimizer=generator_optimizer,
                                                discriminator_optimizer=discriminator_optimizer,
                                                weight_rotation_loss_g=0.1,
                                                weight_rotation_loss_d=0.1)
    else:
        print("ERR: Select a valid model wrapper option:\n\t0: Original SGP\n\t1: Self-Supervised SGP w/ Aux Rotation")
        exit(1)

    # Performs training - TODO: Update one of the wrappers so I can get rid of unnecessary if statement
    if bool(args.train):
        if args.model_wrapper == 0:
            # Testing - Jamie 12/02 15:19
            # print("Here 0")
            # exit(0)
            model_wrapper.train(epochs=args.epochs, device=args.device, w_rec=0.1, w_div=0.1)
        elif args.model_wrapper == 1:
            # Testing - Jamie 12/02 15:19
            # print("Here 1")
            # exit(0)
            model_wrapper.train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    # Perform testing
    if bool(args.test):
        print('FID=', model_wrapper.validate(device=args.device))
        model_wrapper.inference(device=args.device)
