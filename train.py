import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.utils.data as dt

from carvana_dataset import CarvanaDataset
from model import SegmenterModel

from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

from tqdm.auto import tqdm
import numpy as np


LOG_DIR = '/data/rvgorb/hw11/log/'
TRAIN_DIR = '/data/rvgorb/hw11/train/'
TRAIN_MASKS_DIR = '/data/rvgorb/hw11/train_masks/'
TEST_DIR = '/data/rvgorb/hw11/test/'
TEST_MASKS_DIR = '/data/rvgorb/hw11/test_masks/'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
else:
    shutil.rmtree(LOG_DIR)

tb_writer = SummaryWriter(log_dir=LOG_DIR)

DEVICE_ID = 6
DEVICE = torch.device(f"cuda:{DEVICE_ID}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    model = SegmenterModel(3, 1).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    ds = CarvanaDataset(TRAIN_DIR, TRAIN_MASKS_DIR)
    ds_test = CarvanaDataset(TEST_DIR, TEST_MASKS_DIR, is_train=False)

    dl      = dt.DataLoader(ds, shuffle=True, 
                            num_workers=8, 
                            batch_size=args.batch_size, pin_memory=True)
    dl_test = dt.DataLoader(ds_test, shuffle=False, 
                            num_workers=8,
                            batch_size=args.batch_size, pin_memory=True)

    global_iter = 0
    for epoch in range(0, args.n_epochs):
        print ("Current epoch: ", epoch)
        epoch_loss = 0
        
        model.train(True)
        for i, (input_batch, target_batch) in enumerate(tqdm(dl)):
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            output_batch = model(input_batch)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            
            global_iter += 1
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / float(len(ds))
        print ("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print ("Make test")
        test_loss = 0
        model.train(False)
        tb_out = np.random.choice(range(0, len(dl_test)), 3)
        for i, (input_batch, target_batch) in enumerate(tqdm(dl_test)):
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            
            with torch.no_grad():
                output_batch = model(input_batch)
            loss = criterion(output_batch, target_batch)
            test_loss += loss.item()

            for img_id, checkpoint in enumerate(tb_out):
                if checkpoint == i:
                    tb_writer.add_image(f'Image/Test_input_{img_id}',  
                                        input_batch[0].cpu(), 
                                        epoch)
                    tb_writer.add_image(f'Image/Test_target_{img_id}', 
                                        target_batch[0].cpu(), 
                                        epoch)
                    tb_writer.add_image(f'Image/Test_output_{img_id}', 
                                        output_batch[0].cpu() > 0, 
                                        epoch)

        test_loss = test_loss / float(len(ds_test))
        print ("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)
        
        torch.save(model.state_dict(), "/data/rvgorb/hw11/unet_dump_recent")


if __name__ == '__main__':
    main()
    