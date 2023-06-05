import torch
import time
import math
import argparse
from utils import *
import numpy as np
from model import U_Network,SpatialTransformer
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default='Data/fixed.nii.gz')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')
# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="Data/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--epochs", type=int, help="number of iterations",
                    dest="epochs", default=1000)
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=4.0)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_epoch", type=int, help="frequency of model saves",
                    dest="n_save_epoch", default=10)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')
# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='Data/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='Data/label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="./Checkpoint/LPBA40.pth")
args = parser.parse_args()

def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    log_name = str(args.epochs) + "_" + str(args.lr) + "_" + str(args.alpha)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")
    writer = SummaryWriter(log_dir=args.log_dir)
    print("Writing logs to ", args.log_dir, log_name)
    
    # 读入fixed图像
    fixed_image = Readimage(args.atlas_file)[np.newaxis, np.newaxis, ...]
    vol_size = fixed_image.shape[2:]
    fixed_image = np.repeat(fixed_image, args.batch_size, axis=0)
    fixed_image = torch.from_numpy(fixed_image).to(device).float()
    
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))

    optimizer = torch.optim.Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = ncc_loss
    grad_loss_fn = gradient_loss

    DS = Dataset(train_dir=args.train_dir)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(0, args.epochs):
        epoch_loss = AverageMeter()
        epoch_total_loss = AverageMeter()
        epoch_step_time = AverageMeter()
        losses = AverageMeter()
        sim_losses = AverageMeter()
        grad_losses = AverageMeter()
        
        eopch_start_time = time.time()
        start_time = time.time()
        for idx, batch_data in enumerate(DL):
            moving_image = batch_data.to(device).float()
            flow_m2f = UNet(moving_image, fixed_image)
            m2f = STN(moving_image, flow_m2f)
            sim_loss = sim_loss_fn(m2f, fixed_image)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.update(loss.item(), n=args.batch_size)
            sim_losses.update(sim_loss.item(), n=args.batch_size)
            grad_losses.update(grad_loss.item(), n=args.batch_size)
            
            print("Epoch {}/{} {}/{}".format(epoch, args.epochs, idx,len(DL)),
                "loss: {:.4f}  sim_loss: {:.4f}  grad_loss: {:.4f}".format(losses.avg, sim_losses.avg, grad_losses.avg),
                "time {:.2f}s".format(time.time() - start_time))
            start_time = time.time()
        print('=============================================================')    
        print("Epoch {}/{}".format(epoch, args.epochs),
                "loss: {:.4f}  sim_loss: {:.4f}  grad_loss: {:.4f}".format(losses.avg, sim_losses.avg, grad_losses.avg),
                "time {:.2f}s".format(time.time() - eopch_start_time))
        print('=============================================================') 
        print("Epoch {}/{}".format(epoch, args.epochs),
                "loss: {:.4f}  sim_loss: {:.4f}  grad_loss: {:.4f}".format(losses.avg, sim_losses.avg, grad_losses.avg),
                "time {:.2f}s".format(time.time() - eopch_start_time),file=f) 
        writer.add_scalar("loss", losses.avg, epoch)
        writer.add_scalar("loss", sim_losses.avg, epoch)
        writer.add_scalar("loss", grad_losses.avg, epoch)
        eopch_start_time = time.time()
        if (epoch + 1) % args.n_save_epoch == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
    f.close()


if __name__ == "__main__":
    train()

    

