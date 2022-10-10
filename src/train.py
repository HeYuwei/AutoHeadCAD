from util.basic import *
from util.metrics import TimeMeter
from models import create_model
from options.base_options import TrainOptions
from datasets import create_dataset
from util.visualizer import Visualizer
import os


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt, 'train')  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if opt.valid_model:
        v_dataset = create_dataset(opt,'valid')
        print('The number of validation images = %d' % len(v_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    valid_iters = 0
    valid_freq = opt.valid_freq

    iter_time_meter = TimeMeter()
    data_time_meter = TimeMeter()
    epoch_time_meter = TimeMeter()

    print('Start to train')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        # shuffle bags
        opt.epoch = epoch
        dataset.dataset.prepare_new_epoch()
        if opt.single_valid_freq_epoch is not None and epoch >= opt.single_valid_freq_epoch:
            valid_freq = len(dataset) // opt.batch_size

        epoch_iter = 0            # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_time_meter.start()  # timer for entire epoch
        data_time_meter.start()
        iter_time_meter.start()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record()

            iter_time_meter.start()
            visualizer.reset()
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            iter_time_meter.record()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                visualizer.print_current_info(epoch, epoch_iter, model, iter_time_meter.val, data_time_meter.val)

            if total_iters % valid_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                model.update_metrics('global')
                visualizer.print_global_info(epoch, epoch_iter, model, iter_time_meter.sum/60,data_time_meter.sum/60)
            
                iter_time_meter.reset()
                data_time_meter.reset()

                model.reset_meters()
                model.clear_info()
                if opt.valid_model:
                    model.validation(v_dataset, visualizer, valid_iter=valid_iters)
                model.update_learning_rate()

                save_suffix = 'optimal'
                model.save_networks(save_suffix,visualizer)

                if model.wait_over:
                    break

                model.reset_meters()
                valid_iters += 1

            data_time_meter.start()
            iter_time_meter.start()

        if model.wait_over:
            print('early stop at %d / %d' % (epoch,epoch_iter))
            break

        epoch_time_meter.record()
        epoch_time_meter.start()
        
        model.next_epoch()

        print('End of epoch %d / %d \t Time Taken: %d hours' % (epoch, opt.niter + opt.niter_decay, epoch_time_meter.sum/3600.))
