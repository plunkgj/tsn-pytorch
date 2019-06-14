# Changes in this fork:
This fork was created to migrate the TSN-Pytorch code to torch 1.0, and to operate on the synthetic SimulatedHome dataset.
## Main changes:
- Currently tested and working for the BNInception convolutional model.
- Changed the model from using 2 greyscale flow images to a single RGB image. Optical flow is output by Unity's ML-Synthesis package, which encodes X and Y flow in a single RGB image. 
- Migrated from Torch0.1 to Torch1.0.
- Convolution models besides BNInception not tested.
- README from origial repository reproduced below:

-------------------------------------------------------------

# TSN-Pytorch

*Now in experimental release, suggestions welcome*.

**Note**: always use `git clone --recursive https://github.com/yjxiong/tsn-pytorch` to clone this project. 
Otherwise you will not be able to use the inception series CNN archs. 

This is a reimplementation of temporal segment networks (TSN) in PyTorch. All settings are kept identical to the original caffe implementation.

For optical flow extraction and video list generation, you still need to use the original [TSN codebase](https://github.com/yjxiong/temporal-segment-networks).

## Training

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For flow models:

```bash
python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_  
```

For RGB-diff models:

```bash
python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB <ucf101_rgb_val_list> ucf101_bninception_rgb_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name>

```

Or for flow models:
 
```bash
python test_models.py ucf101 Flow <ucf101_rgb_val_list> ucf101_bninception_flow_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name> --flow_pref flow_

```
