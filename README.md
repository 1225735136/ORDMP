# ORDMP
Our base model is [Manydepth2](https://github.com/kaichen-z/Manydepth2)
## Data Preparation

Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.  
Please refer to [manydepth](https://github.com/nianticlabs/manydepth) and [SfMLearner]( https://github.com/tinghuiz/SfMLearner/tree/master )to prepare your Cityscapes data.  
Follow the manydepth we get the groundtruth of cityscapes.  

Different from SfMLearner ,we use the following instruction  
<pre>
```python prepare_train_data.py --dataset_dir /home/jsw/datasets/cityscapes/cs_raw_sequence --dataset_name cityscapes --dump_root /home/jsw/datasets/cityscapes/cs_raw_sequence_format/ --seq_length 3 --img_width
1024 --img_height 512 --num_threads 24
```
</pre>
This instruction resize the image from 1024x2048 to 512x1024, and crop the bottom part contained car logo. Finally the image will be 384*1024, and the saved image will be three pictures joined together which has the size of 384x3072.
The train images are stored at the cs_raw_sequence_format/ and the test images are stored in the leftImg8bit/test  
When trainning on cityscapes, the original image is a long image which has three images in one picture, we get the first part as previous image,seconde part as current image, third part as future image.  
## **Train On Kitti**
<pre>
```
python train.py --data_path /home/jsw/datasets/cityscapes/cs_raw_sequence_preprocessed --log_dir logs --png --freeze_teacher_epoch 5 --model_name models_many2_new10 --pytorch_random_seed 1 --batch_size 12 --mode many2 --split cityscapes_preprocessed --dataset cityscapes_preprocessed --eval_split cityscapes --height 192 --width 512  
```
</pre>
## **Train On Cityscapes**
You can train the model on cityscapes. But the result is bad. We think the better choice is finetune the model which pretrained on the kitti dataset.
<pre>
```
python train.py --data_path /home/jsw/datasets/cityscapes/cs_raw_sequence_preprocessed --log_dir logs --png --freeze_teacher_epoch 5 --model_name models_many2_new10 --pytorch_random_seed 1 --batch_size 12 --mode many2 --split cityscapes_preprocessed --dataset cityscapes_preprocessed --eval_split cityscapes --height 192 --width 512  
```
</pre>
</pre>
## **Finetune On Cityscapes**
The better choice: Finetune the model on Cityscapes which has the pretrained weight on the kitti dataset.
<pre>
```
python train.py --weights_init pretrained --load_weights_folder /home/jsw/Manydepth2-master/manydepth2/logs/models_many2_kitti/models/weights_19 --data_path /home/jsw/datasets/cityscapes/cs_raw_sequence_preprocessed --log_dir logs --png --freeze_teacher_epoch 5 --model_name models_many2_new10 --pytorch_random_seed 1 --batch_size 12 --mode many2 --split cityscapes_preprocessed --dataset cityscapes_preprocessed --eval_split cityscapes --height 192 --width 640 
```
</pre>




