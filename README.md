Automatic Number (License) Plate Recognition
============================================
![](https://github.com/stevefielding/tensorflow-anpr/raw/master/uploads/parkingLotShortClip.gif)
Detect vehicle license plates in videos and images using the tensorflow/object_detection API.  
Train object detection models for license plate detection using TFOD API, with either a single detection stage
or a double detection stage.  
The single stage detector, detects plates and plate characters in a single inference stage.  
The double stage detector detects plates in the first inference stage,  
![](https://github.com/stevefielding/tensorflow-anpr/raw/master/uploads/detect_plate.png)  
crops the detected plate from the image, passes the cropped plate image to the second inference stage, 
which detects plate characters.  
![](https://github.com/stevefielding/tensorflow-anpr/raw/master/uploads/detect_chars.png)   
The double stage detector uses a single detection model that has been trained to detect plates in full images containing cars/plates, 
and trained to detect plate text in images containing tightly cropped plate images. 


You will need a starting point for training. You can either use
the exported_models (which speeds up training) or you can download ssd_inception_v2_coco_2018_01_28, 
and faster_rcnn_resnet101_coco_2018_01_28 from the [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  
The models should be copied to experiment_ssd/YYYY_MM_DD/training, and experiment_faster_rcnn/YYYY_MM_DD/training  
In the config files, you will need to modify the paths:

train_config
  fine_tune_checkpoint path/to/my/model.ckpt
train_input_reader
  tf_record_input_reader
    input_path path/to/my/train.record
  label_map_path path/to/my/classes.pbtxt
train_eval_reader
  tf_record_input_reader
    input_path path/to/my/test.record
  label_map_path path/to/my/classes.pbtxt
````
##### Performance
Single stage Faster RCNN:  
[INFO] Processed 69 frames in 37.62 seconds. Frame rate: 1.83 Hz  
[INFO] platesWithCharCorrect_recall: 97.1%, platesWithCharCorrect_precision: 97.1%,  
       plateFrames_recall: 100.0%, plateFrames_precision: 100.0%,  
       chars_recall: 99.6%, chars_precision: 99.6%  
Two stage SSD:  
[INFO] Processed 69 frames in 6.17 seconds. Frame rate: 11.19 Hz  
[INFO] platesWithCharCorrect_recall: 95.7%, platesWithCharCorrect_precision: 97.1%,  
       plateFrames_recall: 98.6%, plateFrames_precision: 100.0%,  
       chars_recall: 98.1%, chars_precision: 99.6%  
It seems that Faster_RCNN is slightly better than SSD, but the sample size is too small to be sure. 

##### Object in object
This two stage technique of using a single model to detect characters within plates could also be used to detect any 
object within another object. The two stage detector can perform inference faster than the single stage
detector, because it can use simpler models such as SSD, rather than Faster RCNN, and it can use a smaller
image size. For example, two stages of SSD based on Inception_V2, and 1280x960 image size can perform inference at 
1.8 fps on a Titan-X GPU, whereas, a single stage of Faster RCNN based on Resnet_101, and a resized 300x300 image 
can perform inference at 15 fps on a Titan-X.  
The two stage SSD implementation opens the possibility of running on less powerful hardware, such as Intel 
i7-4790K CPU @ 4.00GHz with 16GB of RAM (2.9 fps), and Nvidia Jetson TX2 (2.8 fps).

Generating labelled images
--------------------------
##### mturk.html:
Defines the web interface that will be used by the MTurk workers to label the images.
Modified from [original](https://github.com/kyamagu/bbox-annotator)
Use this html/js code with Amazon mechanical Turk.
Instructions [here](https://blog.mturk.com/tutorial-annotating-images-with-bounding-boxes-using-amazon-mechanical-turk-42ab71e5068a).  
This code needs to be improved to allow image zoom. Without zoom capability it is too difficult for the workers 
to create the boxes for the small characters of the license plate text. 
Consequently you can spend a lot of time re-arranging the boxes in the labelimg utility.

Train the object_detection model
--------------------------------
Now you can use the TFOD API, at tensorflow/models/research/object_detection, to train the model.
It goes something like this. Assuming python virtualenv called tensorflow, 
a single GPU for training and CPU for eval:

##### Training
````
workon tensoflow  
cd tensorflow/models/research/object_detection
python train.py --logtostderr \  
--pipeline_config_path ../anpr/experiment_faster_rcnn/2018_06_12/training/faster_rcnn_anpr.config \  
--train_dir ../anpr/experiment_faster_rcnn/2018_06_12/training
````
##### Eval
If you are running the eval on a CPU, then limit the number of images to evaluate by modifying your config file:  
````
130 eval_config: {  
131 num_examples: 5  
````
New terminal 
```` 
cd tensorflow/models/research/object_detection
workon tensoflow  
export CUDA_VISIBLE_DEVICES=""  
python eval.py --logtostderr \  
--checkpoint_dir ../anpr/experiment_faster_rcnn/2018_06_12/training \  
--pipeline_config_path ../anpr/experiment_faster_rcnn/2018_06_12/training/faster_rcnn_anpr.config \  
--eval_dir ../anpr/experiment_faster_rcnn/2018_06_12/evaluation
````
New terminal
````
cd tensorflow/models/research  
workon tensoflow  
tensorboard --logdir anpr/experiment_faster_rcnn  
````
##### Export model
````
cd tensorflow/models/research/object_detection
workon tensorflow  
python export_inference_graph.py --input_type image_tensor \  
--pipeline_config_path ../anpr/experiment_faster_rcnn/2018_06_12/training/faster_rcnn_anpr.config \  
--trained_checkpoint_prefix ../anpr/experiment_faster_rcnn/2018_06_12/training/model.ckpt-60296 \  
--output_directory ../anpr/experiment_faster_rcnn/2018_06_12/exported_model
````
Testing the trained model
-------------------------
##### predict_images.py
Back to this project directory to run predict_images.py
Test your exported model against an image dataset. Works with single and double stage prediction.
Prints the detected plate text, and displays the annotated image.  
pred_stages = 1
````
workon tensorflow
python predict_images.py --model datasets/experiment_faster_rcnn/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
 --pred_stages 1 \
 --labels datasets/records/classes.pbtxt \
 --imagePath images/SJ7STAR_images/2018_02_24_9-00 \
 --num-classes 37 \
 --image_display True 
````
pred_stages = 2
````
python predict_images.py --model datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
 --pred_stages 2 \
 --labels datasets/records/classes.pbtxt \
 --imagePath images/SJ7STAR_images/2018_02_24_9-00 \
 --num-classes 37 \
 --image_display True 
````
##### predict_video.py
Test your exported model against a video dataset. Uses one or two two stage prediction.
Outputs an annotated video and a series of still images along with image annotations. The still images are grouped to reduce 
the output of images with duplicate plates.
The image annotations can be viewed and edited using labelImg, and they can be used to further enlarge the training dataset.
````
python predict_video.py --conf conf/lplates_smallset_ssd.json
````
##### predict_images_and_score.py
Test a trained model against an annotated dataset. Annotations must be in PASCAL VOC style xml files.
Run with image_display true if you wish to see each annotated image displayed.
Can be executed with single or double prediction stages.  
pred_stages = 1:  
plates and characters are predicted in a single pass
````
python predict_images_and_score.py --model datasets/experiment_faster_rcnn/2018_07_15/exported_model/frozen_inference_graph.pb \
--labels datasets/records/classes.pbtxt \
--annotations_dir images_verification \
--num-classes 37 \
--min-confidence 0.5 \
--pred_stages 1
````
pred_stages = 2:  
The first prediction stage predicts plates, crops the predicted plate from the image, and then 
the cropped plate image is used as input to the second prediction stage, which predicts characters. 
````
python predict_and_score.py --model datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
--labels datasets/records/classes.pbtxt \
--annotations_dir images_verification \
--num-classes 37 \
--min-confidence 0.1 \
--pred_stages 2
Single stage faster R-CNN usage
````
Your results should look something like this:
````
[INFO] Processed 925 frames in 70.13 seconds. Frame rate: 13.19 Hz
[INFO] platesWithCharCorrect_recall: 93.2%, platesWithCharCorrect_precision: 93.9%, 
       plateFrames_recall: 99.2%, plateFrames_precision: 100.0%, 
       chars_recall: 98.3%, chars_precision: 99.2%
[INFO] Definitions. Precision: Percentage of all the objects detected that are correct. Recall: Percentage of ground truth objects that are detected
[INFO] Processed 925 xml annotation files
````
And an explanation of the performance metrics:

````
platesWithCharCorrect_recall - Plates detected in correct location and containing correct characters in the correct locations
                               divided by the number of ground truth plates
platesWithCharCorrect_precision - Plates detected in correct location and containing correct characters in the correct locations
                                  divided by the total number of plates predicted (ie true pos plus false pos)
plateFrames_recall - Plate frames in correct location (no checking of characters) divided by the
                     number of ground truth plates
plateFrames_precision - Plate frames in correct location (no checking of characters) divided by the
                        the total number of plates predicted (ie true pos plus false pos)
chars_recall - Characters detected in the correct place with the correct contents
               divided by the number of ground truth characters
chars_precision - Chars detected outside of the correct location, or the location is correct,
                  but the contents are wrong. Divided by the total number of plates predicted (ie true pos plus false pos)
 ````
