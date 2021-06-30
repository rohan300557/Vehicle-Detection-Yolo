


# Vehicle-Detection-Yolo
The goal of this project is to detect vehicles in an image and also in an video. In this project I implemented object detection using custom yolo model build using darknet and Opencv libray for detcting the object.
The Object detection means the detection on every single frame and frame after frame.

“You Only Look Once” (YOLO) is a popular algorithm for performing object detection due to its fast speed and ability to detect objects in real time.
The YOLO approach of the object detection is consists of two parts: the neural network part that predicts a vector from an image, and the postprocessing part that interpolates the vector as boxes coordinates and class probabilities.
Yolo algorithm “only looks once” at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.
We will use Yolo with Darknet framework. Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation The framework features You Only Look Once (YOLO), a state-of-the-art, real-time object detection system..

The demo link for the folowing Detection performed on video is [here](https://user-images.githubusercontent.com/60709999/124021170-dd295880-da08-11eb-9abd-4e3ddd2e912b.mp4)



### Training Yolo For Custom Data:

Firstly, we need a suitable dataset to train our custom object detection model.
*  In this project I took total of 556 images of different vehicles (Car, Ambulance, Bus and Truck).
* Then we will Label the images using [LabelImg](https://tzutalin.github.io/labelImg/). [LabelImg](https://github.com/tzutalin/labelImg#labelimg) is a graphical image annotation tool. The Annotations are saved as TXT files in YOLO format.
	The below image shown how the Annotation tool look like:
	
	<img src="https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/src/Labelimg.png" data-canonical-src="https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/src/Labelimg.png" width="400" height="300" />	
	
	The output of the following Format is given below:
	
    ![format.png](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/src/file_format.png)

* We will use [Darknet](https://github.com/pjreddie/darknet), an open source neural network framework to train the detector. We will  clone the the official darknet repository
	```python:
	!git clone https://github.com/AlexeyAB/darknet
	```
*  To train our object detector we can use the existing pre trained weights that are already trained on huge data sets. And we will download the pretrained weights which is previuosly trained on coco dataset from [here](https://pjreddie.com/media/files/darknet53.conv.74).  
Using the concept of transfer learning to train a custom model, using a basic trained model and use its learning to make learn a model for custom data.
* Changes in the custom config file:
	-   Uncomment Training and Comment Testing line of batch and subdivison. 
	-   Change line batch to batch=64
	-   Change line subdivisions to subdivisions=16
	-   Set network size width=416 height=416 or any value multiple of 32
	-   Change line max_batches to (classes*2000 but not less than the number of training images and not less than 6000), i.e. max_batches = 8000 as we have to train for 4 classes.
	-   Change [filters=32] to filters=(classes + 5)x3 in the  **3**  **[convolutional]**  before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.
	-   Change line classes=4 to number of objects in each of  **3 [yolo]**-layers.
![custom_cnf.png](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/src/custom_cng.png)
* Make changes in the makefile to enable OPENCV, GPU and CUDNN: 
	 ```python:
	!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
	!sed -i 's/GPU=0/GPU=1/' Makefile
	!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
	```
  Then run `!make` command to build darknet.
*   We will create a new file within a code or text editor called  classes.names  and this file will be exactly the same as the classes.txt which contain the classes label.
* Now we will create `laballed.data` which should contain information regarding the train data sets and split the data for train and test by calling the Python files creating-files-data-and-name and creating-train-and-test-txt-files.
For train.txt file we will create which hold the relative paths to all our training images. or we can say each row in the file should have the location of train dataset.

	```python: 
	from glob import glob
	imgs_list1 = glob("training_image/*jpg")
	imgs_list2 = glob("training_image/*jpeg")
	imgs_list = imgs_list1+imgs_list2
	file = open("training_image/train.txt",'w')
	file.write("\n".join(imgs_list))
	file.close()
	```
* Start Training:- As data is prepared we will move forward for training part. We will use below command to start training.
	```python:
	!darknet/darknet detector train training_image/labelled_data.data darknet/cfg/yolov3_custom.cfg custom_weight/darknet53.conv.74 -dont_show
	```
* To avoid being kicked off Colab VM we will use this simple hack for Auto-Click. Press `Ctrl + Shift + i` , Go to console and Paste the following code and press Enter.

```
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
ClickConnect()
```

Note: If in some case the training does not finish and get disconnected, We can restart the training from where we left off. We will use the weights that were saved last. Which will be saved in backup folder. The name of the last file will be **yolov3_custom_last.weights** . backup directory is the location where newly trained weights would be saved.

### Runnning Custom Object Detector
* We can perform detection with OpenCV DNN as it is a fast DNN implementation for CPU.
	* We will switch to testing by first uncommenting the testing lines for batch and subdivison in yolov3_custom.cfg file. And commenting the training lines.
	* And we will use yolov3_custom_last.weights file as a weigths for detection.
	* Now for object detection in an image we will use this
 [file](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/creating-files-data-and-name.py) and this [file](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/creating-train-and-test-txt-files.py) for object detection in video.

Below is an example of image used as input.
![input](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/test/vehicle.jpg)
Now as the Image proccessed the following object in the image will be detect and the output image will be:
![output](https://github.com/rohan300557/Vehicle-Detection-Yolo/blob/main/output/output.jpg)
	
