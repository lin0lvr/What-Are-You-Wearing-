# What-Are-You-Wearing

Clothing Classification Model

This model is used to classify what people are wearing to help accurately enforce dress codes. It is trained on an imagenet Resnet-18 model using transfer learning.

![image](https://github.com/lin0lvr/What-Are-You-Wearing-/assets/140644065/5f46ba8b-5935-40b3-8c5a-dde27be4bd7c)

![image](https://github.com/lin0lvr/What-Are-You-Wearing-/assets/140644065/3eae6ae7-a7c0-4fcb-9c68-994f7f65246c)

![image](https://github.com/lin0lvr/What-Are-You-Wearing-/assets/140644065/9b70b3bf-1520-4453-aead-5e4fb85ec9f3)

![image](https://github.com/lin0lvr/What-Are-You-Wearing-/assets/140644065/6a2834f7-4be5-4e9a-8947-76ccff70f6d9)







## The Algorithm
This project uses a resnet18 model that was retrained with 4 different data sets. Each data set contained a different kind of clothing. The algorithm uses imagenet to identify the article of clothing. I ran this model with 500 epochs so the model is mostly accurate but can still make mistakes. 

## Running this project
Download the jetson-inference container from github to a jetson-nano: https://github.com/dusty-nv/jetson-inference

Change directories into jetson-inference/python/training/classification/data

Create a directory called "clothes"

Download the dataset at https://github.com/alexeygrigorev/clothing-dataset-small.git 

delete dress, longsleeve, outwear, shorts, skirt, t-shirt from the test, train, and val folders. 

Make a .txt file called "labels.txt" in the "clothes" directory and write down the following on a separate line (in exactly that order)

hat, pants, shoes, shirt

Make a new directory called "clothes" in jetson-inference/python/training/classification/models
Execution

Change directories to jetson-inference

Type and run ./docker/run.sh in the terminal

Then change directories to jetson-inference/python/training/classification

Now train the model by running this command in the terminal python3 train.py --model-dir=models/clothes data/clothes

Note: Depending on how many epochs you run this might take a while 

Once done, export the model by running this script
python3 onnx_export.py --model-dir=models/clothes

If you are in the docker container, exit it by pressing ctrl+d or typing 'exit'

Change directories to jetson-inference/python/training/classification

In the terminal enter in NET=models/clothes and DATASET=data/clothes

Then enter in the place of '...' hat, pants, shirt or shoes

imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/.../malignant8.png output.jpg

If you look in your classification directory there will be a file called output.jpg (or whatever you named the output to be)

Display the file to see what the article of clothing is 

View a video explanation here
