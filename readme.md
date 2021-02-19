# Readme 

Pretrained Model:
https://drive.google.com/drive/folders/1WLS7RH50THoDb0BQwmKO5McB5Ots6C2Q?usp=sharing

* Steps 
	* Step 1:  For feature extraction we need to use CNN, We have to resize the MSCOCO image dataset to the fixed size of 224x224 (depends on CNN used). Here we are using orthogonalty enforced RESNET34 and it requires input in size 224x224. Run the resize.py then resized images will be stored in image/train2014_resized/ and image/val2014_resized/ directory.
	* Step 2: Before training the model, we need to preprocess the MSCOCO caption dataset. To generate caption dataset and extract the image feature.
	* Step 3: To train the image captioning model, run train_hindi_corrected.py .
	
## Test

	* Run test.py to test the model.
