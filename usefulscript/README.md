## ONNX
###  Run Model
Start run your model with following command:

`$pytest -v -s test_samples_onnx.py -k test_onnx["model_name"] --model_dir="local_model_directory"`

* Required command argument:
	- `"model_name"`  model name. It's can not be empty. The options are following:
        - `"inception-v1-9"`
        - `"resnet18-v1-7"`
        - `"resnet50-caffe2-v1-9"`
        - `"squeezenet1.1-7"`
        - `"mobilenetv2-7"`
        - `"yolov2-coco-9"`
        - `"squeezenet1.0-9"`
        - `"vgg16-bn-7"`
        - `"densenet-9"`
        - `"densenet201_224_224_without_trainning"`
        - `"inceptionv3_224_224"`
        - `"mobilenet_224_224"`
        - `"segnet_224_224_without_trainning"`
        - `"senet-50_224_224"`
        - `"yolov3-416_416"`
        - `"yolov3_tiny_1088_1920"`
        - `"yolov3_tiny_416_416"`
        - `"yolov3_tiny_544_960"`
        - `"yolov3_tiny_608_608"`
        - `"yolov3_tiny_736_1280"`
* Optional command arguments:
	- `--max_batch`
		Config the max number of batch. Default value is 1. 
	- `--exec_batch`
		The number of batch in execution time you want. Default value is 1. 
	- `--weight_share`
		Specify if the weight is share between clusters. The options are following:
        - `0` weight is only used in cluster0. Default value.
        - `1` weight is only used in cluster1.
        - `2` weight is only used in cluster2.
        - `3` weight is only used in cluster3.
        - `01` weight is shared between cluster0 and cluster1.
        - `23` weight is shared between cluster2 and cluster3.
        - `0123` weight is shared between all 4 clusters.
	- `--out_node`
		Specify the output node. Open the onnx file with Netron, then chose the output node you want, it could be last or inner node in model. Default value is the last node in model. 
    - `--model_dir`
        Specify the model directory, can not be empty.

	**Example**: Runing resnet18-v1-7 model with max batch size 1, execution batch size 1, in single cluster0.
    
    `$pytest -v -s test_samples_onnx.py -k test_onnx["resnet18-v1-7"] --max_batch=1 --exec_batch=1 --weight_share=0 --model_dir="local_model_directory"`
    
	
### Serialize

Serialize a model to a file with following command:

`$pytest -v -s test_samples_onnx.py -k test_onnx_serialize["model_name"] --file_name="onnx.slz" --model_dir="local_model_directory"`

* Required command arguments:
	- `--file_name` the name of serialize file.
    - `--model_dir` local model directory, can not be empty.
    - `--max_batch`
            Config the max number of batch. Default value is 1.
    - `"model_name"` the model to be serialized.
        - `"inception-v1-9"`
        - `"resnet18-v1-7"`
        - `"resnet50-caffe2-v1-9"`
        - `"squeezenet1.1-7"`
        - `"mobilenetv2-7"`
        - `"yolov2-coco-9"`
        - `"squeezenet1.0-9"`
        - `"vgg16-bn-7"`
        - `"densenet-9"`
        - `"densenet201_224_224_without_trainning"` 
        - `"inceptionv3_224_224"`
        - `"mobilenet_224_224"`
        - `"segnet_224_224_without_trainning"`
        - `"senet-50_224_224"`
        - `"yolov3-416_416"`
        - `"yolov3_tiny_1088_1920"`
        - `"yolov3_tiny_416_416"`
        - `"yolov3_tiny_544_960"`
        - `"yolov3_tiny_608_608"`
        - `"yolov3_tiny_736_1280"`

### Deserialize

Deserialize a file with following command:

`$pytest -v -s test_samples_onnx.py -k test_onnx_deserialize --file_name="onnx.slz"`

* Required command argument:
	- `--file_name` the name of serialized file. **Must** same with serialized model file.
   - `--exec_batch`
                The number of batch in execution time you want. Default value is 1.
