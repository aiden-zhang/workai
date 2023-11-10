from dlnneEngine import DlnneEngine
import numpy as np
import cv2

class DetObj():
    def __init__(self, box, confidence, id):
        self.box = box
        self.conf = confidence
        self.id = id
        self.mark = 0
    
    def __eq__(self, other):
        return self.conf == other.conf
    
    def __le__(self, other):
        return self.conf < other.conf
    
    def __gt__(self, other):
        return self.conf > other.conf


class YoloV5(DlnneEngine):
    def __init__(self, model_path, inputs_shape_dict, output_op_list, max_batch_size, seq_len, config_key):
        super(YoloV5, self).__init__(model_path, 
        inputs_shape_dict, output_op_list, max_batch_size, seq_len, config_key)

    def preprocess(self, cv_image):
        return cv2.dnn.blobFromImage(image=cv_image, scalefactor= 1/255.0, size=(640, 640), swapRB=True, crop=False).astype(np.float32)

    def iou(self, box1, box2):
        f32Eps = 1e-15
        left = max(box1[0], box2[0])
        top = max(box1[1], box2[1])
        right = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])

        inter_area = max(0, bottom - top + 1) * max(0, right - left + 1)
        # if abs(inter_area) < 1e-6:
        #     return 0
        
        area1 = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)
        area2 = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)
        return  inter_area / (area1 + area2 - inter_area + f32Eps)

    
    def nms(self, detObjs, threshold):
        sorted_objs= sorted(detObjs)
        for i in range(0, len(sorted_objs)):
            for j in range(i + 1, len(sorted_objs)):
                if sorted_objs[i].id != sorted_objs[j].id:
                    continue
                else:
                    iou = self.iou(sorted_objs[i].box, sorted_objs[j].box)
                    if iou > threshold:
                        sorted_objs[j].mark = -1
        return sorted_objs

    def postprocess(self, cv_image, outs):
        h, w, c = cv_image.shape
        class_ids = []
        confidences = []
        boxes = []
        image_height, image_width = h, w
        x_factor = image_width / 640
        y_factor =  image_height / 640
        det_objs = []

        for out in outs:
            for detection in out[0]:
                x, y, w, h, conf = detection[0], detection[1], detection[2], detection[3], detection[4]

                if conf >= 0.5:
                    classes_scores = detection[5: ]
                    class_id = np.argmax(classes_scores)
                    #  Continue if the class score is above threshold.
                    if (classes_scores[class_id] > 0.4):
                        confidences.append(conf)
                        class_ids.append(class_id)
                        cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, left + width - 1, top + height - 1])
                        boxes.append(box)
                        detObj = DetObj(box, conf, class_id)
                        det_objs.append(detObj)
        
        sorted_objs = self.nms(det_objs, 0.5)
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        for i in range(0, len(sorted_objs)):
            if sorted_objs[i].mark != -1:   
                cv2.rectangle(cv_image, (sorted_objs[i].box[0], sorted_objs[i].box[1]), (sorted_objs[i].box[2], sorted_objs[i].box[3]), (0, 0, 255), 3)
        return cv_image

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:,:, 0] = x[:,:, 0] - x[:,:, 2] / 2  # top left x
    y[:,:, 1] = x[:,:, 1] - x[:,:, 3] / 2  # top left y
    y[:,:, 2] = x[:,:, 0] + x[:,:, 2] / 2  # bottom right x
    y[:,:, 3] = x[:,:, 1] + x[:,:, 3] / 2  # bottom right y
    return y

def get_box(input_feature_np):
    batch_size = input_feature_np.shape[0]
    boxes_feat = input_feature_np[:,:,0:4]
    boxes_xyxy = xywh2xyxy(boxes_feat)
    boxes_np = np.reshape(boxes_xyxy,[batch_size,1,-1,4]) #reshpe to [batch_size,1,num_boxes,4]
    return boxes_np

def get_score(input_feature_np):
    confidence_feat = input_feature_np[:,:,4:5]
    classes_feat = input_feature_np[:,:,5:]
    score = confidence_feat*classes_feat 
    score_np = np.transpose(score,[0,2,1]) #transpose to [bacth_size,classes,num_boxes]
    return score_np

def unitest():
    yolov5 = YoloV5(model_path='./nms-out4.onnx', inputs_shape_dict={'input_box':[1,1,25200,4], 'input_class':[1,80,25200]}, 
        output_op_list = ['nmsed_classes', 'nmsed_boxes','nmsed_scores','num_detections'],
        # output_op_list = ['num_detections','nmsed_boxes','nmsed_scores'],
        # output_op_list = ['num_detections'],
        max_batch_size = 1,
        seq_len = 0,
        config_key = '0'
    )
    input_data = np.load('input.npy') # 1 25200 85

    input_box = get_box(input_data)
    input_class = get_score(input_data)

    output_data = yolov5.inference({'input_class':input_class, 'input_box':input_box}, 1)
    for _data in output_data:        
        print(_data.shape)
    # np.savetxt('nmsed_classes.txt', output_data[0])
    # np.savetxt('nmsed_boxes.txt', output_data[1])
    # np.savetxt('nmsed_scores.txt', output_data[2])
    # np.savetxt('num_detections.txt', output_data[3])
    print(output_data[3][0])

def test():
    yolov5 = YoloV5(model_path='./yolov5s_nms.onnx', inputs_shape_dict={'images':[1, 3, 640, 640]}, 
        output_op_list = ['nmsed_classes', 'nmsed_boxes','nmsed_scores','num_detections'],
        # output_op_list = ['bboxes', 'scores'],
        # output_op_list = ['output'],
        max_batch_size = 1,
        seq_len = 0,
        config_key = '0'
    ) 
    image = cv2.imread('test.jpg') 

    input_img = yolov5.preprocess(image)

    output_data = yolov5.inference({'images':input_img}, 1)
    for _data in output_data:        
        print(_data.shape)
    # np.savetxt('nmsed_classes.txt', output_data[0])
    # np.savetxt('nmsed_boxes.txt', output_data[1])
    # np.savetxt('nmsed_scores.txt', output_data[2])
    # np.savetxt('num_detections.txt', output_data[3])
    print(output_data[3][0])

if __name__ == '__main__':
    unitest()
    # test()


