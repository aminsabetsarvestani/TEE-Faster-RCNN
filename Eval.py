
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frames_path',         type=str,                default='/video_frames/data/', 				       help='dirctory for test video frames, anotation file need to be in YOLO format')
parser.add_argument('--Eval_mode',           type=str,                default='TEE',                 					   help='The evaluation mode TEE for TEEM_faster_RCNN and base for original faster_RCNN ')
parser.add_argument('--num_workers',         type=int,                default=0,                                           help='num_workers in dataloader')
parser.add_argument('--iou_threshs',         type=list,               default=[0.3,0.35,0.40,0.45,0.5],                    help='List of IoU threshols to measure mAP')
parser.add_argument('--dataset_path',        type=str,                default='/video_frames/',                            help='directory of video dataset')
parser.add_argument('--TEEMs_path',          type=str,                default='/TrainedTEEMs/',                            help='directory of pre_trained TEEMs')
parser.add_argument('--confidence',          type=float,                default=0.6,                                       help='confidence of object detection for measureing mAP')
args = parser.parse_args()

import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as func
from TEE_faster_RCNN import TEE_fasterrcnn_resnet50_fpn
from TEEM_module import TEEM
import torchvision.transforms as T
from fiftyone import ViewField as F
import fiftyone as fo
import fiftyone.zoo as foz
import glob

frames_path = os.getcwd()+ args.frames_path
dataset_path = os.getcwd()+ args.dataset_path
TEEM_path = os.getcwd() + args.TEEMs_path

list_of_frames = []
for filename in glob.glob(frames_path+'/*.jpg'): #assuming gif
    list_of_frames.append(filename)
list_of_frames.sort()

# Run the model on GPU if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Define TEEMs
TEEM1 = TEEM(256,128)
TEEM2 = TEEM(512,256)
TEEM3 = TEEM(1024,128)
TEEM4 = TEEM(2048,256)
#Load TEEM parameters
TEEM1.load_state_dict(torch.load(TEEM_path+"TEEM1.pth"))
TEEM2.load_state_dict(torch.load(TEEM_path+"TEEM2.pth"))
TEEM3.load_state_dict(torch.load(TEEM_path+"TEEM3.pth"))
TEEM4.load_state_dict(torch.load(TEEM_path+"TEEM4.pth"))
#Load TEEM to Eval mode
TEEM1.eval()
TEEM2.eval()
TEEM3.eval()
TEEM4.eval()
#Transfer TEEMs to the device
TEEM1.to(device)
TEEM2.to(device)
TEEM3.to(device)
TEEM4.to(device)

# Load pre-trained TEE-Faster-RCNN and Faster-RCNN models
TEE_model = TEE_fasterrcnn_resnet50_fpn(pretrained=True, min_size=224,Early_Exits= [TEEM1, TEEM2, TEEM3, TEEM4])
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print("Model ready")

# Get class list labels from Coco dataste Faster RCNN is trained on COCO
dataset_base = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="evaluate-detections-tutorial",
)
classes_base= dataset_base.default_classes
# Video Dataset
try:
    dataset = fo.Dataset.from_dir(dataset_path, fo.types.YOLODataset,shuffle=False)
except:
    dataset = fo.load_dataset('video_testset')

dataset.persistent = False
classes = dataset.default_classes



def faster_rcnn_evaluation():
	
	model.to(device)
	model.eval()
	latest_pred = None
	with fo.ProgressBar() as pb:
		for frame in list_of_frames:
			sample = dataset[frame]
			# Load image
			image = Image.open(sample.filepath)
			resize = T.Resize(size=(224, 224))
			image = resize(image)
			image = func.to_tensor(image).to(device)
			c, h, w = image.shape
			# Perform inference
			preds = model([image])[0]			
			labels = preds["labels"].cpu().detach().numpy()
			scores = preds["scores"].cpu().detach().numpy()
			boxes = preds["boxes"].cpu().detach().numpy()

			detections = []
			for label, score, box in zip(labels, scores, boxes):
				x1, y1, x2, y2 = box
				rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

				if classes_base[label] in classes:
					detections.append(
					fo.Detection(
					label=classes_base[label],
					bounding_box=rel_box,
					confidence=score
					)
					)

			# Save predictions to dataset
			sample["faster_rcnn"] = fo.Detections(detections=detections)
			sample.save()

	# Only contains detections with confidence >= 0.70
	high_conf_view = dataset.filter_labels("faster_rcnn", F("confidence") > args.confidence)
	results = high_conf_view.evaluate_detections(
	"faster_rcnn",
	gt_field="ground_truth",
	eval_key="eval",
	compute_mAP=True,
	classes = classes,
	iou_threshs = args.iou_threshs
	)
	print('mAP: ',results.mAP())
	

def TEE_faster_rcnn_evaluation():
	TEE_model.to(device)
	TEE_model.eval()
	#Initial features into TEEMs
	TEEM1 = torch.zeros(1,256,56,56).to(device)
	TEEM2 = torch.zeros(1,512,28,28).to(device)
	TEEM3 = torch.zeros(1,1024,14,14).to(device)
	TEEM4 = torch.zeros(1,2048,7,7).to(device)

	with fo.ProgressBar() as pb:
		for frame in list_of_frames:

				sample = dataset[frame]
				# Load image
				image = Image.open(sample.filepath)
				resize = T.Resize(size=(224, 224))
				image = resize(image)
				image = func.to_tensor(image).to(device)
				c, h, w = image.shape
				output = TEE_model([image],[TEEM1,TEEM2,TEEM3,TEEM4])
				
				if output[1]!=None:
					latest_preds = output[0][0]
					TEEM1 = output[1]['0']
					TEEM2 = output[1]['1']
					TEEM3 = output[1]['2']
					TEEM4 = output[1]['3']
				
				labels = latest_preds["labels"].cpu().detach().numpy()
				scores = latest_preds["scores"].cpu().detach().numpy()
				boxes =  latest_preds["boxes"].cpu().detach().numpy()
				
				detections = []
				for label, score, box in zip(labels, scores, boxes):
					x1, y1, x2, y2 = box
					rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

					if classes_base[label] in classes:
						detections.append(
							fo.Detection(
										label=classes_base[label],
										bounding_box=rel_box,
										confidence=score
										)
										)
				sample["faster_rcnn"] = fo.Detections(detections=detections)
				sample.save()
				
	# Only contains detections with confidence >= 0.75
	high_conf_view = dataset.filter_labels("faster_rcnn", F("confidence") > args.confidence)
	results = high_conf_view.evaluate_detections(
		"faster_rcnn",
		gt_field="ground_truth",
		eval_key="eval",
		compute_mAP=True,
		classes = classes,
		iou_threshs = args.iou_threshs
		)
	print('mAP: ',results.mAP())


if __name__ == "__main__":

	if args.Eval_mode == 'TEE':
		TEE_faster_rcnn_evaluation()
	else:
		faster_rcnn_evaluation()

