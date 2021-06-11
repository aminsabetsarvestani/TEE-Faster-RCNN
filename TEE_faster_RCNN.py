from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from typing import Dict
from torch import nn
import warnings
from torchvision.ops import misc as misc_nn_ops
from torchvision.models import resnet
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
import torch
from scipy.stats import entropy
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead,FastRCNNPredictor






class TEE_GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(TEE_GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections,features):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections,features

    def forward(self, images, TEEM_features=None, targets=None):
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features,TEEMs_features = self.backbone(images.tensors,TEEM_features)
        if TEEMs_features == None:
            return None,None,None
        if not features:
            return [0]
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections,TEEMs_features
        else:
            return self.eager_outputs(losses, detections,TEEMs_features)


class TEE_IntermediateLayerGetter(nn.ModuleDict):
    
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str],Early_Exits = []) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(TEE_IntermediateLayerGetter, self).__init__(layers)
        # load pretrained TEEM modules
        self.EarlyExit1 = Early_Exits[0]
        self.EarlyExit2 = Early_Exits[1]
        self.EarlyExit3 = Early_Exits[2]
        self.EarlyExit4 = Early_Exits[3]
        self.return_layers = orig_return_layers
        self.state = True
        
    def forward(self, x,TEEM_features=None):
        out = OrderedDict()
        self.state = True
        for name, module in self.items():

               if name in ['EarlyExit1','EarlyExit2','EarlyExit3','EarlyExit4']:
                   pass
               else:
                   x = module(x)
                   
                   if name == 'layer1' and TEEM_features is not None and self.state:
                        exit1 = self.EarlyExit1(TEEM_features[0],x)
                        entrop = entropy(exit1.detach().cpu().numpy()[0],base=2)
                        _, preds = torch.max(exit1.data, 1)
                        if preds == 1 and entrop<0.8:
                            return None
                        elif preds == 0 and entrop<0.8:
                            self.state = False

                   if name == 'layer2'and TEEM_features is not None and self.state:
                        exit2 = self.EarlyExit2(TEEM_features[1],x)
                        entrop = entropy(exit2.detach().cpu().numpy()[0],base=2)
                        _, preds = torch.max(exit2.data, 1)
                        if preds == 1 and entrop<0.8:
                            return None
                        elif preds == 0 and entrop<0.8:
                            self.state = False
                           
                   if name == 'layer3'and TEEM_features is not None and self.state:
                        exit3 = self.EarlyExit3(TEEM_features[2],x)
                        entrop = entropy(exit3.detach().cpu().numpy()[0],base=2)
                        _, preds = torch.max(exit3.data, 1)
                        if preds == 1 and entrop<0.8:
                            return None
                        elif preds == 0 and entrop<0.90:
                            self.state = False

                    
                   if name== 'layer4' and TEEM_features is not None and self.state:
                        exit4 = self.EarlyExit4(TEEM_features[3],x)
                        entrop = entropy(exit4.detach().cpu().numpy()[0],base=2)
                        _, preds = torch.max(exit4.data, 1)
                        if preds == 1 and entrop<0.97:
                            return None
                        elif preds == 0 and entrop<0.97:
                            self.state = False
                            

                   if name in self.return_layers:
                      out_name = self.return_layers[name]
                      out[out_name] = x
        return out


class TEE_BackboneWithFPN(nn.Module):
    
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, Early_Exits,extra_blocks=None):
        super(TEE_BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = TEE_IntermediateLayerGetter(backbone, return_layers=return_layers, Early_Exits = Early_Exits)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x,TEEM_features=None):
        y = self.body(x,TEEM_features)
        if y==None:
            return None,None
        x = self.fpn(y)
        return x,y


def TEE_resnet_fpn_backbone(
    backbone_name,
    pretrained,
    Early_Exits,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return TEE_BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, Early_Exits, extra_blocks=extra_blocks)

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
}



class TEE_FasterRCNN(TEE_GeneralizedRCNN):


    def __init__(self, backbone ,num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(TEE_FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)



def TEE_fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, Early_Exits=[], trainable_backbone_layers = None, **kwargs):
  
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False


    backbone = TEE_resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers = trainable_backbone_layers, Early_Exits = Early_Exits)
    model = TEE_FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)
        overwrite_eps(model, 0.0)
    return model