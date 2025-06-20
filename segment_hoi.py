import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    

def merge_bounding_boxes(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    
    xmin_merged = min(xmin1, xmin2)
    ymin_merged = min(ymin1, ymin2)
    xmax_merged = max(xmax1, xmax2)
    ymax_merged = max(ymax1, ymax2)
    
    return np.array([xmin_merged, ymin_merged, xmax_merged, ymax_merged])


def init_sam(
    device="cuda",
    ckpt_path='/users/kchen157/scratch/weights/SAM/sam_vit_h_4b8939.pth'
    ):
    sam = sam_model_registry['vit_h'](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def segment_hand_and_object(
    predictor,
    image, 
    hand_kpts, 
    hand_mask=None,
    box_shift_ratio = 0.3,
    box_size_factor = 2.,
    area_threshold = 0.2,
    overlap_threshold = 200):
    # Find bounding box for HOI
    input_box = {}
    for hand_type in ['right', 'left']:
        if hand_type not in hand_kpts:
            continue
        input_box[hand_type] = np.stack([hand_kpts[hand_type].min(axis=0), hand_kpts[hand_type].max(axis=0)])
        box_trans = input_box[hand_type][0] * box_shift_ratio + input_box[hand_type][1] * (1 - box_shift_ratio)
        input_box[hand_type] = ((input_box[hand_type] - box_trans) * box_size_factor + box_trans).reshape(-1)

    if len(input_box) == 2:
        input_box = merge_bounding_boxes(input_box['right'], input_box['left'])
        input_point = np.array([hand_kpts['right'][0], hand_kpts['left'][0]])
        input_label = np.array([1, 1])
    elif 'right' in input_box:
        input_box = input_box['right']
        input_point = np.array([hand_kpts['right'][0]])
        input_label = np.array([1])
    elif 'left' in input_box:
        input_box = input_box['left']
        input_point = np.array([hand_kpts['left'][0]])
        input_label = np.array([1])

    box_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])

    # segment hand using the wrist point
    predictor.set_image(image)
    if hand_mask is None:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        hand_mask = masks[0]

    # segment object in hand 
    input_label = np.zeros_like(input_label)
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=False,
    )
    object_mask = masks[0]

    if  (masks[0].astype(int) * hand_mask).sum() > overlap_threshold:
        # print('False positive: The mask overlaps the hand.')
        object_mask = np.zeros_like(object_mask)
    elif object_mask.astype(int).sum() / box_area > area_threshold:
        # print('False positive: The area is very big, probably the background')
        object_mask = np.zeros_like(object_mask)

    return object_mask, hand_mask