import io 
import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_hand3d(keypoints):
    # Define the connections between keypoints as tuples (start, end)
    bones = [
        ((0, 1), 'red'), ((1, 2), 'green'), ((2, 3), 'blue'), ((3, 4), 'purple'),
        ((0, 5), 'orange'), ((5, 6), 'pink'), ((6, 7), 'brown'), ((7, 8), 'cyan'),
        ((0, 9), 'yellow'), ((9, 10), 'magenta'), ((10, 11), 'lime'), ((11, 12), 'blueviolet'),
        ((0, 13), 'olive'), ((13, 14), 'teal'), ((14, 15), 'crimson'), ((15, 16), 'cornsilk'),
        ((0, 17), 'aqua'), ((17, 18), 'silver'), ((18, 19), 'maroon'), ((19, 20), 'fuchsia')
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the bones
    for bone, color in bones:
        start_point = keypoints[bone[0], :]
        end_point = keypoints[bone[1], :]

        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], color=color)
    
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], color='gray', s=15)

    # Set the aspect ratio to be equal
    max_range = np.array([keypoints[:,0].max()-keypoints[:,0].min(), 
                          keypoints[:,1].max()-keypoints[:,1].min(), 
                          keypoints[:,2].max()-keypoints[:,2].min()]).max() / 2.0

    mid_x = (keypoints[:,0].max()+keypoints[:,0].min()) * 0.5
    mid_y = (keypoints[:,1].max()+keypoints[:,1].min()) * 0.5
    mid_z = (keypoints[:,2].max()+keypoints[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def visualize_hand(joints, img):
# Define the connections between joints for drawing lines and their corresponding colors
    connections = [
        ((0, 1), 'red'), ((1, 2), 'green'), ((2, 3), 'blue'), ((3, 4), 'purple'),
        ((0, 5), 'orange'), ((5, 6), 'pink'), ((6, 7), 'brown'), ((7, 8), 'cyan'),
        ((0, 9), 'yellow'), ((9, 10), 'magenta'), ((10, 11), 'lime'), ((11, 12), 'indigo'),
        ((0, 13), 'olive'), ((13, 14), 'teal'), ((14, 15), 'navy'), ((15, 16), 'gray'),
        ((0, 17), 'lavender'), ((17, 18), 'silver'), ((18, 19), 'maroon'), ((19, 20), 'fuchsia')
    ]
    H, W, C = img.shape
    
    # Create a figure and axis
    plt.figure()
    ax = plt.gca()
    # Plot joints as points
    ax.imshow(img)
    ax.scatter(joints[:, 0], joints[:, 1], color='white', s=15)
    # Plot lines connecting joints with different colors for each bone
    for connection, color in connections:
        joint1 = joints[connection[0]]
        joint2 = joints[connection[1]]
        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], color=color)

    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.grid(False)
    ax.set_axis_off()
    ax.invert_yaxis()
    plt.subplots_adjust(wspace=0.01)
    plt.show()
    

def draw_hand_skeleton(joints, image_size, thickness=5):
    # Create a blank white image
    image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

    # Define the connections between joints
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    # Draw lines connecting joints
    for connection in connections:
        joint1 = joints[connection[0]].astype("int")
        joint2 = joints[connection[1]].astype("int")
        cv2.line(image, tuple(joint1), tuple(joint2), color=1, thickness=thickness)

    return image


def draw_hand(joints, img):
    # Define the connections between joints for drawing lines and their corresponding colors
    connections = [
        ((0, 1), 'red'), ((1, 2), 'green'), ((2, 3), 'blue'), ((3, 4), 'purple'),
        ((0, 5), 'orange'), ((5, 6), 'pink'), ((6, 7), 'brown'), ((7, 8), 'cyan'),
        ((0, 9), 'yellow'), ((9, 10), 'magenta'), ((10, 11), 'lime'), ((11, 12), 'indigo'),
        ((0, 13), 'olive'), ((13, 14), 'teal'), ((14, 15), 'navy'), ((15, 16), 'gray'),
        ((0, 17), 'lavender'), ((17, 18), 'silver'), ((18, 19), 'maroon'), ((19, 20), 'fuchsia')
    ]
    H, W, C = img.shape
    
    # Create a figure and axis with the same size as the input image
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    # Plot joints as points
    ax.imshow(img)
    ax.scatter(joints[:, 0], joints[:, 1], color='white', s=15)
    # Plot lines connecting joints with different colors for each bone
    for connection, color in connections:
        joint1 = joints[connection[0]]
        joint2 = joints[connection[1]]
        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], color=color)

    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.grid(False)
    ax.set_axis_off()
    ax.invert_yaxis()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory

    # Load the image from the buffer into a PIL image and then into a numpy array
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    
    return img_arr[..., :3]


def keypoint_heatmap(pts, size, var=1.0):
    H, W = size
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack((xv, yv), axis=-1)
    
    # Expanding dims for broadcasting subtraction between pts and every grid position
    modes_exp = np.expand_dims(np.expand_dims(pts, axis=1), axis=1)
    
    # Calculating squared difference
    diff = grid - modes_exp
    normal = np.exp(-np.sum(diff**2, axis=-1) / (2 * var)) / (
        2.0 * np.pi * var
    )
    return normal


def check_keypoints_validity(keypoints, image_size):
    H, W = image_size
    # Check if x coordinates are valid: 0 < x < W
    valid_x = (keypoints[:, 0] > 0) & (keypoints[:, 0] < W)
    
    # Check if y coordinates are valid: 0 < y < H
    valid_y = (keypoints[:, 1] > 0) & (keypoints[:, 1] < H)
    
    # Combine the validity checks for both x and y
    valid_keypoints = valid_x & valid_y
    
    # Convert boolean array to integer (1 for True, 0 for False)
    return valid_keypoints.astype(int)


def find_bounding_box(mask, margin=30):
    """Find the bounding box of a binary mask. Return None if the mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():  # Mask is empty
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    return xmin, ymin, xmax, ymax


def adjust_box_to_image(xmin, ymin, xmax, ymax, image_width, image_height):
    """Adjust the bounding box to fit within the image boundaries."""
    box_width = xmax - xmin
    box_height = ymax - ymin
    # Determine the side length of the square (the larger of the two dimensions)
    side_length = max(box_width, box_height)
    
    # Adjust to maintain a square by expanding or contracting sides
    xmin = max(0, xmin - (side_length - box_width) // 2)
    xmax = xmin + side_length
    ymin = max(0, ymin - (side_length - box_height) // 2)
    ymax = ymin + side_length
    
    # Ensure the box is still within the image boundaries after adjustments
    if xmax > image_width:
        shift = xmax - image_width
        xmin -= shift
        xmax -= shift
    if ymax > image_height:
        shift = ymax - image_height
        ymin -= shift
        ymax -= shift
    
    # After shifting, double-check if any side is out-of-bounds and adjust if necessary
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_width, xmax)
    ymax = min(image_height, ymax)
    
    # It's possible the adjustments made the box not square (due to boundary constraints),
    # so we might need to slightly adjust the size to keep it as square as possible
    # This could involve a final adjustment based on the specific requirements,
    # like reducing the side length to fit or deciding which dimension to prioritize.

    return xmin, ymin, xmax, ymax


def scale_keypoint(keypoint, original_size, target_size):
    """Scale a keypoint based on the resizing of the image."""
    keypoint_copy = keypoint.copy()
    keypoint_copy[:, 0] *= target_size[0] / original_size[0]
    keypoint_copy[:, 1] *= target_size[1] / original_size[1]
    return keypoint_copy


def crop_and_adjust_image_and_annotations(image, hand_mask, obj_mask, hand_pose, intrinsics, target_size=(512, 512)):
    # Find bounding boxes for each mask, handling potentially empty masks
    xmin, ymin, xmax, ymax = find_bounding_box(hand_mask) if np.any(hand_mask) else None

    # Adjust bounding box to fit within the image and be square
    xmin, ymin, xmax, ymax = adjust_box_to_image(xmin, ymin, xmax, ymax, image.shape[1], image.shape[0])
    
    # Crop the image and mask
    # masked_hand_image = (image * np.maximum(hand_mask, obj_mask)[..., None].astype(float)).astype(np.uint8)
    cropped_hand_image = image[ymin:ymax, xmin:xmax]
    cropped_hand_mask = hand_mask[ymin:ymax, xmin:xmax].astype(np.uint8)
    cropped_obj_mask = obj_mask[ymin:ymax, xmin:xmax].astype(np.uint8)  

    # Resize the image
    resized_image = resize(cropped_hand_image, target_size, anti_aliasing=True)
    resized_hand_mask = cv2.resize(cropped_hand_mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    resized_obj_mask = cv2.resize(cropped_obj_mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    
    # adjust and scale 2d keypoints
    for hand_type, kps2d in hand_pose.items():
        kps2d[:, 0] -= xmin
        kps2d[:, 1] -= ymin
        hand_pose[hand_type] = scale_keypoint(kps2d, (xmax - xmin, ymax - ymin), target_size)
        
    # adjust instrinsics
    resized_intrinsics= np.array(intrinsics, copy=True)
    resized_intrinsics[0, 2] -= xmin
    resized_intrinsics[1, 2] -= ymin
    resized_intrinsics[0, :] *= target_size[0] / (xmax - xmin)
    resized_intrinsics[1, :] *= target_size[1] / (ymax - ymin)
    
    return (resized_image, resized_hand_mask, resized_obj_mask, hand_pose, resized_intrinsics)
