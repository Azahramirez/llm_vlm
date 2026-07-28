import cv2
import json

# Define the bounding box data generated previously
bbox_data = [
    # Table
    {"bbox_2d": [85, 288, 586, 580], "label": "table"},

    # Center forklift
    {"bbox_2d": [606, 335, 737, 522], "label": "forklift"},

    # Right forklift
    {"bbox_2d": [928, 112, 1278, 574], "label": "forklift"},

    # Crates on table (left to right)
    {"bbox_2d": [92, 350, 157, 456], "label": "crate"},
    {"bbox_2d": [203, 370, 274, 453], "label": "crate"},
    {"bbox_2d": [277, 371, 403, 453], "label": "crate"},
    {"bbox_2d": [439, 383, 508, 453], "label": "crate"},
    {"bbox_2d": [110, 309, 165, 349], "label": "crate"},
    {"bbox_2d": [462, 309, 526, 349], "label": "crate"},
    {"bbox_2d": [185, 309, 270, 348], "label": "crate"}
]

bbox_data= [
        {"bbox_2d": [697, 208, 945, 573], "label": "yellow forklift"},
        {"bbox_2d": [487, 319, 550, 523], "label": "yellow forklift"},
        {"bbox_2d": [82, 316, 350, 587], "label": "table"},
        {"bbox_2d": [92, 377, 162, 487], "label": "crate"},
        {"bbox_2d": [132, 412, 207, 487], "label": "crate"},
        {"bbox_2d": [172, 417, 245, 487], "label": "crate"},
        {"bbox_2d": [250, 421, 309, 487], "label": "crate"},
        {"bbox_2d": [716, 0, 999, 153], "label": "text"}
]

bbox_data= [
       {"bbox_2d": [714,86,945,563], "label": "yellow forklift"},
       {"bbox_2d": [482,315,551,520], "label": "yellow forklift"},
        {"bbox_2d": [82,312,350,579], "label": "table"},
        
    
]

bbox_data= [
       {"bbox_2d": [918, 74, 1230, 416], "label": "yellow forklift"},
       {"bbox_2d": [625, 231, 712, 384], "label": "yellow forklift"},
        {"bbox_2d": [82, 312, 350, 586], "label": "table"},
        
    
]
# Load the image
image = cv2.imread('my_image.jpg') # Replace with your image path

# The coordinates are likely based on the 1280x720 resolution shown in the overlay.
# We resize the image to ensure boxes align correctly.
image_resized = cv2.resize(image, (1280, 720))
#image_resized=image

# Draw bounding boxes
for obj in bbox_data:
    x_min, y_min, x_max, y_max = obj['bbox_2d']
    label = obj['label']
    
    # Define color (Green: BGR 0, 255, 0)
    color = (0, 255, 0)
    
    # Draw rectangle
    cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw label text
    # Position: just above the top-left corner of the box
    cv2.putText(image_resized, label, (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save or display the result
#cv2.imwrite('output_with_boxes.jpg', image_resized)
cv2.imshow('Result', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()