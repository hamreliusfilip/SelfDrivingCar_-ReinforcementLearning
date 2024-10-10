import numpy as np
import cv2

WIDTH, HEIGHT = 800, 600

## BARRIÄRER ##
image = cv2.imread('new_track.png')
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 100)
#cv2.imshow('edges', edges)

# Extract contours with minimum length to prevent other lines except for the contours of the track to be selected
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
min_contour_length = 150
long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

# Outer and inner contour of the track
epsilon_outer = 0.001 * cv2.arcLength(long_contours[0], True)
approx_contours_outer = cv2.approxPolyDP(long_contours[0], epsilon_outer, True)
approx_contours_outer = np.squeeze(approx_contours_outer)

epsilon_inner = 0.001 * cv2.arcLength(long_contours[-1], True)
approx_contours_inner = cv2.approxPolyDP(long_contours[-1], epsilon_inner, True)
approx_contours_inner = np.squeeze(approx_contours_inner)

BARRIERS = [approx_contours_outer, approx_contours_inner]

## CHECKPOINTS ## X1, Y1, X2, Y2
CHECKPOINTS = np.array([[ 381, 450, 381, 540],
                        [ 267, 450, 267, 540],
                        [ 153, 450, 153, 540],
                        [ 153, 450, 63, 450],
                        [ 153, 154, 63, 154],
                        [ 153, 253, 63, 253],
                        [ 153, 339, 63, 339],
                        [ 267, 154, 267, 63],
                        [ 153, 154, 153, 63],
                        [ 269, 154, 365, 154],
                        [ 269, 205, 365, 205],
                        [ 365, 205, 365, 300],
                        [ 645, 205, 645, 300],
                        [ 505, 205, 505, 300],
                        [ 645, 300, 735 ,300],
                        [ 645, 447, 735 ,447],
                        [ 645, 447, 645 ,537],
                        [ 505, 447, 505, 537],])

x1 = CHECKPOINTS[:,0]
y1 = CHECKPOINTS[:,1]
x2 = CHECKPOINTS[:,2]
y2 = CHECKPOINTS[:,3]

CHECKPOINT_RADIUS = 30  # Adjust this value based on your game scale

# Correct midpoint calculation (average between the two points)
mid_x = (x1 + x2) / 2
mid_y = (y1 + y2) / 2

# Combine mid_x and mid_y to form the starting positions
starting_pos = np.column_stack((mid_x, mid_y))
ANGLES = np.degrees(np.arctan2(y2 - y1, x2 - x1))

# Convert to integers
STARTING_POSITIONS = starting_pos.astype(int)

index = 0
if index != 0:
    CHECKPOINTS = np.vstack((CHECKPOINTS[index:], CHECKPOINTS[:index]))
elif index == 0:
    CHECKPOINTS = CHECKPOINTS

INIT_POS = STARTING_POSITIONS[index]
INIT_ANGLE = ANGLES[index]

