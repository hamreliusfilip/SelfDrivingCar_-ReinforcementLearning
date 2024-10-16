import numpy as np
import cv2

WIDTH, HEIGHT = 800, 600

# Load the image
image = cv2.imread('new_track.png')
image = cv2.resize(image, (WIDTH, HEIGHT))

## BARRIÄRER ##
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 100)

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
# Nya checkpoints - Nu är det i rätt ordning (vänster-varv)
CHECKPOINTS = np.array([[ 381, 450, 381, 540],
                        [ 505, 447, 505, 537],
                        [ 645, 447, 645 ,537],
                        [ 645, 447, 735 ,447],
                        [ 645, 300, 735 ,300],
                        [ 645, 205, 645, 300],
                        [ 505, 205, 505, 300],
                        [ 365, 205, 365, 300],
                        [ 269, 205, 365, 205],
                        [ 269, 154, 365, 154],
                        [ 267, 154, 267, 63],
                        [ 153, 154, 153, 63],
                        [ 153, 154, 63, 154],
                        [ 153, 253, 63, 253],
                        [ 153, 339, 63, 339],
                        [ 153, 445, 63, 445],
                        [ 153, 450, 153, 540],
                        [ 267, 450, 267, 540],
                        ])

x1 = CHECKPOINTS[:,0]
y1 = CHECKPOINTS[:,1]
x2 = CHECKPOINTS[:,2]
y2 = CHECKPOINTS[:,3]

CHECKPOINT_RADIUS = 30  # Justera detta värde baserat på skalan

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

# -------------- Visualization --------------

# Draw checkpoints
for i, checkpoint in enumerate(CHECKPOINTS):
    # Extract coordinates
    x1, y1, x2, y2 = checkpoint

    # Draw the checkpoint as a line
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Draw a circle at the midpoint of the checkpoint
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.circle(image, midpoint, 5, (0, 255, 0), -1)

    # Put text showing the order number of the checkpoint
    cv2.putText(image, f'{i+1}', (midpoint[0] - 10, midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Show the image with checkpoints
cv2.imshow('Track with Checkpoints', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
