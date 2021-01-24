import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

#This function will start from the very bottom and go to 3/4th of the coordinate

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])



def canny(lane_image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)#5,5 is kernal
    canny = cv2.Canny(blur,50,150)#50 and 150 are ratio of low threshold to high threshold
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image



"""

Syntax: cv2.line(image, start_point, end_point, color, thickness)

Parameters:
image: It is the image on which line is to be drawn.
start_point: It is the starting coordinates of line.
The coordinates are represented as tuples of two values
 i.e. (X coordinate value, Y coordinate value).
end_point: It is the ending coordinates of line.
The coordinates are represented as tuples of two
 values i.e. (X coordinate value, Y coordinate value).
color: It is the color of line to be drawn. For BGR, we
 pass a tuple. eg: (255, 0, 0) for blue color.
thickness: It is the thickness of the line in px.

Return Value: It returns an image.
            """


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


image = cv2.imread('test_image.jpg')
lane_image=np.copy(image)
canny_image = canny(lane_image)#we've passed initial rgb image here
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)



"""cv2.HoughLinesP(image,rho, theta, threshold, np.array ([ ]), minLineLength=xx, maxLineGap=xx)
edges: Output of the edge detector.
lines: A vector to store the coordinates of the start and end of the line.
rho: The resolution parameter \rho in pixels.
theta: The resolution of the parameter \theta in radians.
threshold: The minimum number of intersecting points to detect a line."""

averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,averaged_lines)

combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
"""
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
rc1 – first input array.
alpha – weight of the first array elements.
src2 – second input array of the same size and channel number as src1.
beta – weight of the second array elements.
dst – output array that has the same size and number of channels as the input arrays.
gamma – scalar added to each sum.
dtype – optional depth of the output array; when both input arrays have the same depth,
 dtype can be set to -1, which will be equivalent to src1.depth().
"""
cv2.imshow("result",combo_image)
cv2.waitKey(0)
