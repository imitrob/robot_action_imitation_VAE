import cv2, math
import numpy as np

def get_body_angles(canvas, body_peaks, draw = True):
    '''Returns arm angles as a List in this order [LShoulderRoll, LElbowRoll, RShoulderRoll, RElbowRoll] '''

    angles = []
    body_edge_pairs = [
        [[1, 2], [2, 3]],
        [[2, 3], [3, 4]],
        [[1, 5], [5, 6]],
        [[5, 6], [6, 7]],
    ]

    body_peaks_as_array = np.array(body_peaks)
    for e, e2 in body_edge_pairs:
        angle = get_angle(e, e2, body_peaks_as_array)

        if angle != -1:
            mid_point = set(e).intersection(set(e2)).pop()
            x, y = body_peaks[mid_point]
            if e == [2,3]:
                if body_peaks_as_array[3][1] - body_peaks_as_array[4][1]  < 0:
                  angle = -1*angle
            if e == [5,6]:
                if body_peaks_as_array[6][1] - body_peaks_as_array[7][1]  < 0:
                  angle = -1*angle
            angles.append(angle)
            if draw:
                cv2.putText(canvas, str(int(angle)) + "deg", (int(x)-20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            angles.append(0)

    if len(angles) < 4:
        return [0, 0, 0, 0]
    return angles


def get_finger_angle(e, peaks, edges, ax, x1, y1, draw = True):
    ''' Returns angle between two finger edges '''

    for ie2, e2 in enumerate(edges):
        if(e != e2):
            angle = get_angle(e, e2, peaks)

            if draw and angle != -1 and not math.isnan(angle):
                ax.text(x1, y1, str(int(angle)))

    return angle


def get_angle(edg1,  edg2, peaks):
    ''' Returns angle between two edges '''

    edge1 = set(edg1)
    edge2 = set(edg2)

    if len(edge1.intersection(edge2)) == 0 or len(edge1.intersection(edge2)) == 2:
        return -1

    mid_point = edge1.intersection(edge2).pop()

    a = (edge1-edge2).pop()
    b = (edge2-edge1).pop()
    if len(peaks) > max(a, b, mid_point):
        v1 = peaks[mid_point]-peaks[a]
        v2 = peaks[mid_point]-peaks[b]

        angle = (math.degrees(np.arccos(np.dot(v1,v2)
                                        /(np.linalg.norm(v1)*np.linalg.norm(v2)))))
        return angle
    
    else:
        return -1
