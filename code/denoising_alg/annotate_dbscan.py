import sys
import argparse
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

CROPPED_RESIZE = (160, 160)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE              = 2
FONT_COLOR              = (255,0,0)


group = ['daesung','gdragon','seungri','taeyang','top']
n_people = len(group)

def ensure(val, max_val):
    # Just to make sure in case coordinates are returned as negative or if
    # they're bigger than the max width/height
    if val < 0:
        return 0
    elif val > max_val:
        return max_val
    else:
        return val

def main(args):
    # Setup video stream
    cap = cv2.VideoCapture(args.input_video_path)
    in_fps, in_4cc = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FOURCC))

#CHANGE? in_fps lowered to make classification faster
#    in_fps = 5
    in_width, in_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#output fps is also changed
    out = cv2.VideoWriter(args.output_video_path, in_4cc, in_fps, (in_width, in_height))
    if (cap.isOpened()==False):
        print('Error opening video stream or file')

    boxes = np.loadtxt(args.csv_path, skiprows = 1, delimiter = ',', dtype = int)

    #performing DBSCAN on the examples
    X = np.zeros((len(boxes),3))
    for k in range(len(boxes)):
        X[k][0] = boxes[k][0]
        X[k][1] = (boxes[k][1] + boxes[k][2])/2.0
        X[k][2] = (boxes[k][3] + boxes[k][4])/2.0

    X = StandardScaler().fit_transform(X)
    for k in range(len(boxes)):
        X[k][0] *= 3
    db = DBSCAN(eps = 0.3, min_samples = 5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #n_noise_ = list(labels).count(-1)

    # Each label does majority vote
    # Group frames using indices

    votes = np.zeros((n_clusters_,n_people))
    for k in range(len(boxes)):
        if labels[k] != -1:
            votes[labels[k]][boxes[k][5]] += 1

    winners = np.ones(n_clusters_, dtype = int)*-1
    for k in range(n_clusters_):
        winners[k] = np.argmax(votes[k])

    shots = []
    newbox = []
    for k in range(len(boxes)):
        if labels[k] != -1:
            if labels[k] > len(shots) - 1:
                shots.append([])

            #shots[labels[k]].append(np.concatenate((boxes[k,0:5], np.array([winners[labels[k]]]))))
            boxes[k][5] = winners[labels[k]]
            newbox.append(np.append(boxes[k],np.array(labels[k])))
            shots[labels[k]].append(newbox[-1])
        else:
            boxes[k][5] = -1
            newbox.append(np.append(boxes[k],np.array(labels[k])))
        if(k % 100 == 0):
            print(len(boxes[k]))


    for shot in shots:
        if len(shot) > 1:
            for k in range(len(shot)-1):
                if shot[k+1][0] - shot[k][0] > 1:
                    gap = shot[k+1][0] - shot[k][0]
                    for j in range(1, gap):
                        newbox.append(np.array((j * shot[k+1] + (gap - j) * shot[k])/gap, dtype = int))
    newbox = np.array(newbox)
    boxes = newbox[newbox[:,0].argsort()]


    first_box = 0
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('Frame {} could not be read.'.format(i))
            break
        while(first_box < len(boxes) and boxes[first_box][0] == i):
            l_x = boxes[first_box][1]
            r_x = boxes[first_box][2]
            u_y = boxes[first_box][3]
            b_y = boxes[first_box][4]

            if boxes[first_box][6] != -1:
                class_txt = str(boxes[first_box][6]) + group[boxes[first_box][5]]
                # Put bounding box and text onto video
                cv2.putText(frame, class_txt, (l_x, b_y), FONT, FONT_SCALE, FONT_COLOR)
                cv2.rectangle(frame, (l_x,u_y), (r_x, b_y), (0, 255, 0), 2)
            first_box += 1
        out.write(frame)
        i += 1
        if (i % 100 == 0):
            print('Processing frame {}/{}...'.format(i, in_nframes))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_video_path', type=str, help='Path to the input video.')
    parser.add_argument('output_video_path', type=str, help='Path to the output video.')
    parser.add_argument('csv_path', type=str, help='Path to the csv containing the noisy bounding boxes and labels.')
#    parser.add_argument('classifier_path', type=str, help='Path to the classifier pkl.')
#    parser.add_argument('nn_path', type=str, help='Path to the nn model that produces embedding.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))