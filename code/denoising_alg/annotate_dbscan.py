import sys
import argparse
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt, exp

#CROPPED_RESIZE = (160, 160)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

group = ['daesung','gdragon','seungri','taeyang','top']
n_people = len(group)

def main(args):
    # Setup video stream
    cap = cv2.VideoCapture(args.input_video_path)
    in_fps, in_4cc = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FOURCC))
    in_width, in_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.output_video_path, in_4cc, in_fps, (in_width, in_height))
    if (cap.isOpened()==False):
        print('Error opening video stream or file')

    boxes = np.loadtxt(args.csv_path, skiprows = 1, delimiter = ',', dtype = int, usecols = range(6))
    confs = np.loadtxt(args.csv_path, skiprows = 1, delimiter = ',', usecols = 6)

    #performing DBSCAN on the examples
    X = np.zeros((len(boxes),5))
    sq12 = sqrt(12)
    for k in range(len(boxes)):
        X[k][0] = boxes[k][0] #* 0.3 * in_nframes / (in_fps * sq12)
        X[k][1] = ((boxes[k][1] + boxes[k][2])/2.0) #/ (in_width * sq12)
        X[k][2] = ((boxes[k][3] + boxes[k][4])/2.0) #/ (in_height * sq12)
        X[k][3] = (boxes[k][2] - boxes[k][1]) #/ (in_width * sq12)
        X[k][4] = (boxes[k][4] - boxes[k][3]) #/ (in_height * sq12)

    X = StandardScaler().fit_transform(X)
    for k in range(len(boxes)):
        X[k][0] *= ((0.3 * in_nframes) / (in_fps * sq12))
        X[k][3] = exp(-X[k][3] * 2)
        X[k][4] = exp(-X[k][4] * 2)
    db = DBSCAN(eps = 0.7, min_samples = 4).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    n_noise_ = list(labels).count(-1)

    # Each label does majority vote
    votes = np.zeros((n_clusters_,n_people))
    for k in range(len(boxes)):
        if labels[k] != -1 and confs[k] > 0.65 and (boxes[k][4]-boxes[k][3]) > 40 and (boxes[k][2]-boxes[k][1]) > 40:
            votes[labels[k]][boxes[k][5]] += confs[k]


    #collect votes, invalidate low-voting ones
    winners = np.ones(n_clusters_, dtype = int)*-1
    for k in range(n_clusters_):
        if max(votes[k]) <= 1:
            for l in range(len(boxes)):
                if labels[l] == k:
                    labels[l] = -1
                #elif labels[l] > k:
                #    labels[l] -= 1
        else:
            winners[k] = np.argmax(votes[k])
    shots = []
    for i in range(n_clusters_):
        shots.append([])
    newbox = []
    #fixed = 0: no fix made
    #fixed = -1: deleted
    #fixed = 1: fixed label only
    #fixed = 2: interpolated boxes
    for k in range(len(boxes)):
        if labels[k] != -1:
            fixed = 0
            #if labels[k] > len(shots) - 1:
            #    shots.append([])
            if boxes[k][5] != winners[labels[k]]:
                boxes[k][5] = winners[labels[k]]
                fixed = 1
            newbox.append(np.append(boxes[k],np.array([labels[k],fixed])))
            shots[labels[k]].append(newbox[-1])
        else:
            boxes[k][5] = -1
            newbox.append(np.append(boxes[k],np.array([labels[k],-1])))

    shots_ppl = []
    for i in range(n_people):
        shots_ppl.append([])
    for shot in shots:
        if len(shot)!= 0:
            shots_ppl[shot[0][5]].append(shot)
            print(shot[0])



    for n in range(n_people):
        for i in range(len(shots_ppl[n])):
            shot = shots_ppl[n][i]
            if len(shot) > 1:
                for k in range(len(shot)-1):
                    if shot[k+1][0] - shot[k][0] > 1:
                        gap = shot[k+1][0] - shot[k][0]
                        for j in range(1, gap):
                            newbox.append(np.array((j * shot[k+1] + (gap - j) * shot[k])/gap, dtype = int))
                            newbox[-1][-1] = 2
                #manual cluster condensation
                if i < len(shots_ppl[n]) - 1:
                    last_frame = shots_ppl[n][i][-1]
                    next_frame = shots_ppl[n][i+1][0]
                    #the remaining dimensions are same anyways
                    x_diff = abs((next_frame[2] + next_frame[1] - (next_frame[2] + next_frame[1]))/2)
                    y_diff = abs((next_frame[4] + next_frame[3] - (next_frame[4] + next_frame[3]))/2)
                    gap = abs(next_frame[0] - last_frame[0])
                    if gap < 100 and gap > 1 and x_diff < 100 and y_diff < 100:
                        for j in range(1, gap):
                            newbox.append(np.array((j * next_frame + (gap - j) * last_frame)/gap, dtype = int))
                            newbox[-1][-1] = 2

    newbox = np.array(newbox)
    boxes = newbox[newbox[:,0].argsort()]


    first_box = 0
    i = 0
    count_all = 0
    count_fixed = 0
    count_added = 0
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
                count_all += 1
                # Put bounding box and text onto video
                cv2.putText(frame, str(boxes[first_box][6]), (r_x,u_y), FONT, FONT_SCALE,BLUE, 2)
                if boxes[first_box][7] == 0:
                    cv2.putText(frame, group[boxes[first_box][5]], (l_x, b_y), FONT, FONT_SCALE, BLUE, 3)
                    cv2.rectangle(frame, (l_x,u_y), (r_x, b_y), BLUE, 3)
                elif boxes[first_box][7] == 1:
                    cv2.putText(frame, group[boxes[first_box][5]], (l_x, b_y), FONT, FONT_SCALE, RED, 3)
                    cv2.rectangle(frame, (l_x,u_y), (r_x, b_y), BLUE, 3)
                    count_fixed += 1
                else:
                    cv2.putText(frame, group[boxes[first_box][5]], (l_x, b_y), FONT, FONT_SCALE, RED, 3)
                    cv2.rectangle(frame, (l_x,u_y), (r_x, b_y), RED, 3)
                    count_fixed += 1
                    count_added += 1
            else:
                #cv2.rectangle(frame, (l_x,u_y), (r_x,b_y), RED, 2)
                cv2.line(frame, (l_x,u_y), (r_x,b_y), RED, 2)
                cv2.line(frame, (l_x,b_y), (r_x,u_y), RED, 2)
            first_box += 1
        out.write(frame)
        i += 1
        if (i % 100 == 0):
            print('Processing frame {}/{}...'.format(i, in_nframes))
    print(float(count_fixed)/float(count_all))
    print(float(count_added)/float(count_all))

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
