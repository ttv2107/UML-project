import sys
import argparse
import cv2
import numpy as np

CROPPED_RESIZE = (160, 160)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE              = 2
FONT_COLOR              = (255,0,0)

bigbang = ['daesung','gdragon','seungri','taeyang','TOP']

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

    i = 0
    boxes = np.loadtxt(args.csv_path, skiprows = 1, delimiter = ',', dtype = int)
    first_box = 0
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

            class_txt = bigbang[boxes[first_box][5]]
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
#    parser.add_argument('classifier_path', type=str, help='Path to the classifier pkl.')
#    parser.add_argument('nn_path', type=str, help='Path to the nn model that produces embedding.')
    parser.add_argument('csv_path', type=str, help='Path to which to write the csv containing bounding boxes and labels.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
