import sys
import argparse
import cv2
import numpy as np
import csv

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
#    in_fps = 15
    in_width, in_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#CHANGE? output fps is also changed
    out = cv2.VideoWriter(args.output_video_path, in_4cc, in_fps, (in_width, in_height))

    if (cap.isOpened()==False):
        print('Error opening video stream or file')

    #open input csv
    boxes = np.loadtxt(args.input_csv_path, skiprows = 1, delimiter = ',', dtype = int)
    first_box = 0
    #shift first_box index to first valid occurrence
    while boxes[first_box][0] < args.start_frame:
        first_box += 1

    #open output csv
    outcsv = open(args.output_csv_path, 'w', newline = '\n')
    fieldnames = ['frame','left_x','right_x','top_y','bottom_y', 'label']
    writer = csv.DictWriter(outcsv, fieldnames = fieldnames)
    writer.writeheader()

    #write to output csv
    while boxes[first_box][0] < args.end_frame:
        writer.writerow({'frame': boxes[first_box][0] - args.start_frame, 'left_x': boxes[first_box][1], 'right_x': boxes[first_box][2],'top_y': boxes[first_box][3], 'bottom_y': boxes[first_box][4],'label':boxes[first_box][5]})
        first_box += 1
    outcsv.close()

    i = 0
    while cap.isOpened() and i < args.end_frame:
        ret, frame = cap.read()
        if not ret:
            print('Frame {} could not be read.'.format(i))
            break
        if i >= args.start_frame: #and i%2 == 0:
            out.write(frame)
        if i % 100 == 0:
            print('Processing frame {}/{}...'.format(i,in_nframes))
        i += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_video_path', type=str, help='Path to the input video.')
    parser.add_argument('output_video_path', type=str, help='Path to the output video.')
#    parser.add_argument('classifier_path', type=str, help='Path to the classifier pkl.')
    parser.add_argument('input_csv_path', type=str, help='Path to the nn model that produces embedding.')
    parser.add_argument('output_csv_path', type=str, help='Path to which to write the csv containing bounding boxes and labels.')
    parser.add_argument('start_frame',type=int,help='Index of the first frame, inclusive')
    parser.add_argument('end_frame',type=int,help='Index of the last frame, exclusive')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
