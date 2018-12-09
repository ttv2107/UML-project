import argparse
import sys
import cv2
import facenet
import pickle
import csv
from scipy import misc
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import numpy as np

CROPPED_RESIZE = (160, 160)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE              = 2
FONT_COLOR              = (255,255,255)

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

#CHANGE: in_fps lowered to make classification faster
    in_fps = 5
    in_width, in_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#output fps is also changed
    out = cv2.VideoWriter(args.output_video_path, in_4cc, in_fps, (in_width, in_height))
    if (cap.isOpened()==False):
        print('Error opening video stream or file')

    # Setup face detector and classifier
    detector = MTCNN()
    with open(args.classifier_path, 'rb') as infile:
        (face_classifier, class_names) = pickle.load(infile)
    print(class_names)
    with tf.Graph().as_default(), tf.Session() as sess, open(args.csv_path, 'w', newline='\n') as csvfile:
            # Get input and output tensors
            facenet.load_model(args.nn_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Prepare csv file for writing
            fieldnames = ['frame', 'left_x', 'right_x', 'top_y', 'bottom_y', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            i = 0
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    print('Frame {} could not be read.'.format(i))
                    break
                for face in detector.detect_faces(frame):
                    # Detect face and preprocess
                    emb_array = np.zeros((1, embedding_size))
                    x, y, w, h = face['box']
                    left_x, right_x = ensure(x, in_width), ensure(x+w, in_width)
                    top_y, bottom_y = ensure(y, in_height), ensure(y+w, in_height)

                    face_img = facenet.prewhiten(misc.imresize(frame[top_y:bottom_y, left_x:right_x, :],
                                                               CROPPED_RESIZE,
                                                               interp='bilinear'))

                    # Find the embedding using the nn
                    feed_dict = { images_placeholder: [face_img],
                                  phase_train_placeholder: False }
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                    # Use svm to find the class
                    predictions = face_classifier.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    class_txt = '{}: {:.3f}'.format(class_names[best_class_indices[0]], best_class_probabilities[0])

                    # Put bounding box and text onto video
                    cv2.putText(frame, class_txt, (x, y+h), FONT, FONT_SCALE, FONT_COLOR)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

                    # Write to csv
                    writer.writerow({'frame': i, 'left_x': left_x, 'right_x': right_x,
                                     'top_y': top_y, 'bottom_y': bottom_y,
                                     'label': best_class_indices[0],'conf': best_class_probabilities[0]})

                out.write(frame)
                i += 1
                if (i % 50 == 0):
                    print('Processing frame {}/{}...'.format(i, in_nframes))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_video_path', type=str, help='Path to the input video.')
    parser.add_argument('output_video_path', type=str, help='Path to the output video.')
    parser.add_argument('classifier_path', type=str, help='Path to the classifier pkl.')
    parser.add_argument('nn_path', type=str, help='Path to the nn model that produces embedding.')
    parser.add_argument('csv_path', type=str, help='Path to which to write the csv containing bounding boxes and labels.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
