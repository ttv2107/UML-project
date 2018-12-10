import os
import sys
from PIL import Image

path = sys.argv[1]
print('Converting images in ' + path)
if not os.path.exists(path + '_png'):
    os.mkdir(path + '_png')
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith(('png', 'jpg', 'jpeg')):
            print('converting ' + name)
            im = Image.open(path + '/' + name)
            im.save(path + '_png/' + name.split('.')[0] + '.png')
