import tensorflow as tf
import numpy as np
import PIL.Image
import tfutil
import argparse
from pathlib import Path

# python 1-dataset_display.py --tfrecords processData/AIS/AIS-r06.tfrecords --number 10 --outdir demo/output
# AIS-review-r02 to -r06 means different resolution of the picture
def argparser():
    parser = argparse.ArgumentParser(description='Dataset demo pipeline: transform picture from tfrecords file to png format')
    parser.add_argument('--tfrecords', type=str, default="processData/AIS/AIS-r06.tfrecords", help='the tfrecords file path')
    parser.add_argument('--number', type=int, default=10, help='set the number of dataset images to show')
    parser.add_argument('--outdir', type=str, default="demo/output", help='the output path')
    return parser.parse_args()

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)
    
def main():
    args = argparser()

    # make output directory
    outdirPATH = Path(args.outdir)
    outdirPATH.mkdir(exist_ok=True, parents=True)

    tfr_file = args.tfrecords
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    tfr_ite = tf.python_io.tf_record_iterator(tfr_file,tfr_opt)
    for i in range(int(args.number)):
        images_tf = next(tfr_ite)

        images = parse_tfrecord_np(images_tf)
        # Convert images to PIL-compatible format.
        PIL.Image.fromarray(images[0]).convert('L').save(outdirPATH / 'dataset_img_{}.png'.format(i))
    print("Finish")
    
if __name__ == '__main__':
    main()