import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import os
import argparse
from pathlib import Path

# python 2-inference.py --model demo/weight/AIS.pkl --number 10 --outdir demo/output

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set up the use of gpu
num_channel = 1

def argparser():
    parser = argparse.ArgumentParser(description='Nodule generation pipeline')
    parser.add_argument('--model', type=str, default="demo/weight/AIS.pkl", help='the model weight file path')
    parser.add_argument('--number', type=int, default=10, help='set the number of output images')
    parser.add_argument('--outdir', type=str, default="demo/output", help='the output path')
    return parser.parse_args()

def main():
    args = argparser()
    # Initialize TensorFlow session.
    tf.InteractiveSession()
    
    # make output directory
    outdirPATH = Path(args.outdir)
    outdirPATH.mkdir(exist_ok=True, parents=True)

    # Import official networks.
    with open(args.model, 'rb') as file:
        G, D, Gs = pickle.load(file)

    for i in range(int(args.number)):
        # Generate latent vectors.
        latents = np.random.randn(1, *Gs.input_shapes[0][1:]) # 1 random latents, generate one image once 

        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
        PIL.Image.fromarray(images[0,:,:,0]).convert('L').save(outdirPATH / 'inference_img_{}.png'.format(i))
    print("Finish")

if __name__ == '__main__':
    main()
