import paddlehub as hub
import cv2
import numpy as np
import glob
import os
import random
import argparse

class segUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="deeplabv3p_xception65_humanseg")

    def doseg(self, frame):
        res = self.module.segmentation(images=[frame], use_gpu=True)
        return res[0]['data']

def randomCrop(frame, h, w):
    bh, bw = frame.shape[:2]
    if bw - w > 0 and bh - h > 0:
        randx = random.randint(0, bw - w)
        randy = random.randint(0, bh - h)

        return frame[randy:randy + h,randx:randx + w]
    else:
        return cv2.resize(frame, (w, h))


def main(args):

    backlist = glob.glob(os.path.join(args.backdir, "*.jpg"))
    handlist = glob.glob(os.path.join(args.handdir, "*.png")) + glob.glob(os.path.join(args.handdir, "*.jpg"))
    print("back image: ", len(backlist))
    print("hand image: ", len(handlist))
    save_dir = args.savedir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        SU = segUtils()

        for handpath in handlist:
            img = cv2.imread(handpath)
            basename = os.path.basename(handpath)
            filename, ext = os.path.splitext(basename)
            mask = SU.doseg(img)
            mask[mask <= 1] = 0
            mask[mask > 1] = 1
            
            mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
            h,w = img.shape[:2]
            for i in range(2):
                rback = cv2.imread(backlist[random.randint(0, len(backlist)-1)])
                crop = randomCrop(rback, h, w)
                res = mask * img + (1 - mask) * crop
                newname = filename + str(i) + ext
                cv2.imwrite(os.path.join(save_dir,newname), res.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backdir", type=str, required=True)
    parser.add_argument("--handdir", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    args = parser.parse_args()
    main(args)
