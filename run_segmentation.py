"""Load Modules"""
import os
import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf
from imageio import imread, imsave
import tensorflow.contrib.slim as slim

from datetime import datetime
from tqdm import tqdm

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

import xml.etree.ElementTree as ET
import argparse



parser = argparse.ArgumentParser(description='Run dhSegment and generate Zone.xml that follows PAGE XML-schema (https://www.primaresearch.org/tools/PAGELibraries)')
parser.add_argument('-i', '--imagepath', type=str, required=True,
                   help='a path to the root directory of Zone xml files')

parser.add_argument('-s', '--savepath', type=str, required=True,
                   help='a path to the root directory of save files')

parser.add_argument('-t', '--threshold', type=int, default=0.005
                   help='a threshold for ignoring small zones [0,1] (default: 0.005')

args = parser.parse_args()


"""Parse Args"""
IMAGE_DIR = args.imagepath
SAVE_DIR  = args.savepath
CONNECTIVITY  = 4
SM_ZONE_RATIO = args.threshold
LOG_DIR = './log'
log_filename = datetime.now().strftime('dhSegment_%H_%M_%d_%m_%Y.log')
os.environ["CUDA_VISIBLE_DEVICES"]="0"


BG_ID     = 0
TEXT_ID   = 1
FIGURE_ID = 2
LINE_ID   = 3
TABLE_ID  = 4


"""Start session"""
session = tf.InteractiveSession()

"""Load model"""
model_dir = './models/ENP_500_model_v3/export/1564890842/'
m = LoadedModel(model_dir, predict_mode='filename')


"""
1. Preparation 

Input batch
"""
image_list = glob(os.path.join(IMAGE_DIR,'**/*.jpg'),recursive=True)

with open(os.path.join(LOG_DIR,log_filename),'w') as fl:
    fl.write("{} file(s) are found under:\n{}\n".format(len(image_list),IMAGE_DIR))

    for _idx,input_path in enumerate(tqdm(image_list)):    
        try:

            fl.write("\n***[{}/{}]***\n".format(_idx+1,len(image_list)))

            # Create the file structure
            data_PcGts = ET.Element('PcGts')
            data_PcGts.set('xmlns','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15')
            data_PcGts.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
            data_PcGts.set('xsi:schemaLocation','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd')
            data_meta = ET.SubElement(data_PcGts, 'Metadata')
            data_page = ET.SubElement(data_PcGts,'Page')

            # Read image
            img = cv2.imread(input_path)

            # Parse filename
            basename   = os.path.basename(input_path)
            basename_wo_ext = os.path.splitext(basename)[0]
            fl.write("input_path\t\t: {}\n".format(input_path))
            fl.write("basename\t\t: {}\n".format(basename))
            fl.write("basename (w/o ext)\t: {}\n".format(basename_wo_ext))



            """
            2. Main

            Run prediction
            """
            # Run prediction
            prediction_outputs = m.predict(input_path)
            pred_labels = np.copy(prediction_outputs['labels'][0]).astype(np.uint8)



            """
            2. Main

            Get basic attributes
            """
            oriH,oriW = np.shape(img)[:2]
            newH,newW = np.shape(pred_labels)

            data_page.set("HEIGHT",str(oriH))
            data_page.set("WIDTH",str(oriW))



            """
            3. Postprocessing

            Generate binary mask for each class
            """
            mask_texts   = np.copy(pred_labels)
            mask_texts[mask_texts != TEXT_ID] = 0



            """
            3. Postprocessing

            Generate polygones for each class
            """
            txt_num_labels, txt_labels, txt_stats, txt_centroids = cv2.connectedComponentsWithStats(mask_texts, CONNECTIVITY, cv2.CV_32S)



            """
            4. Postprocessing (TextRegion; Rectangle)

            """
            factor_h = oriH/newH
            factor_w = oriW/newW

            # Get rectangle region
            cnt_remove = 0
            THRESHOLD_SM_ZONE = (newH*newW)*SM_ZONE_RATIO
            region_idx = 0
            for bb_idx in range(1,txt_num_labels):
                if txt_stats[bb_idx][4] < THRESHOLD_SM_ZONE:
                    cnt_remove+=1
                    continue
                # Resize predicted coordinate
                left,top,width,height   = txt_stats[bb_idx][:4]

                left   = int(left*factor_w)
                width  = int(width*factor_w)
                top    = int(top*factor_h)
                height = int(height*factor_h)

                p1 = (left,top)
                p2 = (left+width,top+height)

                region_idx +=1

                # Inject coordinates
                data_textBlock = ET.SubElement(data_page, 'TextBlock')
                data_textBlock.set("ID",str(region_idx))
                data_textBlock.set("HEIGHT",str(height))
                data_textBlock.set("WIDTH",str(width))
                data_textBlock.set("HPOS",str(left))
                data_textBlock.set("VPOS",str(top))

            # Finalize file structure in xml format
            data_page_xml = ET.tostring(data_PcGts)

            # Save xml
            save_xml_filename = basename_wo_ext + '_dhSegment' + '.xml'
            with open(os.path.join(SAVE_DIR, save_xml_filename), "wb") as data_page_xml_file:
                data_page_xml_file.write(data_page_xml)

            fl.write("Total {} textRegion(s) are found.\n...{} region(s) are removed from the original finding.\n".format(txt_num_labels-cnt_remove,cnt_remove))
        except Exception as e:
            fl.write("unexpected error occurred:\n{}".format(e))
            pass



