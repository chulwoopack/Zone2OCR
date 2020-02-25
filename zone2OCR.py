import os, sys, errno
import numpy as np
import cv2
from xml.dom import minidom
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm
from glob import glob
import argparse
import json



parser = argparse.ArgumentParser(description='Read OCR.xml that follows PAGE XML-schema (https://www.primaresearch.org/tools/PAGELibraries) and map zone-level coordinates to the corresponding OCR text contents')
parser.add_argument('-zx', '--zonexmlpath', type=str, required=True,
                   help='a path to the root directory of Zone xml files')

parser.add_argument('-ox', '--ocrxmlpath', type=str, required=True,
                   help='a path to the root directory of OCR xml files')

parser.add_argument('-t', '--iouthreshold', type=float, default=0.1,
                   help='an IoU threshold ([0,1]) for mapping (default=0.1)')

parser.add_argument('-s', '--savepath', type=str, required=True,
                   help='a path to the root directory of save files')

parser.add_argument('-v', '--verbose',
                    help='increase output verbosity',
                    action='store_false')

args = parser.parse_args()



"""
PARAMS
"""
ZONE_XML_PATH = args.zonexmlpath
OCR_XML_PATH  = args.ocrxmlpath
SAVE_PATH     = args.savepath
IOU_THRESHOLD = args.iouthreshold
DEBUG         = args.verbose

if DEBUG:
    print("ZONE_XML_PATH\t: {}".format(ZONE_XML_PATH))
    print("OCR_XML_PATH\t: {}".format(OCR_XML_PATH))
    print("SAVE_PATH\t: {}".format(SAVE_PATH))



"""
MAIN
"""
# Preparation
zone_xml_file_paths = sorted(glob(os.path.join(ZONE_XML_PATH,'**/*.xml'),recursive=True))
ocr_xml_file_paths  = sorted(glob(os.path.join(OCR_XML_PATH,'**/*.xml'),recursive=True))

for idx in tqdm(len(zone_xml_file_paths)):
    zone_xml_file_path = zone_xml_file_paths[idx]
    ocr_xml_file_path  = ocr_xml_file_paths[idx]

    print("[{}/{}] Processing \nzone xml: {}\nOCR xml: {}".format(idx+1,len(zone_xml_file_paths),zone_xml_file_path,ocr_xml_file_path))
    


    """
    Zone Processing
    """
    # Read Zone xml
    xmldoc = minidom.parse(zone_xml_file_path)

    # Get image dimension and resize factor
    img_w  = int(xmldoc.getElementsByTagName('Page')[0].attributes['WIDTH'].value)
    img_h  = int(xmldoc.getElementsByTagName('Page')[0].attributes['HEIGHT'].value)

    # Count number of text-blocks
    zone_textBlocks = xmldoc.getElementsByTagName('TextBlock')

    if DEBUG:
        print("Zone XML:")
        print("{} \tWidth (factored)".format(img_w)) 
        print("{} \tHeight (factored)".format(img_h))
        print("{} \tTextBlock(s)".format(len(zone_textBlocks)))

    """
    OCR Processing
    """
    # Read OCR xml
    xmldoc = minidom.parse(ocr_xml_file_path)

    # Get image dimension and resize factor
    image_width  = int(xmldoc.getElementsByTagName('Page')[0].attributes['WIDTH'].value)
    image_height = int(xmldoc.getElementsByTagName('Page')[0].attributes['HEIGHT'].value)

    _processingStepSettings = str(xmldoc.getElementsByTagName('processingStepSettings')[0].childNodes[0].nodeValue)
    _attributes = _processingStepSettings.split('\n')
            
    factor = img_w/image_width

    # Count number of text-blocks
    ocr_textBlocks = xmldoc.getElementsByTagName('TextBlock')

    if DEBUG:
        print("OCR XML:")
        print("{} \tWidth (original)".format(image_width)) 
        print("{} \tHeight (original)".format(image_height))
        print("{} \tWidth (factored)".format(img_w)) 
        print("{} \tHeight (factored)".format(img_h))
        print("{} \tTextBlock(s)".format(len(ocr_textBlocks)))



    """
    MAPPING
    """
    # output json
    map_json = []

    for zone_idx,zone_textBlock in enumerate(tqdm(zone_textBlocks)):
        # zone coordinates
        zone_width  = int(float(zone_textBlock.attributes["WIDTH"].value))
        zone_height = int(float(zone_textBlock.attributes["HEIGHT"].value))
        zone_vpos   = int(float(zone_textBlock.attributes["VPOS"].value))
        zone_hpos   = int(float(zone_textBlock.attributes["HPOS"].value))

        zone_p1 = (zone_hpos,zone_vpos)
        zone_p2 = ((zone_hpos+zone_width),zone_vpos)
        zone_p3 = (zone_hpos,(zone_vpos+zone_height))
        zone_p4 = ((zone_hpos+zone_width),(zone_vpos+zone_height))

        zone_coord = [zone_p3, zone_p4, zone_p2, zone_p1]

        # Build json
        _textBlock_xml = {}
        _textBlock_xml["zone_coord"] = [list(zone_p3), list(zone_p4), list(zone_p2), list(zone_p1)]
        _set_ocr_textBlocks = []
        _set_ocr_contents   = []

        _sub_ocr_contents   = []

        for ocr_idx,ocr_textBlock in enumerate(ocr_textBlocks):
            # OCR coordinates
            ocr_width  = int(float(ocr_textBlock.attributes["WIDTH"].value))
            ocr_height = int(float(ocr_textBlock.attributes["HEIGHT"].value))
            ocr_vpos   = int(float(ocr_textBlock.attributes["VPOS"].value))
            ocr_hpos   = int(float(ocr_textBlock.attributes["HPOS"].value))

            width  = int(ocr_width*factor)
            height = int(ocr_height*factor)
            vpos   = int(ocr_vpos*factor)
            hpos   = int(ocr_hpos*factor)

            ocr_p1 = (hpos,vpos)
            ocr_p2 = ((hpos+width),vpos)
            ocr_p3 = (hpos,(vpos+height))
            ocr_p4 = ((hpos+width),(vpos+height))

            ocr_coord = [ocr_p3, ocr_p4, ocr_p2, ocr_p1]

            # Find matching regions
            zone_polygon = Polygon(zone_coord)
            ocr_polygon  = Polygon(ocr_coord)

            iou = zone_polygon.intersection(ocr_polygon).area / zone_polygon.union(ocr_polygon).area

            if(iou >= IOU_THRESHOLD):
                # Set of OCR touching the Zone
                set_contents = ''
                sub_contents = ''
                ocr_textLines = ocr_textBlock.getElementsByTagName('TextLine')

                for ocr_textline in ocr_textLines:
                    # Textline coordinates
                    txt_width  = int(float(ocr_textline.attributes["WIDTH"].value))
                    txt_height = int(float(ocr_textline.attributes["HEIGHT"].value))
                    txt_vpos   = int(float(ocr_textline.attributes["VPOS"].value))
                    txt_hpos   = int(float(ocr_textline.attributes["HPOS"].value))

                    width  = int(txt_width*factor)
                    height = int(txt_height*factor)
                    vpos   = int(txt_vpos*factor)
                    hpos   = int(txt_hpos*factor)

                    txt_p1 = (hpos,vpos)
                    txt_p2 = ((hpos+width),vpos)
                    txt_p3 = (hpos,(vpos+height))
                    txt_p4 = ((hpos+width),(vpos+height))

                    txt_coord  = [txt_p3, txt_p4, txt_p2, txt_p1]

                    txt_polygon = Polygon(txt_coord)

                    # Textline string
                    strings = ocr_textline.getElementsByTagName('String')
                    for string in strings:
                        set_contents += (str(string.attributes["CONTENT"].value) + ' ')

                        # Subset of OCR within the Zone
                        if(zone_polygon.intersects(txt_polygon)):
                            sub_contents += (str(string.attributes["CONTENT"].value) + ' ')

                # Build json
                _sub_ocr_contents.append(sub_contents)
                _set_ocr_textBlocks.append([list(ocr_p3), list(ocr_p4), list(ocr_p2), list(ocr_p1)])
                _set_ocr_contents.append(set_contents)

        # Build json
        _textBlock_xml["zone_texts"] = _sub_ocr_contents
        _textBlock_xml["ocr_coords"] = _set_ocr_textBlocks
        _textBlock_xml["ocr_texts"]  = _set_ocr_contents
        map_json.append(_textBlock_xml)

    # Save json
    data = json.dumps(map_json)
    out_json_filename = os.path.basename(ocr_xml_file_path).split('.')[0] + '.json'
    with open(os.path.join(SAVE_PATH,out_json_filename),'w') as out_json_fp:
        json.dump(map_json, out_json_fp) 


print("Done.")
