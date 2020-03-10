import os, sys, errno
import re
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from xml.dom import minidom
from shapely.geometry import Polygon
from tqdm import tqdm
from glob import glob

"""process_zone

Args:
    zone_xml_file_path (str): Path to Segmented output xml file
    DEBUG (bool):
        True: Print details
        False: Silent
    
Returns:
    zone_textBlocks (list of DOM elements): list of DOM elements
""" 
def process_zone(zone_xml_file_path=None, DEBUG=False):
    if(zone_xml_file_path==None):
        sys.exit("Zone XML not found.")

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
        
    return zone_textBlocks



"""process_ocr

Args:
    ocr_xml_file_path (str): Path to OCR xml file
    DEBUG (bool):
        True: Print details
        False: Silent
    
Returns:
    ocr_textBlocks (list of DOM elements): list of DOM elements
    factor (float): factor = image_size / actual_scanned_image_size
""" 
def process_ocr(ocr_xml_file_path=None, DEBUG=False):
    if(ocr_xml_file_path==None):
        sys.exit("OCR XML not found.")
        
    # Read OCR xml
    xmldoc = minidom.parse(ocr_xml_file_path)

    # Get image dimension and resize factor
    image_width  = int(xmldoc.getElementsByTagName('Page')[0].attributes['WIDTH'].value)
    image_height = int(xmldoc.getElementsByTagName('Page')[0].attributes['HEIGHT'].value)

    _string      = xmldoc.getElementsByTagName('processingStepSettings')[0].firstChild.nodeValue
    _image_w     = re.search('(?<=width:)[0-9]+', _string)
    img_w        = int(_image_w.group(0))

    _string      = xmldoc.getElementsByTagName('processingStepSettings')[0].firstChild.nodeValue
    _image_h     = re.search('(?<=height:)[0-9]+', _string)
    img_h        = int(_image_h.group(0))

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
        
    return ocr_textBlocks, factor

    
    
"""save_json

Args:
    save_path (str): Path to save directory
    out_json_filename (str): Name of output JSON file
    map_json (JSON instance): Returned object from mapping
    
Returns:
""" 
def save_json(save_path=None, out_json_filename=None, map_json=None):
    if(save_path==None):
        sys.exit("Save path not found.")
    if(out_json_filename==None):
        sys.exit("Provide save filename.")
    if(map_json==None):
        sys.exit("Invalid mapped json file.")
        
    data = json.dumps(map_json)
    
    out_json_path = os.path.join(save_path,out_json_filename)
    with open(out_json_path, 'w') as out_json_fp:
        json.dump(map_json, out_json_fp) 

    print("\nOutput is stored at {}".format(out_json_path))
    
    
    
"""visualize

Args:
    json_file_path (str): Path to mapped JSON file
    usecase (int): One of following options
        1: OCR only
        2: OCR + Segmentation
        3: OCR + Segmentation (exclusive)
    region_idx (int): Index of region of interest
    vis_all (bool):
        True: visualize all OCR textblocks in usecase 1
        False: Visualize a particular textblock based on region_idx
    
Returns:
"""    
def visualize(json_file_path=None, usecase=0, region_idx=None, vis_all=False):
    if(json_file_path==None):
        sys.exit("Mapped JSON not found.")
    if(usecase==0):
        sys.exit("Select usecase: 1, 2, or 3")
    
    # Load JSON
    data = None
    with open(json_file_path) as in_json_fp:
        data = json.load(in_json_fp)
    print("{} is loaded.".format(json_file_path))
    
    if(usecase==1):
        print("Total {} OCR textblocks are found.\n".format(len(data)))
    else:
        print("Total {} zones are found by dhSegment.\n".format(len(data)))
        print("<Inspect zone {} out of {}>\n".format(region_idx+1,len(data)))

    # Load image
    img = cv2.imread(os.path.join("../example/images",os.path.basename(json_file_path).split('.')[0] + '.jpg'))
    if(img is None):
        sys.exit("Cannot read image from {}".format(image_path))
    canvas = np.copy(img)

    # USECASE 1
    if(usecase==1):
        # Visualize all regions
        if(vis_all):
            print("<Visualize all OCR textblocks>\n")
            for region_idx in range(len(data)):
                ocr_textBox = data[region_idx]
                # OCR coords
                cv2.drawContours(canvas,np.int32([np.array(ocr_textBox['ocr_coords'])]),0,(0,0,255),10)
        # Visualize a particular textblock
        else:
            print("<Inspect OCR textblock {} out of {}>\n".format(region_idx+1,len(data)))
            # Grab a textblock
            ocr_textBox = data[region_idx]
            # OCR texts
            out_text = ocr_textBox['ocr_texts']
            print("OCR texts: {}".format(out_text))
            # OCR coords
            cv2.drawContours(canvas,np.int32([np.array(ocr_textBox['ocr_coords'])]),0,(0,0,255),10)
        
    else:
        # Grab a textblock
        zone_textBox = data[region_idx]
        # Draw zone region (red color)
        cv2.drawContours(canvas,np.int32([np.array(zone_textBox['zone_coord'])]),0,(255,0,0),10)
        
        # USECASE 2
        if(usecase==2):
            out_text = ""
            for ocr_idx in range(len(zone_textBox['ocr_coords'])):
                # Draw zone region (blue color)
                cv2.drawContours(canvas,np.int32([np.array(zone_textBox['ocr_coords'][ocr_idx])]),0,(0,0,255),10)

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                topLeftCornerOfText    = tuple(zone_textBox['ocr_coords'][ocr_idx][1])
                fontScale              = 6
                fontColor              = (0,0,255)
                lineType               = 6
                cv2.putText(canvas, str(ocr_idx+1), topLeftCornerOfText, font, fontScale, fontColor,lineType, cv2.LINE_AA)

                out_text += "\nOCR text of region {}:\n{}\n".format(ocr_idx+1,zone_textBox['ocr_texts'][ocr_idx])
            print(out_text)

        # USECASE 3
        else:
            print("Zone texts ({} OCR blocks) within the OCR:\n{}\n".format(len(zone_textBox['zone_texts']), zone_textBox['zone_texts']))
                
    # Visualize
    plt.figure(figsize=(15,15))
    plt.imshow(canvas)
    plt.show()

    
    
"""mapping

Args:
    zone_textBlocks (list of DOM elements): Returned object from process_zone
    factor (float): factor = image_size / actual_scanned_image_size
    usecase (int): One of following options
        1: OCR only
        2: OCR + Segmentation
        3: OCR + Segmentation (exclusive)
    iou_threshold (float): Threshold for intersection over union
    
Returns:
    map_json (json object): Final mapped result in JSON format
"""
def mapping(zone_textBlocks=None, ocr_textBlocks=None, factor=1.0, usecase=1, iou_threshold=0.05):
    # output json
    map_json = []
    
    # USECASE 1
    if(usecase==1):
        for ocr_idx,ocr_textBlock in tqdm(enumerate(ocr_textBlocks)):
            # Build json
            _textBlock_xml = {}
            _set_ocr_textBlocks = []
            _set_ocr_contents = []

            # OCR textblock coordinates
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


            ocr_textLines = ocr_textBlock.getElementsByTagName('TextLine')
            set_contents  = ""
            for ocr_textline in ocr_textLines:
                # OCR textline coordinates
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

            # Build json
            _textBlock_xml["ocr_coords"] = [list(ocr_p3), list(ocr_p4), list(ocr_p2), list(ocr_p1)]
            _textBlock_xml["ocr_texts"]  = set_contents
            map_json.append(_textBlock_xml)

            
            
    # USECASE 2 and 3
    else:
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

                if(iou >= iou_threshold):
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

    return map_json