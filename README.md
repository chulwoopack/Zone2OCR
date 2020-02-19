
# Zone2OCR
Zone2OCR is a tool for document layout analysis. This tool aims at mapping a seg of zones generated by a segmentation algorithm (e.g., dhSegment) to the regions generated by OCR engine.


## Installation
1. Clone this repository
2. Install Anaconda or Miniconda ([installation procedure](https://docs.anaconda.com/anaconda/install/]))
3. Create a virtual environment and activate it
```
conda create -n <ENV_NAME>
conda activate <ENV_NAME>
```
4. Install Zone2OCR dependencies with
```
pip install .
```
(Optional) If one wants to run the dhSegment algorithm pretrained by [ImageNet]([http://www.image-net.org/](http://www.image-net.org/)) + [Europeana Historical Newspaper]([https://www.primaresearch.org/datasets/ENP](https://www.primaresearch.org/datasets/ENP)), install dhSegment dependencies with
```
pip install ./dhsegment/.
```
and install TensorFlow 1.13 with
```
conda install tensorflow-gpu=1.13.1
```
## Usage
1. Make sure to prepare a valid file structure as below: 
(Note: all segmentation result xml files should match with OCR xml files)
```
.root
├── zone_xmls     # segmentation results
│   ├── image1.xml  
│   ├── ...
│   └── image8.xml
├── ocr_xmls      # OCR results
│   ├── image1.xml  
│   ├── ...
│   └── image8.xml
└── images        # (optional) images for visual inspection
    ├── image1.jpg  
    ├── ...
    └── image8.jpg
```
(Optional) Run pretrained dhSegment to collect segmentation result xml files
```
python run_segmentation.py -i <IMAGE_DIR> -s <SAVE_DIR> [-t <SMALL_REGION_THRESHOLD>] [-v (True|False)]
```
* `-i`: The path to the folder containing image to be processed
* `-s`: The path to the folder to store output xml file
* `-t`: (Optional) A threshold of *area(zone)/area(full_page)* ratio for ignoring small zones [0,1] (default: 0.005)
* `-v`: (Optional) Increase output verbosity (default: False) 

2. Run mapping
```
python mapping.py -zx <ZONE_XML_DIR> -ox <OCR_XML_DIR> [-t <IOU_THRESHOLD>] -s <SAVE_DIR> [-v (True|False)]
```
* `-zx`: The path to the folder containing segmentation result xml files
* `-ox`: The path to the folder containing OCR xml files
* `-t`: (Optional) A threshold of intersection over union to ignore small zones [0,1] (default: 0.1)
* `-s`: The path to the folder to store output `JSON` file
* `-v`: (Optional) Increase output verbosity (default: False) 

## Remark
* Both segmentation result and OCR XML file have to follow [PAGE XML-schema](https://www.primaresearch.org/tools/PAGELibraries)
* Output `JSON` file follows the below structure:
```
[
  {
    "zone_coord" : [
    	[x1,y1],[x2,y2],[x3,y3],[x4,y4]            // Found zone 1
    ],
    "ocr_coord" : [
      [
        [x1,y1],[x2,y2],[x3,y3],[x4,y4],           // Matched OCR zone 1
        [x1',y1'],[x2',y2'],[x3',y3'],[x4',y4'],   // Matched OCR zone 2
        ...,
      ]
    ]
    "ocr_texts" : [
      "text1",                                     // Matched OCR zone 1's text contents
      "text2",                                     // Matched OCR zone 2's text contents
      ...,
    ]
  },
  {
  	...                                            // Found zone 2
  }
]
```

## Authors
- **Chulwoo Pack** - University of Nebraska-Lincoln - _email_ - [cpack@cse.unl.edu](mailto:cpack@cse.unl.edu)
- **Benoit Seguin** and **Sofia Ares Oliveira** - DHLAB, EPFL - _git_ - [https://github.com/dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment)

## License
This project is licensed under the GPL License - see the [LICENSE](/LICENSE) file for details