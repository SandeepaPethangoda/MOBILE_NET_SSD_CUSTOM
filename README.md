# MOBILE_NET_SSD_CUSTOM

## Overview

This project implements object detection using a custom-trained MobileNet SSD model. It is designed for fast and efficient detection of multiple objects in images or video streams.

## Features

- Real-time object detection
- Customizable classes
- Lightweight and fast inference
- Easy integration with Python scripts

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Object-Detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your images or videos in the `data/` directory.
2. Run the detection script:
    ```bash
    python detect.py --input data/sample.jpg
    ```
3. View results in the `output/` directory.

## Model Training

To train your own MobileNet SSD model:
1. Prepare your dataset in VOC or COCO format.
2. Update the configuration files as needed.
3. Run the training script:
    ```bash
    python train.py --config configs/custom_config.yaml
    ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.