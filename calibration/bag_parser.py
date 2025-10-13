import os
import cv2
import numpy as np
import random
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from tqdm import tqdm

def parse_random_images_from_bag(bag_path, topics, num_images=20, input_dir=""):
    """
    Parse a ROS2 bag file, extract a specified number of random compressed images from given topics,
    decode them, and save to temporary files, returning a list of file paths.
    
    Args:
        bag_path (str): Path to the ROS2 bag file directory
        topics (dict): Dictionary mapping topic names to their output directories
        num_images (int): Number of images to randomly select from each topic
        output_base_dir (str): Base directory to save temporary images
    
    Returns:
        list: List of file paths to the saved images
    """
    # Ensure bag_path exists
    if not os.path.exists(bag_path):
        print(f"Error: Bag file path {bag_path} does not exist")
        return []

    # Create base output directory
    output_base_dir = os.path.join(input_dir, 'ros2_bag_calibration_dataset')

    # Dictionary to store images (topic -> list of (timestamp, image_data))
    topic_images = {topic: [] for topic in topics}

    # Open the bag file
    with Reader(bag_path) as reader:
        # Check if topics exist in the bag
        available_topics = reader.topics.keys()
        for topic in topics:
            if topic not in available_topics:
                print(f"Warning: Topic {topic} not found in bag file")
                continue

        # Count total messages for progress bar
        total_messages = sum(reader.topics[topic].msgcount for topic in topics if topic in reader.topics)
        progress_bar = tqdm(total=total_messages, desc="Processing ROS2 bag messages", unit="msg")

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in topics:
                try:
                    # Deserialize the message (assuming sensor_msgs/CompressedImage)
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    
                    # Decode the compressed image
                    img_data = np.frombuffer(msg.data, np.uint8)
                    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if image is None:
                        print(f"Error decoding image from {connection.topic} at timestamp {timestamp}")
                        progress_bar.update(1)
                        continue
                    
                    # Store the image data with timestamp
                    topic_images[connection.topic].append((timestamp, image))
                    
                except Exception as e:
                    print(f"Error processing message from {connection.topic}: {e}")
                progress_bar.update(1)
        progress_bar.close()

    print("")

    image_paths = []

    for topic, images in topic_images.items():
        # Randomly select up to num_images images
        selected_images = random.sample(images, min(num_images, len(images)))

        # Save selected images
        for idx, (timestamp, image) in enumerate(selected_images):
            filepath = os.path.join(output_base_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(filepath, image)
            image_paths.append(filepath)
            print(f"Saved image from {topic} to {filepath}")
        
        print("")
    
    return image_paths
