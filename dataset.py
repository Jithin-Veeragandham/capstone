import fiftyone.zoo as foz
import fiftyone as fo
dataset = foz.load_zoo_dataset(
    "coco-2017", 
    split="train", 
    label_types=["detections"], 
    classes=["person", "car", "truck", "bicycle", "motorcycle", "cat", "dog"]
)

export_dir = "C:\\Users\\jithi\\OneDrive\\Desktop\\VsCode\\coco"

# Export dataset in YOLO format
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv4Dataset,
    label_field="ground_truth",  # this should be the name of the detections field in your FiftyOne dataset
)
