import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, ANSWER_LIST_4_SEGPLUSBOX, SHORT_QUESTION_LIST_4_SEGPLUSBOX

from model.bbox_head.bbox_utils import get_bounding_box_from_mask

def init_SAMed2D(base_image_dir):
    with open("path_to_your_train_file.json") as f: 
        SAMed2D_train = json.load(f)

    print("SAMed2D_train(image): ", len(SAMed2D_train))
    return SAMed2D_train








class SAMed2DDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        med_seg_data="SAMed2D",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        

        self.short_question_list = SHORT_QUESTION_LIST_4_SEGPLUSBOX
        self.answer_list = ANSWER_LIST_4_SEGPLUSBOX

        self.image2mask = {}
        self.mask2class = {}
        self.data2list = {}

        self.SAMed2D_train = {}

        self.med_seg_data = med_seg_data.split("||")
        for ds in self.med_seg_data:


            SAMed2D_train = eval("init_{}".format(ds))(base_image_dir)


            self.data2list[ds] = (list(SAMed2D_train.keys()), list(SAMed2D_train.values()))  



    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.med_seg_data) - 1)   
        ds = self.med_seg_data[ds]

        if ds in ["SAMed2D"]:
            image, labels2class = self.data2list[ds]    
            idx = random.randint(0, len(image) - 1)    

            image_path = image[idx]
            label_paths_and_classes = labels2class[idx]              


            if len(label_paths_and_classes) >= self.num_classes_per_sample:
                mask_paths_and_classes = random.sample(list(label_paths_and_classes.items()), self.num_classes_per_sample)
            else:
                mask_paths_and_classes = list(label_paths_and_classes.items())   


            labels = []
            sampled_classes = []
            bboxes = []   
            for mask_path, class_text in mask_paths_and_classes:
                label = Image.open(os.path.join(self.base_image_dir, "SAMed2Dv1", mask_path))
                label = np.array(label)
                label = torch.from_numpy(label).long()

                sampled_classes.append(class_text)
                labels.append(label)
                bbox_temp = get_bounding_box_from_mask(label)
                bboxes.append(bbox_temp)
            bboxes = torch.tensor(bboxes)



            img = cv2.imread(os.path.join(self.base_image_dir, "SAMed2Dv1", image_path))
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image) 
            resize = image.shape[:2]

            


        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))



            class_ids.append(255)  


        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        
        masks = []
        for idx, class_id in enumerate(class_ids):
            if 255 not in labels[idx]:
                raise ValueError("Tensor does not contain the value 255")
            masks.append(labels[idx] == class_id)
        masks = torch.stack(masks, dim=0)
        label = labels[0]

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,   
            resize,
            questions,
            sampled_classes,
            bboxes,   
        )




