import torch.nn as nn
import torch.nn.functional as F
import torch
import editdistance as ed
from tqdm import tqdm
import re
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextAccuracy(nn.Module):
    def __init__(self, charset_path, case_sensitive, model_eval):
        self.charset_path = charset_path
        self.case_sensitive = case_sensitive

        self.model_eval = model_eval
        assert self.model_eval in ['vision', 'language', 'alignment']
        self._names = ['ccr', 'cwr', 'ted', 'ned', 'ted/w', 'words', 'time']

        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.
        self.total_ed = 0.
        self.total_ned = 0.
        self.inference_time = 0.

    def compute(self, model, dataloader):
        test_data_loader_iter = iter(dataloader)
        for test_iter in tqdm(range(len(test_data_loader_iter))):
            image_tensors, label_tensors = test_data_loader_iter.next()
            image_tensors = image_tensors.to(device)
            start_time = time.time()
            out_dec = model(image_tensors, text=None, return_loss=False, test_speed=False)
            label_indexes, label_scores = model.module.label_convertor.tensor2idx(out_dec)
            pt_text = model.module.label_convertor.idx2str(label_indexes)
            self.inference_time += time.time() - start_time
            gt_text = label_tensors[0]
            comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
            for i in range(len(gt_text)):
                if not self.case_sensitive:
                    gt_text_lower = gt_text[i].lower()
                    pred_text_lower = pt_text[i].lower()
                    gt_text_lower_ignore = comp.sub('', gt_text_lower)
                    pred_text_lower_ignore = comp.sub('', pred_text_lower)
                if gt_text_lower_ignore == pred_text_lower_ignore:
                    self.correct_num_word += 1

                distance = ed.eval(gt_text_lower_ignore, pred_text_lower_ignore)
                self.total_ed += distance
                self.total_ned += float(distance) / max(len(gt_text[i]), 1)
                self.total_num_word += 1

                for j in range(min(len(gt_text[i]), len(pt_text[i]))):
                    if gt_text[i][j] == pt_text[i][j]:
                        self.correct_num_char += 1
                self.total_num_char += len(gt_text[i])
        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                self.total_ed,
                self.total_ned,
                self.total_ed / self.total_num_word,
                self.total_num_word,
                self.inference_time]
        return dict(zip(self._names, mets))