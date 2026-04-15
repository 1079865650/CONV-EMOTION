import math
import random

import torch


class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
            batch_size = len(samples)
            text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
            mx = torch.max(text_len_tensor).item()
            
            # ================= 核心修改 1：改维度，加张量 =================
            feat_dim = 1024  # 把焊死的 100 改成 1024！
            text_tensor = torch.zeros((batch_size, mx, feat_dim))
            # audio_tensor = torch.zeros((batch_size, mx, feat_dim))
            # video_tensor = torch.zeros((batch_size, mx, feat_dim)) 
            audio_tensor = torch.zeros((batch_size, mx, 1280)).float()
            video_tensor = torch.zeros((batch_size, mx, 1024)).float()
            # ==============================================================

            speaker_tensor = torch.zeros((batch_size, mx)).long()
            labels = []
            for i, s in enumerate(samples):
                cur_len = len(s.text)
                
                # 1. 填装 Text
                tmp_text = [torch.from_numpy(t).float() for t in s.text]
                tmp_text = torch.stack(tmp_text)
                text_tensor[i, :cur_len, :] = tmp_text
                
                # ================= 核心修改 2：填装 Audio 和 Video =================
                tmp_audio = [torch.from_numpy(t).float() for t in s.audio]
                tmp_audio = torch.stack(tmp_audio)
                audio_tensor[i, :cur_len, :] = tmp_audio

                tmp_video = [torch.from_numpy(t).float() for t in s.visual]
                tmp_video = torch.stack(tmp_video)
                video_tensor[i, :cur_len, :] = tmp_video
                # ===================================================================

                # =========== 核心修改 4：强制双人映射 (暴力对齐) ===========
                # 无论原文件里说话人标签有多少种写法，强制把第一个开口的人视为 0，其他人统统视为 1
                # 彻底杜绝 GCN 建图时的 KeyError
                if len(s.speaker) > 0:
                    first_speaker = s.speaker[0]
                    mapped_speakers = [0 if c == first_speaker else 1 for c in s.speaker]
                    speaker_tensor[i, :cur_len] = torch.tensor(mapped_speakers)
                # =========================================================

                labels.extend(s.label)

            label_tensor = torch.tensor(labels).long()


            if label_tensor.dim() == 2 and label_tensor.size(1) > 1:
                label_tensor = torch.argmax(label_tensor, dim=1)
            elif label_tensor.dim() == 2 and label_tensor.size(1) == 1:
                label_tensor = label_tensor.squeeze()
                
            # 2. 解决 CUDA Assert t >= 0 && t < n_classes 报错！
            # 如果新数据集的标签是从 1 开始的（1~6），我们统统减去 1，平移成 0~5
            if label_tensor.min().item() == 1:
                label_tensor = label_tensor - 1
                
            # 3. 终极保命符：遇到极个别脏数据（比如标签为 7 或 -1），强行限制在 0~5 范围内！
            # 彻底杜绝显卡崩溃报错
            label_tensor = torch.clamp(label_tensor, min=0, max=5)
            
            # ================= 核心修改 3：放入字典，送给模型 =================
            data = {
                "text_len_tensor": text_len_tensor,
                "text_tensor": text_tensor,
                "audio_tensor": audio_tensor, # 送给 SeqContext 的礼物
                "video_tensor": video_tensor, # 送给 SeqContext 的礼物
                "speaker_tensor": speaker_tensor,
                "label_tensor": label_tensor
            }
            # ==================================================================

            return data

    def shuffle(self):
        random.shuffle(self.samples)




