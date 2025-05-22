import os
import os.path as osp
import json
from PIL import Image
import numpy as np
import pandas as pd
import logging

from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.mlvu import *

FAIL_MSG = 'Failed to obtain answer via API.'

class Halludataset(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    def __init__(self, dataset='Halludataset',
                 json_file='/home/ubuntu/LMUData/hallucination/hallucinations_question.json',
                 frame_root='/home/ubuntu/LMUData/hallucination/valid',
                 dataset_path = '/home/ubuntu/LMUData/hallucination',
                 nframe=0,
                 fps=-1,
                 pack=False):
        self.json_file = json_file
        self.frame_root = frame_root
        self.dataset_path = dataset_path

        super().__init__(dataset=dataset, nframe=nframe, fps=fps, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['Halludataset']

    def prepare_dataset(self, dataset_name='Halludataset'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['video'])):
                    return False
            return True

        try:
            with open(self.json_file, 'r') as f:
                ori_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {self.json_file}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {self.json_file}")
            raise

        if self.dataset_path is not None and check_integrity(self.dataset_path):
            dataset_path = self.dataset_path
        else:
            self.data_list = []
            data_file = osp.join(self.dataset_path, f'{dataset_name}.tsv')
            for index in ori_data:
                data = ori_data[index]
                frames_dir = osp.join(self.frame_root, index)
                files = os.listdir(frames_dir)
                n_frames = len(files)
                for task_type in data:
                    questions = data[task_type]
                    if len(questions) > 0:
                        for question in questions:
                            self.data_list.append({
                                'video': index,
                                'task_type': task_type,
                                'question': question,
                                'options': ['Yes', 'No'],
                                'answer': 'No',
                                'n_frames': n_frames,
                            })
            data_df = pd.DataFrame(self.data_list[:10])
            data_df = data_df.assign(index=range(len(data_df)))
            data_df.to_csv(data_file, sep='\t', index=False)
            dataset_path = self.dataset_path

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            video = self.videos[idx]
            sub_data = [item for item in self.data if item['video'] == video]
            return sub_data
        else:
            return self.data[idx]

    def build_prompt(self, idx, video_llm):
        if self.pack:
            raise NotImplementedError
        else:
            item = self.data.iloc[idx]
            return self._build_single_prompt(item, video_llm)


    def _build_single_prompt(self, item, video_llm):
        question = f"Question: {item['question']}\n"
        question += "Options:\n"
        question += item['options'].replace('[','').replace(']','')
        question = question.rstrip()

        message = [
            {"type": "text", "value": "Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.", "role": "system"},
            {"type": "text", "value": question},
        ]

        video = item['video']

        folder_path = osp.join(self.frame_root, video)
        files = os.listdir(folder_path)
        paths = [osp.join(folder_path, f) for f in files if f.lower().endswith('.jpg')]
        paths.sort()
        if self.nframe > 0:
            indices = np.linspace(0, len(paths) - 1, self.nframe).astype(int)
            frame_paths = [paths[i] for i in indices]
        elif self.fps > 0:
            total_frames = item['n_frames']
            video_fps = 20
            total_duration = total_frames / video_fps
            required_frames = int(total_duration * self.fps)
            indices = np.linspace(0, len(paths) - 1, num=required_frames).astype(int)
            frame_paths = [paths[i] for i in indices]
        else:
            frame_paths = paths

        if video_llm:
            for frame_path in frame_paths:
                message.append(dict(type="image", value=frame_path))
            # message.append(dict(type="video", value=frame_paths))
        else:
            for frame_path in frame_paths:
                message.append(dict(type="image", value=frame_path))

        message.append({"type": "text", "value": "\nOnly give the best option."})
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.setdefault('model', 'chatgpt-0125')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                options = eval(data.loc[data['index'] == idx, 'options'].values[0])
                answer_idx = -1
                for id, c in enumerate(options):
                    if c == ans:
                        answer_idx = id
                ans = f"({chr(ord('A') + answer_idx)}) {ans}"
                input_item = data.loc[data['index'] == idx].to_dict(orient='records')[0]
                for id, option_content in enumerate(eval(input_item['options'])):
                    input_item[chr(ord('A') + id)] = option_content
                    if option_content == input_item['answer']:
                        input_item['answer'] = chr(ord('A') + id)

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans_with_model(
                        pred, ans, model,
                        input_item,
                        'Halludataset'
                    ))

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating