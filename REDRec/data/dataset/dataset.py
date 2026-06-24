import os
import math
import json
import torch
import random
import time
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from logging import getLogger
from .utils.kv import NoteInfoCacher, UserEngageLastnCacher
from .utils.reader import read_data_from_parquet
from .utils.data_preprocess import note_content_preprocess, get_lastn_info, lastn_sort_by_ts
from transformers import AutoTokenizer, AutoProcessor
import datetime


# -----------------------------------------------------------------------------
# Schema-tolerant history adapter: normalize heterogeneous jsonl records into
# the canonical v1-style ``history`` list of ``{"cid", "ts", "label"}`` dicts
# the rest of the precomputed pipeline expects. Two recognized record shapes:
#
#   1. Flat 512-d schema (datav1):
#        {"qimei36": ..., "history": [{"cid": ..., "ts": ..., "label": "seq"|"pos"}, ...]}
#      Items already carry per-item label, so we return ``record['history']``
#      unchanged.
#
#   2. Array-form 64-d schema (datav2):
#        {"qimei36": ..., "content_ids": [c1, c2, ...], "event_times": [...],
#         "actions": [...], "durations": [...], "busi_types": [...]}
#      Items are a single time-ordered behavior log with NO per-item label and
#      no pre-split history/target. The convention is ``content_ids[:-pos_n]``
#      becomes ``label='seq'`` (input history), ``content_ids[-pos_n:]``
#      becomes ``label='pos'`` (target candidates). ts is set to 0 since the
#      pos_random_seq_input target strategy does not consult it (event_times
#      strings would only add per-row datetime parse cost without changing
#      sample selection).
#
# Naming uses ``64d`` rather than ``v2`` even though the actual discriminator
# is the schema (presence of ``history`` vs ``content_ids``), because in this
# project the array-form schema only ships with the 64-d item-pool data --
# treating ``64d`` as the user-facing label keeps yaml config ergonomic
# (``precomputed_64d_pos_n``) while still detecting the correct branch
# automatically by record shape.
#
# Returns None if neither schema is recognized or the record does not have
# enough items to split; callers should treat None as "skip this row".
# -----------------------------------------------------------------------------
def normalize_history_record(record, pos_n=10):
    history = record.get('history', None)
    if isinstance(history, list):
        return history

    content_ids = record.get('content_ids', None)
    if isinstance(content_ids, list):
        n = len(content_ids)
        if pos_n <= 0 or n <= pos_n:
            return None
        seq_part = [{'cid': c, 'ts': 0, 'label': 'seq'} for c in content_ids[:-pos_n]]
        pos_part = [{'cid': c, 'ts': 0, 'label': 'pos'} for c in content_ids[-pos_n:]]
        return seq_part + pos_part

    return None


def process_action(line):

    if line is None:
        return None

    note_id = line['note_id']
    ts = line['timestamp']
    
    like = line['is_like']
    collect = line['is_collect']
    comment = line['is_comment']
    share = line['is_share']
    # follow = line['is_follow']
    # hide = line['is_hide']
    nns = line['is_nns']
    click_profile = line['is_click_profile']
    is_engage = is_engage = (
        int(line['is_click_profile']) |
        int(line['is_collect']) |
        int(line['is_comment']) |
        int(line['is_share'])
    )

    dt = datetime.datetime.fromtimestamp(line['timestamp'])
    hour_id = dt.hour

    action_list = [share, comment, collect, like, nns]  # follow, share, comment, collect, like

    return action_list, hour_id

## retrace dataset
class REDRecDatasetNoteRetrace(torch.utils.data.IterableDataset):
    def __init__(
        self,
        global_rank, 
        world_size,
        config,
    ):
        super(REDRecDatasetNoteRetrace, self).__init__()
        self.config = config
        self.global_rank = global_rank # global_rank
        self.world_size = world_size
        
        self.note_info_root = config.eval.note_eval.parquet_note_info_root
        file_list = os.listdir(self.note_info_root)
        self.file_list = [os.path.join(self.note_info_root, per) for per in file_list if ('parquet' in per and 'temp' not in per)]
        self.file_list.sort()
        part_len = int(np.ceil(len(self.file_list) / world_size))
        
        self.startIndex = int(global_rank * part_len)
        self.endIndex = self.startIndex + part_len
        
        print(f">>> world size is: {world_size}, all parquet file nums is: {len(self.file_list)}, curr process data range: {self.startIndex}-{self.endIndex}")
        

    def __len__(self):
        return 100000000
    

    def _sample_generator(self, start_index, end_index, worker_id):
        cur_worker_selected_file_list = self.file_list[start_index:end_index]
        print(f'>>>[data processor] global_rank: {self.global_rank}, worker_id: {worker_id}, file index: {start_index}-{end_index}')


        for file_path in cur_worker_selected_file_list:
            for note_info in read_data_from_parquet(file_path, folder=False):
                try:
                    note_id = note_info['note_id']
                    note_type = note_info['type'] if note_info['type'] is not None else '1'
                    title = note_info['title'] if note_info['title'] is not None else ''
                    content = note_info['content'] if note_info['content'] is not None else ''
                    ocr = note_info['ocr'] if note_info['ocr'] is not None else ''
                    fimg_url = note_info['furl']
                    
                    msg = {
                        'note_id': note_id,
                        'note_type': note_type,
                        'title': title, 
                        'content': content, 
                        'ocr': ocr, 
                        'fimg_url': fimg_url
                    }
            
                    yield msg
                except:
                    continue


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = None
        if worker_info is None: # only one worker
            iter_start = self.startIndex
            iter_end = self.endIndex
        else:  # in a worker process
            per_worker = int(math.ceil((self.endIndex - self.startIndex + 1) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.startIndex + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.endIndex)
        sampler_iterator = self._sample_generator(iter_start, iter_end, worker_id)
        
        return sampler_iterator


# Convert item data to batched tensors
def items_to_batch(item_data_list):
    if not item_data_list:
        return [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # Find max length for padding
    max_len = max(len(item[1]) for item in item_data_list)  # item[1] is input_ids
    
    note_ids = []
    batched_input_ids = []
    batched_position_ids = []
    batched_attention_mask = []
    
    for note_id, input_ids, position_ids in item_data_list:
        note_ids.append(note_id) # hack for training
        attention_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
        
        # Pad sequences
        padded_input_ids = input_ids + [0] * (max_len - len(input_ids))
        padded_position_ids = position_ids + [0] * (max_len - len(position_ids))
        
        batched_input_ids.append(padded_input_ids)
        batched_position_ids.append(padded_position_ids)
        batched_attention_mask.append(attention_mask)
    
    return (note_ids,
            torch.tensor(batched_input_ids, dtype=torch.int64),
            torch.tensor(batched_position_ids, dtype=torch.int64),
            torch.tensor(batched_attention_mask, dtype=torch.bool))


def process_click_lastn(click_lastn, config, engage_ts_thresh=None, training=True):
    start_ts = 0
    end_ts = 100000000000
    
    # step1: process click_lastn to engage_lastn format
    new_click_lastn_ads = []
    new_click_lastn_homefeed = []
    flag = None

    for line in click_lastn['ads_click_noteid_lastn']:
        cur_action_ts = line.get('timestamp', 1)
        note_id = line['note_id']
        action_set, hour_label = process_action(line)
        duration = line.get('duration', 1)

        info = {
            'ts': cur_action_ts,
            'action': action_set,
            'note_id': note_id,
            'duration': duration,
            'hour_id': hour_label
        }
        new_click_lastn_ads.append(info)
    
    # print("for homefeed eval")
    for line in click_lastn['homefeed_noteid_lastn']:
        cur_action_ts = line.get('timestamp', 1)
        note_id = line['note_id']
        action_set, hour_label = process_action(line)
        duration = line.get('duration', 1)

        info = {
            'ts': cur_action_ts,
            'action': action_set,
            'note_id': note_id,
            'duration': duration,
            'hour_id': hour_label
        }
        new_click_lastn_homefeed.append(info)
    
    if training:
        sampled_click_lastn_ads = random.sample(new_click_lastn_ads, config.data.get('lastn_max_click_note_num_ads', 64)+10) if len(new_click_lastn_ads) > config.data.get('lastn_max_click_note_num_ads', 64)+10 else new_click_lastn_ads
        sampled_click_lastn_homefeed = random.sample(new_click_lastn_homefeed, config.data.get('lastn_max_click_note_num_homefeed', 64)+10) if len(new_click_lastn_homefeed) > config.data.get('lastn_max_click_note_num_homefeed', 64)+10 else new_click_lastn_homefeed
        return sampled_click_lastn_homefeed, sampled_click_lastn_ads

    else:
        return new_click_lastn_homefeed[-config.data.get('lastn_max_click_note_num_homefeed', 64):], new_click_lastn_ads[-config.data.get('lastn_max_click_note_num_ads', 64):]


class REDRecDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        global_rank, 
        world_size,
        config,
    ):
        super(REDRecDataset, self).__init__()
        self.config = config
        self.global_rank = global_rank # global_rank
        self.world_size = world_size
        
        self.logger = getLogger()

        # init kv
        self.trainNoteinfoCacher = NoteInfoCacher(**self.config.data.train_noteinfo_kv)
        self.engageLastnCache = UserEngageLastnCacher(**self.config.data.engage_lastn_kv)
    
        # processor
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.item_pretrain_dir, trust_remote_code=True)
        self.item_prompt = config.data.item_prompt
        
        # note content config
        self.max_text_len = config.data.max_text_len
        self.max_topic_nums = config.data.max_topic_nums
        self.max_input_token_len = config.data.max_input_token_len

        # file process
        
        # if loading from single local jsonl file:
        # self.lastn_data_jsonl = config.data.lastn_data_jsonl
        # self.total_lines = self._count_lines(self.lastn_data_jsonl)
        # lines_per_process = self.total_lines // self.world_size
        # self.start_line = self.global_rank * lines_per_process
        # self.end_line = (self.global_rank + 1) * lines_per_process if self.global_rank < self.world_size - 1 else self.total_lines

        # else from several seperated jsonl files:
        self.all_train_click_lastn_files = os.listdir(config.data.click_lastn_data_root)
        self.all_train_click_lastn_files = [os.path.join(config.data.click_lastn_data_root, per) for per in self.all_train_click_lastn_files if 'parquet' in per and 'temp' not in per]
        
        random.shuffle(self.all_train_click_lastn_files)
        
        part_len = len(self.all_train_click_lastn_files) // world_size
        self.startIndex = global_rank * part_len
        self.endIndex = self.startIndex + part_len

        # read train_lastn_note_appear_cnt_path
        if config.data.get('train_lastn_note_appear_cnt_path', False):
            with open(config.data.train_lastn_note_appear_cnt_path, 'r') as f:
                self.train_lastn_note_appear_cnt = json.load(f)
        else:
            self.train_lastn_note_appear_cnt = None
        
        self.neg_pool_size_per_gpu = config.data.get('neg_pool_size_per_gpu', 1000000)


        self.logger.info(f">>> world size is: {world_size}, lastn file nums is: {len(self.all_train_click_lastn_files)}, file nums for per process is: {part_len}")
        if self.train_lastn_note_appear_cnt is not None:
            self.logger.info('>>> apply hot decay...')
        
        self.all_cached_cnt = 0
        self.all_cnt = 0

    def __len__(self):
        return 1000000000 
    
    def process_item(self, note_id, noteid2noteinfo=None):
        # read noteinfo by noteid
        if noteid2noteinfo is None:
            try:
                note_info = self.trainNoteinfoCacher.get_note_info(note_id)
            except:
                self.logger.info(f'{note_id} get noteinfo failed.')
                return None
        else:
            note_info = noteid2noteinfo[note_id]


        fimg_url = note_info['fimg_url']

        title = note_info['title']
        content = note_info['content']
        ocr = note_info['ocr']

        title = title.strip()
        content = content.strip()
        ocr = ocr.strip()

        title = title[:22]
        ocr = ocr[:32]

        content = note_content_preprocess(content, self.max_text_len, max_topictags=self.max_topic_nums)

        text_str = f"{self.item_prompt}\ntitle:{title}\ncontent:{content}\nocr:{ocr}"
        
        return text_str


    def llama_process(self, text_str):
        ids = self.tokenizer.encode(text_str)
        ids = ids[:self.max_input_token_len]
        mask = [1] * len(ids)
        return ids, mask


    def prepare_batchdata_for_prefetch(self, raw_sequence_batch_data, num_negative=500):
        # Store individual items instead of concatenated sequences
        pos_item_data_homefeed = []  # List of (input_ids, position_ids) for each item
        pos_item_data_ads = []
        neg_item_data = []

        # Store user-level information
        pos_user_action_ads, pos_user_action_homefeed = [], []
        input_seq_len_homefeed = []
        input_seq_len_ads = []
        pos_hour_labels_homefeed, pos_hour_labels_ads = [], []

        # pos note process
        all_noteid2noteinfo = {}

        # Process homefeed data
        for user_lastn in raw_sequence_batch_data:
            user_lastn = user_lastn["cur_user_info_homefeed"]
            cur_user_seq = user_lastn['lastn_noteids']
            cur_user_action = user_lastn['actions']
            all_noteid2noteinfo.update(user_lastn['noteid2info'])

            input_seq_len_homefeed.append(len(cur_user_seq))
            pos_user_action_homefeed.append(cur_user_action)
            
            # Process each item individually
            for note_id in cur_user_seq:
                text = self.process_item(note_id, all_noteid2noteinfo)
                ids, _ = self.llama_process(text)
                item_input_ids = ids + [0]  # special token tokenizer is 0
                item_position_ids = (torch.arange(len(ids) + 1) + (self.max_input_token_len - len(ids))).tolist()
                pos_item_data_homefeed.append((" ", item_input_ids, item_position_ids))

        # Process ads data
        for user_lastn in raw_sequence_batch_data:
            user_lastn = user_lastn["cur_user_info_ads"]
            cur_user_seq = user_lastn['lastn_noteids']
            cur_user_action = user_lastn['actions']
            all_noteid2noteinfo.update(user_lastn['noteid2info'])

            input_seq_len_ads.append(len(cur_user_seq))
            pos_user_action_ads.append(cur_user_action)
            
            # Process each item individually
            for note_id in cur_user_seq:
                text = self.process_item(note_id, all_noteid2noteinfo)
                ids, _ = self.llama_process(text)
                item_input_ids = ids + [0]  # special token tokenizer is 0
                item_position_ids = (torch.arange(len(ids) + 1) + (self.max_input_token_len - len(ids))).tolist()
                pos_item_data_ads.append((" ", item_input_ids, item_position_ids))

        # Process negative samples
        neg_note_ids = random.sample(self.all_note_lists, num_negative + 200)
        neg_count = 0
        for note_id in neg_note_ids:
            text = self.process_item(note_id)
            if text is None:
                continue
            
            neg_count += 1
            ids, _ = self.llama_process(text)
            item_input_ids = ids + [0]
            item_position_ids = (torch.arange(len(ids) + 1) + (self.max_input_token_len - len(ids))).tolist()
            neg_item_data.append((" ", item_input_ids, item_position_ids))

            if neg_count >= num_negative:
                break

        # Convert to batches
        _, pos_input_ids_homefeed, pos_position_ids_homefeed, pos_attention_mask_homefeed = items_to_batch(pos_item_data_homefeed)
        _, pos_input_ids_ads, pos_position_ids_ads, pos_attention_mask_ads = items_to_batch(pos_item_data_ads)
        _, neg_input_ids, neg_position_ids, neg_attention_mask = items_to_batch(neg_item_data)

        # Pad action sequences and hour labels to consistent lengths
        def pad_sequences(sequences, pad_value=0):
            if not sequences:
                return []
            max_len = max(len(seq) for seq in sequences)
            return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

        pos_user_action_homefeed = pad_sequences(pos_user_action_homefeed)
        pos_user_action_ads = pad_sequences(pos_user_action_ads)
        pos_hour_labels_homefeed = pad_sequences(pos_hour_labels_homefeed)
        pos_hour_labels_ads = pad_sequences(pos_hour_labels_ads)

        outputs = {
            "pos_input_action_homefeed": torch.tensor(pos_user_action_homefeed, dtype=torch.int64),
            "pos_input_action_ads": torch.tensor(pos_user_action_ads, dtype=torch.int64),
            "pos_input_ids_homefeed": pos_input_ids_homefeed,
            "pos_input_ids_ads": pos_input_ids_ads,
            "pos_position_ids_homefeed": pos_position_ids_homefeed,
            "pos_position_ids_ads": pos_position_ids_ads,
            "pos_attention_mask_homefeed": pos_attention_mask_homefeed,
            "pos_attention_mask_ads": pos_attention_mask_ads,
            "neg_input_ids": neg_input_ids,
            "neg_position_ids": neg_position_ids,
            "neg_attention_mask": neg_attention_mask,
            "input_seq_len_homefeed": torch.tensor(input_seq_len_homefeed, dtype=torch.int64),
            "input_seq_len_ads": torch.tensor(input_seq_len_ads, dtype=torch.int64),
            "pos_hour_labels_homefeed": torch.tensor(pos_hour_labels_homefeed, dtype=torch.int64),
            "pos_hour_labels_ads": torch.tensor(pos_hour_labels_ads, dtype=torch.int64)
        }
        # import pdb;pdb.set_trace()
        return outputs
    

    def process_click_lastn(self, click_lastn, engage_ts_thresh=None):
        start_ts = 0
        end_ts = 100000000000
        
        # step1: process click_lastn to engage_lastn format
        new_click_lastn_ads = []
        new_click_lastn_homefeed = []
        
        for line in click_lastn['homefeed_noteid_list']:
            cur_action_ts = line.get('ts', 1)
            note_id = line['note_id']
            action_set = line.get('action_set', None)
            duration = line.get('duration', 1)

            info = {
                'ts': cur_action_ts,
                'action': [],
                'note_id': note_id,
                'duration': duration
            }
            new_click_lastn_homefeed.append(info)
        
        # sampled_click_lastn_ads = random.sample(new_click_lastn_ads, self.config.data.get('lastn_max_click_note_num_ads', 64)) if len(new_click_lastn_ads) > self.config.data.get('lastn_max_click_note_num_ads', 64) else new_click_lastn_ads
        sampled_click_lastn_homefeed = random.sample(new_click_lastn_homefeed, self.config.data.get('lastn_max_click_note_num_homefeed', 64)) if len(new_click_lastn_homefeed) > self.config.data.get('lastn_max_click_note_num_homefeed', 64) else new_click_lastn_homefeed
        
        # sampled_click_lastn_ads = sorted(sampled_click_lastn_ads, key=lambda x: x['ts'])
        sampled_click_lastn_homefeed = sorted(sampled_click_lastn_homefeed, key=lambda x: x['ts'])
        sampled_click_lastn_ads = []

        return sampled_click_lastn_homefeed, sampled_click_lastn_ads

    def _sample_generator(self, start_index, end_index, worker_id):
        cur_worker_selected_files = self.all_train_click_lastn_files[start_index:end_index]
        random.shuffle(cur_worker_selected_files)
        self.logger.info(f'>>> global_rank: {self.global_rank}, worker_id: {worker_id}, file index: {start_index}-{end_index}')
        
        # read all_note_lists
        self.all_note_lists = set()

        self.logger.info('>>> read note_ids for neg sample...')
        t1 = time.time()
        flag = None
        finish_flag = False
        if self.config.training.target == "ads": 
            flag = "ads_click_noteid_list"
        elif self.config.training.target == "homefeed": 
            flag = "homefeed_noteid_list"

        for file in cur_worker_selected_files:
            # print("len:", len(cur_worker_selected_files))
            for line in read_data_from_parquet(file, folder=False):
                # for per_note in line['note_list']:
                for per_note in line[flag]:
                    self.all_note_lists.add(per_note['note_id'])

                    if len(self.all_note_lists) >= self.neg_pool_size_per_gpu:
                        
                        finish_flag = True
                        break
                if finish_flag:
                    break
            if finish_flag:
                break 
        
        self.logger.info(f'>>> load neg pool done, cost: {int(time.time() - t1)}s, all {len(self.all_note_lists)}')
        self.all_note_lists = list(self.all_note_lists)

    
        while True:
            random.shuffle(cur_worker_selected_files)
            batch_sequence = []
            for file in cur_worker_selected_files:
                for line in read_data_from_parquet(file, folder=False):
                    click_lastn = line
                    user_id = line['user_id']
                    
                    try:
                        click_engage_lastn_homefeed, click_engage_lastn_ads = process_click_lastn(click_lastn, engage_ts_thresh=1735660800)  # from 2025-01-01
                        cur_user_info_homefeed = self.process_per_user_lastn(user_id, click_engage_lastn_homefeed, label="homefeed")
                        cur_user_info_ads = self.process_per_user_lastn(user_id, click_engage_lastn_ads, label="ads")

                        if self.config.data.get('lastn_max_click_note_num_homefeed', 64) > 0 and cur_user_info_homefeed is not None and len(cur_user_info_homefeed['lastn_noteids']) < 4:
                            continue
                        if self.config.data.get('lastn_max_click_note_num_ads', 64) > 0 and cur_user_info_ads is not None and len(cur_user_info_ads['lastn_noteids']) < 4:
                            continue
                                    
                    except:
                        self.logger.info(f'user: {user_id} info process failed')
                        error_info = traceback.format_exc()
                        print(f'error! {error_info}')
                        continue
                    
                    if cur_user_info_homefeed is None or cur_user_info_ads is None:
                        continue
                    else:
                        batch_sequence.append({
                            "cur_user_info_ads": cur_user_info_ads,
                            "cur_user_info_homefeed": cur_user_info_homefeed
                        })

                    if len(batch_sequence) == self.config.data.train_batch_size:
                        try:
                            train_batch_data = self.prepare_batchdata_for_prefetch(batch_sequence, self.config.data.get('neg_samples_per_gpu', 500))
                            yield train_batch_data
                        except:
                            print(f'prepare batch_data for train failed, error info: {traceback.format_exc()}')
                        finally:
                            batch_sequence = []
    

    def process_per_user_lastn(self, user_id, lastn, label=None):

        if label == None:
            seq_len = self.config.data.get('seq_len', 64)
        elif label == "homefeed":
            seq_len = self.config.data.get('lastn_max_click_note_num_homefeed', 64)
        elif label == "ads":
            seq_len = self.config.data.get('lastn_max_click_note_num_ads', 64)
        
        lastn_noteids, actions, ts = get_lastn_info(lastn, self.train_lastn_note_appear_cnt)

        # trunction for get note info
        lastn_noteids = lastn_noteids[-(seq_len + 10):]
        actions = actions[-(seq_len + 10):]
        ts = ts[-(seq_len + 10):]

        # download note_info
        note_id2noteinfo = {}
        for note_id in lastn_noteids:
            try:
                note_info = self.trainNoteinfoCacher.get_note_info(note_id)
            except:
                self.logger.info(f'{note_id} get noteinfo failed.')
                continue

            if note_info is not None:
                note_id2noteinfo[note_id] = note_info
        
        keep_index = []
        for idx, note_id in enumerate(lastn_noteids):
            if note_id in note_id2noteinfo:
                keep_index.append(idx)
        
        new_lastn_noteids = []
        new_actions = []

        for idx in keep_index:
            new_lastn_noteids.append(lastn_noteids[idx])
            new_actions.append(actions[idx])

        # trunction
        new_lastn_noteids = new_lastn_noteids[-seq_len:]
        new_actions = new_actions[-seq_len:]

        ret = {
            'user_id': user_id,
            'lastn_noteids': new_lastn_noteids,
            'actions': new_actions,
            'noteid2info': note_id2noteinfo
        }

        return ret


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = None
        if worker_info is None: # only one worker
            iter_start = self.startIndex
            iter_end = self.endIndex
        else:  # in a worker process
            per_worker = int(math.ceil((self.endIndex - self.startIndex + 1) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.startIndex + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.endIndex)
        sampler_iterator = self._sample_generator(iter_start, iter_end, worker_id)
        
        return sampler_iterator


class REDRecPrecomputedEmbeddingDataset(torch.utils.data.IterableDataset):
    """Train REDRec user tower from local user histories and precomputed item embeddings.

    Input jsonl format:
        {"qimei36": "...", "history": [{"cid": "...", "ts": 123, "label": "pos"}, ...]}

    Precomputed embedding npy format:
        {"cid": np.ndarray[str], "embedding": np.ndarray[float32]}
    """

    def __init__(self, global_rank, world_size, config):
        super(REDRecPrecomputedEmbeddingDataset, self).__init__()
        self.config = config
        self.global_rank = int(global_rank or 0)
        self.world_size = int(world_size or 1)
        self.logger = getLogger()

        self.user_history_jsonl = config.data.precomputed_user_history_jsonl
        # Embedding source: either a single .npy (legacy / gold-standard format
        # produced by `np.save({'cid': ..., 'embedding': ...})`) OR a
        # preprocessed memmap directory (cids.npy + embeddings.bin + meta.json,
        # shuffle-data format). The latter avoids loading the whole 6+GB
        # embedding tensor into RAM and supports much larger item pools.
        self.embedding_npy = config.data.get('precomputed_embedding_npy', None)
        self.embedding_dir = config.data.get('precomputed_embedding_dir', None)
        if not self.embedding_npy and not self.embedding_dir:
            raise ValueError(
                'Neither precomputed_embedding_npy nor precomputed_embedding_dir '
                'is set in config.data. One of them is required.'
            )
        self.max_seq_len = config.data.get('lastn_max_click_note_num_homefeed', 96)
        self.train_batch_size = config.data.get('train_batch_size', 1)
        self.neg_samples_per_gpu = config.data.get('neg_samples_per_gpu', 400)
        self.min_history_len = config.data.get('precomputed_min_history_len', 4)
        self.positive_label = config.data.get('precomputed_positive_label', 'pos')
        self.query_nums = int(config.model.get('query_nums', 1))
        self.window_size = int(config.model.get('window_pos', self.query_nums if self.query_nums > 0 else 1))
        # simple_target_strategy:
        #   'window' (default, official): take the last `window_size` positives as targets,
        #            the rest as input history.
        #   'random': randomly pick 1 positive as target, the rest as input history.
        #            Greatly increases per-user sample diversity (epoch-wise data augmentation).
        #   'pos_random_seq_input': use items with label==seq_label as input history
        #            (kept in original order, NOT sorted by ts); randomly pick 1 item
        #            with label==positive_label as the target. Designed for V0-style
        #            data where 'seq' = browsing history and 'pos' = ground-truth
        #            target click. target_window_size is forced to 1.
        self.target_strategy = str(config.data.get('simple_target_strategy', 'window'))
        # Input-side label for 'pos_random_seq_input' mode (only used when
        # target_strategy == 'pos_random_seq_input'); defaults to 'seq' which
        # matches the V0 jsonl convention.
        self.input_seq_label = str(config.data.get('precomputed_input_seq_label', 'seq'))
        if self.target_strategy in ('random', 'pos_random_seq_input'):
            self.target_window_size = 1
        else:
            self.target_window_size = self.window_size
        self.context_max_len = max(1, self.max_seq_len - self.target_window_size)
        self.sample_lastn = config.data.get('precomputed_sample_lastn', True)
        self.loop_forever = config.data.get('precomputed_loop_forever', True)
        self.max_lines = config.data.get('precomputed_max_lines', None)
        self.stats_interval = config.data.get('precomputed_stats_interval', 10000)
        # How many trailing items in a 64-d (array-form) record to mark as
        # 'pos' targets vs 'seq' history. Ignored when the input jsonl uses
        # the flat (datav1) schema with per-item labels. Defaults to 10 to
        # match the upstream split convention; see normalize_history_record
        # at the top of this module for the full schema-detection rule.
        self.pos_n_for_64d = int(config.data.get('precomputed_64d_pos_n', 10))

        if self.embedding_dir is not None:
            # ---- memmap mode (shuffle-data format) ----
            # Layout under <embedding_dir>:
            #   cids.npy        (1-D, int64 or str, num_items)
            #   embeddings.bin  (raw, num_items * embed_dim * dtype)
            #   meta.json       ({"num_items":..., "embed_dim":..., "dtype":...})
            self.logger.info(
                f'>>> load precomputed item embeddings from dir (memmap mode): {self.embedding_dir}'
            )
            with open(os.path.join(self.embedding_dir, 'meta.json'), 'r') as mf:
                meta = json.load(mf)
            num_items = int(meta['num_items'])
            embed_dim = int(meta['embed_dim'])
            dtype = np.dtype(meta.get('dtype', 'float32'))

            cids_path = os.path.join(self.embedding_dir, 'cids.npy')
            cids_raw = np.load(cids_path)
            # Stringify cids so the dataset's cid2idx / jsonl-string lookup
            # path stays identical to the gold-standard single-npy mode
            # (jsonl history stores cids as decimal strings).
            self.cids = cids_raw.astype(str)

            emb_bin = os.path.join(self.embedding_dir, 'embeddings.bin')
            self.embeddings = np.memmap(
                emb_bin, dtype=dtype, mode='r', shape=(num_items, embed_dim)
            )
            self.cid2idx = {cid: idx for idx, cid in enumerate(self.cids)}
            self.embed_dim = embed_dim
            self.all_indices = np.arange(num_items, dtype=np.int64)
            self.logger.info(
                f'>>> precomputed embeddings loaded: num={num_items}, dim={self.embed_dim} (memmap)'
            )
        else:
            # ---- single-npy mode (legacy / gold-standard format) ----
            self.logger.info(f'>>> load precomputed item embeddings from: {self.embedding_npy}')
            data = np.load(self.embedding_npy, allow_pickle=True).item()
            self.cids = np.asarray(data['cid']).astype(str)
            self.embeddings = np.asarray(data['embedding'], dtype=np.float32)
            self.cid2idx = {cid: idx for idx, cid in enumerate(self.cids)}
            self.embed_dim = int(self.embeddings.shape[1])
            self.all_indices = np.arange(len(self.cids), dtype=np.int64)
            self.logger.info(f'>>> precomputed embeddings loaded: num={len(self.cids)}, dim={self.embed_dim}')

    def __len__(self):
        return 1000000000

    def _cid_exists(self, cid):
        return cid in self.cid2idx

    def _get_embedding(self, cid):
        return self.embeddings[self.cid2idx[cid]]

    def _sample_negative_embeddings(self):
        replace = len(self.all_indices) < self.neg_samples_per_gpu
        neg_indices = np.random.choice(self.all_indices, size=self.neg_samples_per_gpu, replace=replace)
        return self.embeddings[neg_indices]

    def _select_training_window(self, history):
        """Build one official-style window-NCE sample from positive clicks.

        Official training uses click lastn as positive behavior sequence, samples a
        lastn window when the sequence is longer than the configured maximum, sorts
        the sampled window by timestamp, and lets model.window_pos define the target
        window.  Non-positive records and cids without precomputed embeddings are
        dropped before this step.
        """
        # ---- pos_random_seq_input: input <- 'seq' items, target <- random 'pos' ----
        # Designed for V0-style jsonl where 'seq' is the browsing history and
        # 'pos' is the ground-truth target click. We do NOT sort by ts (many
        # seq events have ts=0 and the upstream feature builder already keeps
        # them in the intended order).
        if self.target_strategy == 'pos_random_seq_input':
            seq_cids, pos_cids = [], []
            for item in history:
                cid = item.get('cid')
                if cid is None:
                    continue
                cid = str(cid)
                if not self._cid_exists(cid):
                    continue
                lbl = item.get('label')
                if lbl == self.input_seq_label:
                    seq_cids.append(cid)
                elif lbl == self.positive_label:
                    pos_cids.append(cid)
            if len(seq_cids) < self.min_history_len or not pos_cids:
                return None, None
            # Cap input length and pick exactly one random target from pos.
            input_cids = seq_cids[-self.context_max_len:] if self.sample_lastn else seq_cids[:self.context_max_len]
            target_cids = [random.choice(pos_cids)]
            if len(input_cids) < self.min_history_len or len(target_cids) != self.target_window_size:
                return None, None
            return input_cids, target_cids

        pos_items = []
        for item in history:
            cid = item.get('cid')
            if cid is None:
                continue
            if item.get('label') != self.positive_label:
                continue
            cid = str(cid)
            if not self._cid_exists(cid):
                continue
            pos_items.append({'cid': cid, 'ts': item.get('ts', 0)})

        pos_items = sorted(pos_items, key=lambda x: x['ts'])
        if self.sample_lastn and len(pos_items) > self.max_seq_len:
            pos_items = random.sample(pos_items, self.max_seq_len)
            pos_items = sorted(pos_items, key=lambda x: x['ts'])
        else:
            pos_items = pos_items[-self.max_seq_len:]

        if len(pos_items) < self.min_history_len + self.target_window_size:
            return None, None

        if self.target_strategy == 'random':
            # Randomly pick 1 positive as target, the rest (kept in chronological order) as input.
            # Effectively augments per-user samples by N (= number of positives).
            pick = random.randrange(len(pos_items))
            target_items = [pos_items[pick]]
            input_items = pos_items[:pick] + pos_items[pick + 1:]
            input_cids = [it['cid'] for it in input_items][-self.context_max_len:]
            target_cids = [it['cid'] for it in target_items]
            if len(input_cids) < self.min_history_len or len(target_cids) != self.target_window_size:
                return None, None
            return input_cids, target_cids

        pos_cids = [item['cid'] for item in pos_items]
        target_cids = pos_cids[-self.target_window_size:]
        input_cids = pos_cids[:-self.target_window_size]

        if len(input_cids) < self.min_history_len or len(target_cids) != self.target_window_size:
            return None, None
        return input_cids, target_cids

    def _build_batch(self, samples):
        batch_size = len(samples)
        input_embeds = np.zeros((batch_size, self.context_max_len, self.embed_dim), dtype=np.float32)
        attention_mask = np.zeros((batch_size, self.context_max_len), dtype=np.int64)
        target_embeds = np.zeros((batch_size, self.target_window_size, self.embed_dim), dtype=np.float32)
        target_mask = np.zeros((batch_size, self.target_window_size), dtype=np.int64)
        user_ids = []
        target_cids = []

        for row, sample in enumerate(samples):
            user_ids.append(sample['user_id'])
            target_cids.append(sample['target_cids'])
            seq_cids = sample['input_cids'][-self.context_max_len:]
            start = self.context_max_len - len(seq_cids)
            for offset, cid in enumerate(seq_cids):
                input_embeds[row, start + offset] = self._get_embedding(cid)
                attention_mask[row, start + offset] = 1

            cur_target_cids = sample['target_cids'][-self.target_window_size:]
            for offset, cid in enumerate(cur_target_cids):
                target_embeds[row, offset] = self._get_embedding(cid)
                target_mask[row, offset] = 1

        neg_embeds = self._sample_negative_embeddings()
        return {
            'precomputed_input_embeds': torch.from_numpy(input_embeds),
            'precomputed_attention_mask': torch.from_numpy(attention_mask),
            'precomputed_target_embeds': torch.from_numpy(target_embeds),
            'precomputed_target_mask': torch.from_numpy(target_mask),
            'precomputed_neg_embeds': torch.from_numpy(neg_embeds),
            'user_ids': user_ids,
            'target_cids': target_cids,
        }

    def _sample_generator(self, worker_id):
        self.logger.info(
            f'>>> precomputed dataset rank={self.global_rank}/{self.world_size}, worker_id={worker_id}, jsonl={self.user_history_jsonl}'
        )
        emitted = 0
        skipped = 0

        while True:
            batch_samples = []
            with open(self.user_history_jsonl, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f):
                    if self.max_lines is not None and line_no >= self.max_lines:
                        break
                    if line_no % self.world_size != self.global_rank:
                        continue
                    if worker_id is not None:
                        worker_info = torch.utils.data.get_worker_info()
                        if worker_info is not None and (line_no // self.world_size) % worker_info.num_workers != worker_id:
                            continue

                    try:
                        record = json.loads(line)
                        history = normalize_history_record(record, pos_n=self.pos_n_for_64d)
                        if history is None:
                            skipped += 1
                            continue
                        input_cids, target_cids = self._select_training_window(history)
                        if input_cids is None or target_cids is None:
                            skipped += 1
                            continue

                        batch_samples.append({
                            'user_id': record.get('qimei36', ''),
                            'input_cids': input_cids,
                            'target_cids': target_cids,
                        })
                    except Exception:
                        skipped += 1
                        continue

                    if len(batch_samples) >= self.train_batch_size:
                        emitted += len(batch_samples)
                        if self.stats_interval and emitted % self.stats_interval == 0:
                            self.logger.info(f'>>> precomputed dataset emitted={emitted}, skipped={skipped}')
                        yield self._build_batch(batch_samples)
                        batch_samples = []

            if batch_samples:
                emitted += len(batch_samples)
                yield self._build_batch(batch_samples)

            if not self.loop_forever:
                break

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else None
        return self._sample_generator(worker_id)

    


# ----------------------------------------------------------------
# ------------------------- eval dataset -------------------------
# ----------------------------------------------------------------

def process_item_for_note_inference(note_info, max_text_len, max_topic_nums, item_prompt):
    
    # read noteinfo by noteid
    noteid = note_info['note_id']
    fimg_url = note_info['fimg_url']
    title = note_info['title']
    content = note_info['content']
    ocr = note_info['ocr']

    title = title.strip()
    content = content.strip()
    ocr = ocr.strip()

    title = title[:22]


    ocr = ocr[:32]


    content = note_content_preprocess(content, max_text_len, max_topictags=max_topic_nums)

    text_str = f"{item_prompt}\ntitle:{title}\ncontent:{content}\nocr:{ocr}"
    
    return text_str


def llama_process_for_note_inference(text_str, tokenizer, max_input_token_len):
    ids = tokenizer.encode(text_str)
    ids = ids[:max_input_token_len]
    mask = [1] * len(ids)
    return ids, mask


def prepare_batchdata_for_note_inference(
        noteinfos,
        tokenizer, 
        max_text_len, 
        max_topic_nums, 
        max_input_token_len, 
        item_prompt
):
    # Store individual items instead of concatenated sequences
    item_data = []  # List of (note_id, input_ids, position_ids) for each item
    
    for note_info in noteinfos:
        try:
            note_id = note_info['note_id']
            text = process_item_for_note_inference(note_info, max_text_len, max_topic_nums, item_prompt)
            
            ids, _ = llama_process_for_note_inference(text, tokenizer, max_input_token_len)
            item_input_ids = ids + [0]  # special token tokenizer is 0
            item_position_ids = (torch.arange(len(ids) + 1) + (max_input_token_len - len(ids))).tolist()
            
            item_data.append((note_id, item_input_ids, item_position_ids))
            
        except Exception as e:
            print(f'>>> get info error for note {note_info.get("note_id", "unknown")}: {e}')
            continue

    # Convert to batches
    note_ids, pos_input_ids, pos_position_ids, pos_attention_mask = items_to_batch(item_data)
    
    outputs = {
        "note_ids": note_ids,
        "pos_input_ids": pos_input_ids,           # [batch_size, seq_len]
        "pos_position_ids": pos_position_ids,     # [batch_size, seq_len]
        "pos_attention_mask": pos_attention_mask, # [batch_size, seq_len]
    }
    return outputs



class REDRecEvalItemDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        global_rank, 
        world_size,
        config,
        lastn_note=False,
    ):
        super(REDRecEvalItemDataset, self).__init__()
        self.config = config
        self.global_rank = global_rank # global_rank
        self.world_size = world_size

        # file process
        
        if lastn_note:
            with open(config.eval.note_eval.lastn_note_info_path, 'r') as f:
                all_note_info = json.load(f)
        else:
            with open(config.eval.note_eval.basepool_note_info_path, 'r') as f:
                all_note_info = json.load(f)

        if world_size <= 1:
            self.note_infos = all_note_info
        else:
            split_note_infos = np.array_split(all_note_info, world_size)
            self.note_infos = split_note_infos[global_rank].tolist()
        
        print(f'>>> world_size: {world_size}, rank: {global_rank}, all note num is: {len(all_note_info)}, cur process note nums: {len(self.note_infos)}')

    def __len__(self):
        return len(self.note_infos) 
    
    def __getitem__(self, idx):
        note_info = self.note_infos[idx]
        return note_info

class REDRecEvalUserDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(REDRecEvalUserDataset, self).__init__()

        user_lastn_path = config.eval.user_eval.user_lastn_path
        with open(user_lastn_path, 'r') as f:
            user_lastn = json.load(f)

        self.config = config

        self.lines = []
        for user_id in user_lastn:
            cur_user_info = user_lastn[user_id]
            click_engage_lastn_homefeed, click_engage_lastn_ads = process_click_lastn(cur_user_info, self.config, engage_ts_thresh=1735660800, training=False) 
            self.lines.append(
                {
                    'user_id': user_id,
                    'homefeed_lastn': click_engage_lastn_homefeed,
                    "ads_lastn": click_engage_lastn_ads,
                    "target": [n["note_id"] for n in cur_user_info["ads_click_noteid_target"]]
                }
            )
            # for debug: only run 1000 user
            # if len(self.lines) > 1000:
            #     break
        
        self.max_lastn_len = config.eval.user_eval.max_lastn_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sample = self.lines[idx]
        user_id = sample['user_id']

        homefeed_lastn = sample.get('homefeed_lastn', [])
        ads_lastn = sample.get('ads_lastn', [])

        note_seq_homefeed = [item['note_id'] for item in homefeed_lastn]
        note_seq_ads = [item['note_id'] for item in ads_lastn]

        action_seq_homefeed = [item['action'] for item in homefeed_lastn]
        action_seq_ads = [item['action'] for item in ads_lastn]

        hour_labels_homefeed = [item['hour_id'] for item in homefeed_lastn]
        hour_labels_ads = [item['hour_id'] for item in ads_lastn]

        duration_homefeed = [item['duration'] for item in homefeed_lastn]
        duration_ads = [item['duration'] for item in ads_lastn]

        ts_homefeed = [item['ts'] for item in homefeed_lastn]
        ts_ads = [item['ts'] for item in ads_lastn]

        target_note = sample["target"]

        ret = {
            'user_id': user_id,
            'note_seq_homefeed': note_seq_homefeed,
            'note_seq_ads': note_seq_ads,
            'action_seq_homefeed': action_seq_homefeed, 
            'action_seq_ads': action_seq_ads, 
            'hour_labels_homefeed': hour_labels_homefeed,
            'hour_labels_ads': hour_labels_ads,
            'duration_homefeed': duration_homefeed,
            'duration_ads': duration_ads,
            'ts_homefeed': ts_homefeed,
            'ts_ads': ts_ads,
            'target_note': target_note
        }
        return ret
    
def user_dataset_collator(samples):
    user_ids = [per['user_id'] for per in samples]
    note_seqs_homefeed = [per['note_seq_homefeed'] for per in samples]
    note_seqs_ads = [per['note_seq_ads'] for per in samples]
    action_seqs_homefeed = [per['action_seq_homefeed'] for per in samples]
    action_seqs_ads = [per['action_seq_ads'] for per in samples]
    hour_labels_homefeed = [per['hour_labels_homefeed'] for per in samples]
    hour_labels_ads = [per['hour_labels_ads'] for per in samples]
    duration_homefeed = [per['duration_homefeed'] for per in samples]
    duration_ads = [per['duration_ads'] for per in samples]
    ts_homefeed = [per['ts_homefeed'] for per in samples]
    ts_ads = [per['ts_ads'] for per in samples]
    target_notes = [per['target_note'] for per in samples]

    ret = {
        'user_ids': user_ids,
        'note_seqs_homefeed': note_seqs_homefeed,
        'note_seqs_ads': note_seqs_ads,
        'action_seqs_homefeed': action_seqs_homefeed,
        'action_seqs_ads': action_seqs_ads,
        'hour_labels_homefeed': hour_labels_homefeed,
        'hour_labels_ads': hour_labels_ads,
        'duration_homefeed': duration_homefeed,
        'duration_ads': duration_ads,
        'ts_homefeed': ts_homefeed,
        'ts_ads': ts_ads,
        'target_notes': target_notes
    }
    return ret


if __name__ == '__main__':
    from functools import partial
    import yaml
    from easydict import EasyDict as edict


    config_path = 'config/demo_multiscene.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    # note inference dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model.item_pretrain_dir, trust_remote_code=True)
    max_text_len = config.data.max_text_len
    max_topic_nums = 12
    max_input_token_len = config.data.max_input_token_len
    item_prompt = config.data.item_prompt

    dataset = REDRecEvalItemDataset(0, 1, config)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=partial(
                                                  prepare_batchdata_for_note_inference, 
                                                  tokenizer=tokenizer, 
                                                  max_text_len=max_text_len, 
                                                  max_topic_nums=max_topic_nums, 
                                                  max_input_token_len=max_input_token_len, 
                                                  item_prompt=item_prompt
                                                )
                                              )

    for sample in tqdm(dataloader):        
        import pdb; pdb.set_trace()
        # print('>>>')
    """


    dataset = REDRecEvalUserDataset(config)
    dataloader = torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=0,
                                                collate_fn=user_dataset_collator,
                                            )
    
    for sample in tqdm(dataloader):
        import pdb; pdb.set_trace()
        # print('>>>')
