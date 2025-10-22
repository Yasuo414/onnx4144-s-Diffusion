import json
import logging
import os
import random
from copy import deepcopy
import librosa

import numpy as np
import torch
import yaml
from resemblyzer import VoiceEncoder
from tqdm import tqdm

from infer_tools.f0_static import static_f0_time
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from preprocessing.hubertinfer import HubertEncoder
from preprocessing.process_pipeline import File2Batch
from preprocessing.process_pipeline import get_pitch_parselmouth, get_pitch_crepe, get_pitch_rmvpe, f0_to_coarse
from utils.hparams import hparams
from utils.hparams import set_hparams
from utils.audio import get_mel_torch
from utils.indexed_datasets import IndexedDatasetBuilder

os.environ["OMP_NUM_THREADS"] = "1"
BASE_ITEM_ATTRIBUTES = ['wav_fn', 'spk_id']


class SvcBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''

    def __init__(self, data_dir=None, item_attributes=None):
        self.spk_map = None
        self.vocoder = NsfHifiGAN()
        self.phone_encoder = HubertEncoder(pt_path=hparams['hubert_path'])
        if item_attributes is None:
            item_attributes = BASE_ITEM_ATTRIBUTES
        if data_dir is None:
            data_dir = hparams['raw_data_dir']
        if 'speakers' not in hparams:
            speakers = hparams['datasets']
            hparams['speakers'] = hparams['datasets']
        else:
            speakers = hparams['speakers']
        assert isinstance(speakers, list), 'Speakers must be a list'
        assert len(speakers) == len(set(speakers)), 'Speakers cannot contain duplicate names'

        self.raw_data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        assert len(speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'
        self.speakers = speakers
        self.binarization_args = hparams['binarization_args']
        self.augmentation_args = self.binarization_args["augmentation_args"]

        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        # load each dataset
        for ds_id, data_dir in enumerate(self.raw_data_dirs):
            self.load_meta_data(data_dir, ds_id)
            if ds_id == 0:
                # check program correctness
                assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))

        if self.binarization_args['shuffle']:
            random.seed(hparams['seed'])
            random.shuffle(self.item_names)

        # set default get_pitch algorithm
        if hparams['use_crepe']:
            self.get_pitch_algorithm = get_pitch_crepe
        elif hparams["use_rmvpe"]:
            self.get_pitch_algorithm = get_pitch_rmvpe
        elif hparams["use_parselmouth"]:
            self.get_pitch_algorithm = get_pitch_parselmouth
        else:
            logging.warning("Pitch extractor is required for data preprocessing. RMVPE, TorchCrepe, and ParselMouth are supported. Please select one of them.")

        print('Speakers: ', set(self.speakers))
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    def split_train_test_set(self, item_names):
        train_item_names = []
        test_item_names = []
        train_item_names_set = set(deepcopy(item_names))

        if hparams['choose_test_manually']:
            manual_test_items = set([str(pr) for pr in hparams['test_prefixes']])
            test_item_names_set = set()
            
            for name in item_names:
                try:
                    current_speaker = None
                    for spk in self.speakers:
                        normalized_name = name.replace('\\', '/')
                        if f"/{spk}/" in normalized_name:
                            current_speaker = spk
                            break
                    
                    if not current_speaker:
                        continue
                    
                    base_filename = os.path.basename(name)
                    filename_no_ext = os.path.splitext(base_filename)[0]

                    reconstructed_key = f"{current_speaker}:{filename_no_ext}"

                    if reconstructed_key in manual_test_items:
                        test_item_names_set.add(name)
                except Exception as e:
                    logging.warning(f"Error parsing item name {name}: {e}")
                    continue

            train_item_names_set.difference_update(test_item_names_set)

            if len(test_item_names_set) != len(manual_test_items):
                found_items_keys = set()
                for name in test_item_names_set:
                    for spk in self.speakers:
                        if f"/{spk}/" in name.replace('\\', '/'):
                            base = os.path.splitext(os.path.basename(name))[0]
                            found_items_keys.add(f"{spk}:{base}")
                
                missing_items = manual_test_items - found_items_keys
                if missing_items:
                    logging.warning(f"Some files from 'test_prefixes' were not found! Missing: {sorted(list(missing_items))}")
            
            test_item_names = sorted(list(test_item_names_set))
            train_item_names = sorted(list(train_item_names_set))

        else:
            auto_test = item_names[-5:]
            test_item_names = auto_test
            train_item_names_set.difference_update(test_item_names)
            train_item_names = sorted(list(train_item_names_set))
        
        logging.info(f"Total items: {len(item_names)}")
        logging.info(f"Training items: {len(train_item_names)}")
        logging.info(f"Test items: {len(test_item_names)}")

        if not test_item_names and hparams['choose_test_manually']:
            logging.error("Test set is empty! Check the format of 'test_prefixes' in your YAML file.")
            logging.error("Expected format is 'speaker_name:file_name' (without .wav).")
            if item_names:
                logging.error(f"Example of the first item name in your dataset: {item_names[0]}")

        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def load_meta_data(self, raw_data_dir, ds_id):
        self.items.update(File2Batch.file2temporary_dict(raw_data_dir, ds_id))

    @staticmethod
    def build_spk_map():
        spk_map = {x: i for i, x in enumerate(hparams['speakers'])}
        assert len(spk_map) <= hparams['num_spk'], 'Actual number of speakers should be smaller than num_spk!'
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def augment_item(self, item, key_shift, speed):
        augmented_item = deepcopy(item)

        wav, sr = librosa.load(augmented_item["wav_fn"], sr=hparams["audio_sample_rate"], mono=True)

        mel = get_mel_torch(
            waveform=wav,
            sample_rate=sr,
            num_mel_bins=hparams["audio_num_mel_bins"],
            hop_size=hparams["hop_size"],
            win_size=hparams["win_size"],
            fft_size=hparams["fft_size"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            keyshift=key_shift,
            speed=speed,
            device=torch.device("cuda")
        )

        f0, _ = self.get_pitch_algorithm(wav, mel, hparams)

        if np.all(f0 == 0):
            raise ValueError(f"F0 estimation failed for augmented item.")
        
        if key_shift != 0:
            f0 *= 2 ** (key_shift / 12)
        
        pitch = f0_to_coarse(f0, hparams)

        augmented_item["mel"] = mel
        augmented_item["f0"] = f0
        augmented_item["pitch"] = pitch
        augmented_item['len'] = mel.shape[0]
        augmented_item['sec'] = mel.shape[0] * hparams['hop_size'] / sr
        augmented_item['spec_min'] = np.min(mel, 0)
        augmented_item['spec_max'] = np.max(mel, 0)

        return augmented_item

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train')

    def process_data_split(self, prefix):
        data_dir = hparams["binary_data_dir"]
        builder = IndexedDatasetBuilder(f"{data_dir}/{prefix}")
        lengths = []
        total_seconds = 0

        if self.binarization_args["with_spk_embed"]:
            voice_encoder = VoiceEncoder().cuda()
        
        tasks = []
        all_item_names = [name for name, _ in self.meta_data_iterator(prefix)]

        for item_name in all_item_names:
            tasks.append({
                "item_name": item_name,
                "key_shift": 0.,
                "speed": 1.
            })

            if prefix == "train":
                random_pitch_shifting_arguments = self.augmentation_args.get("random_pitch_shifting", {})

                if random_pitch_shifting_arguments.get("enabled", False) and random.random() < random_pitch_shifting_arguments.get("probability", 0.0):
                    key_min, key_max = random_pitch_shifting_arguments["range"]
                    shift = random.uniform(key_min, key_max)
                    tasks.append({
                        "item_name": item_name,
                        "key_shift": shift,
                        "speed": 1.
                    })
                
                fixed_pitch_shifting_arguments = self.augmentation_args.get("fixed_pitch_shifting", {})
                if fixed_pitch_shifting_arguments.get("enabled", False) and random.random() < fixed_pitch_shifting_arguments.get("probability", 0.0):
                    for target in fixed_pitch_shifting_arguments["targets"]:
                        tasks.append({
                            "item_name": item_name,
                            "key_shift": float(target),
                            "speed": 1.
                        })
                
                random_time_stretching_arguments = self.augmentation_args.get("random_time_stretching", {})
                if random_time_stretching_arguments.get("enabled", False) and random.random() < random_time_stretching_arguments.get("probability", 0.0):
                    stretch_min, stretch_max = random_time_stretching_arguments["range"]
                    speed = random.uniform(stretch_min, stretch_max)
                    tasks.append({
                        "item_name": item_name,
                        "key_shift": 0.,
                        "speed": speed
                    })
                
                fixed_time_stretching_arguments = self.augmentation_args.get("fixed_time_stretching", {})
                if fixed_time_stretching_arguments.get("enabled", False) and random.random() < fixed_time_stretching_arguments.get("probability", 0.0):
                    for target in fixed_time_stretching_arguments["targets"]:
                        tasks.append({
                            "item_name": item_name,
                            "key_shift": 0.,
                            "speed": float(target)
                        })
            
        print(f"| Total tasks for '{prefix}': {len(tasks)} (Original: {len(all_item_names)})")

        spec_min = []
        spec_max = []
        f0_dict = {}

        for task in tqdm(tasks, desc=f"Processing {prefix}"):
            item_name = task["item_name"]
            meta_data = self.items[item_name]
            key_shift = task["key_shift"]
            speed = task["speed"]

            try:
                original_item = File2Batch.temporary_dict2processed_input(item_name, deepcopy(meta_data), self.phone_encoder)

                if original_item is None:
                    continue

                if key_shift != 0. or speed != 1.:
                    item = self.augment_item(original_item, key_shift, speed)
                else:
                    item = original_item
                
                if self.binarization_args["with_spk_embed"]:
                    wav, _ = librosa.load(item["wav_fn"], sr=hparams["audio_sample_rate"], mono=True)
                    item["spk_embed"] = voice_encoder.embed_utterance(wav)
                
                spec_min.append(item["spec_min"])
                spec_max.append(item["spec_max"])
                f0_dict[f"{item['wav_fn']}_{task['key_shift']}_{task['speed']}"] = item["f0"]
                builder.add_item(item)
                lengths.append(item["len"])
                total_seconds += item["sec"]
            except Exception as e:
                print(f"| WARNING: Skipping {item_name} due to error: {e}")
        
        if prefix == "train" and len(spec_min) > 0:
            spec_max = np.max(spec_max, 0)
            spec_min = np.min(spec_min, 0)
            pitch_time = static_f0_time(f0_dict)

            with open(hparams["statictics_path"], encoding="utf-8") as f:
                _hparams = yaml.safe_load(f)
                _hparams["spec_max"] = spec_max.tolist()
                _hparams["spec_min"] = spec_min.tolist()

                if len(self.speakers) == 1:
                    _hparams["f0_static"] = json.dumps(pitch_time)
            
            with open(hparams["statictics_path"], "w", encoding="utf-8") as f:
                yaml.safe_dump(_hparams, f)
            
        builder.finalize()
        np.save(f"{data_dir}/{prefix}_lengths.npy", lengths)
        print(f"| {prefix} total duration: {total_seconds:2f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.process_pipeline import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder)


if __name__ == "__main__":
    set_hparams()
    SvcBinarizer().process()
