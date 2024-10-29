"""Mapped datasets info
"""

__all__ = [
  'DatasetInformer',
  'Audioset',
  'AudiosetStrong',
  'Freesound50k',
  'Esc50',
  'Singapura',
  'MaestroReal',
  'Urbansas',
  'MavdTraffic',
  'IdmtTraffic',
  'UrbanSound8k',
  'TutSoundEvents2016',
  'TutSoundEvents2017',
  'MaestroSynthetic',
  'Sonyc',
  'Archeo',
  'TauSpatialSoundEvents2020',
  'TutRareSoundEvents',
  'ChimeHome',
  'Realised',
  'DesedReal',
  'Starss22',
  'Starss23',
  'Mats',
  'Nonspeech7k',
  'AnimalSound',
  'Nigens'
]

def get_non_subset_datasets_info():
  # Create a dictionary mapping class names to class objects
  dataset_classes = {name: globals()[name] for name in __all__
                     if name != 'DatasetInformer'}

  # Instantiate each dataset class
  datasets = [cls() for cls in dataset_classes.values()]

  # Filter out datasets that are subsets of others
  non_subset_datasets = [dataset for dataset in datasets
                         if dataset.subset_of is None]

  info = {}
  for db_obj in non_subset_datasets:
    info[db_obj.name] = db_obj.get_info()

  return info


def get_datasets_info():
  # Create a dictionary mapping class names to class objects
  dataset_classes = {name: globals()[name] for name in __all__
                     if name != 'DatasetInformer'}

  # Instantiate each dataset class
  datasets = [cls() for cls in dataset_classes.values()]

  info = {}
  for db_obj in datasets:
    info[db_obj.name] = db_obj.get_info()

  return info


def get_mapped_dataset():
  # Create a dictionary mapping class names to class objects
  dataset_classes = {name: globals()[name] for name in __all__
                     if name != 'DatasetInformer'}

  return [cls().name for cls in dataset_classes.values()]




class DatasetInformer():
  """Abstract class
  """
  def __init__(self,
               name,
               mapping_id,
               url,
               subset_of=None,
               description=None,
               source_audio=None) -> None:
    self.name = name
    self.mapping_id = mapping_id
    self.url = url
    self.subset_of = subset_of
    self.description = description
    self.source_audio = source_audio

  def get_info(self):
    subset_db = 'None'
    if self.subset_of is not None:
      subset_db = self.subset_of.name

    info = {
      'name': self.name,
      'mapping_id': self.mapping_id,
      'url': self.url,
      'subset_of': subset_db,
      'description': self.description,
    }
    if self.source_audio is not None:
      info['source_audio'] = self.source_audio

    return info


class Audioset(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='AudioSet',
      mapping_id='AudioSet',
      url='https://research.google.com/audioset/',
      subset_of=None,
      description=('2,084,320 human-labeled 10-second sound clips drawn'
                   ' from YouTube'),
      source_audio=None
    )


class AudiosetStrong(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='AudioSet strong',
      mapping_id='AudioSet_strong',
      url='https://research.google.com/audioset/download_strong.html',
      subset_of=Audioset(),
      description=None,
      source_audio=None
    )


class Freesound50k(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='FreeSound 50k',
      mapping_id='Fsd50k',
      url='https://annotator.freesound.org/fsd/release/FSD50K/',
      subset_of=None,
      description='Strong annoatation of AudioSet.',
      source_audio=None
    )


class Esc50(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='ESC-50',
      mapping_id='ESC50',
      url='https://github.com/karolpiczak/ESC-50/',
      subset_of=None,
      description='Labeled collection of 2000 environmental audio recordings.',
      source_audio=None
    )


class Singapura(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='SINGA:PURA',
      mapping_id='Singapura',
      url='https://paperswithcode.com/dataset/singa-pura/',
      subset_of=None,
      description=('Strongly-labelled polyphonic urban sound dataset '
                   'with spatiotemporal context'),
      source_audio=None
    )


class MaestroReal(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='MAESTRO Real - Multi-Annotator Estimated Strong Labels',
      mapping_id='MAESTROreal',
      url=(
        'https://researchportal.tuni.fi/en/datasets/'
        'maestro-real-multi-annotator-estimated-strong-labels'),
      subset_of=None,
      description=('Evaluation dataset for the DCASE challenge 2023 '
                   'task 4 B - Sound Event Detection with Soft Labels. '
                   'The evaluation dataset contains 26 real-life audio '
                   'files from 5 different acoustic scenes.'),
      source_audio=(
        'TUT Acoustic Scenes 2016: cafe/restaurant, city center, '
        'grocery store, metro station and residential area')
    )


class Urbansas(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='Urban Sound & Sight',
      mapping_id='Urbansas',
      url='https://ieeexplore.ieee.org/document/9747644',
      subset_of=None,
      description=('12 hours of unlabeled data along with 3 hours of '
                   'manually annotated data, including bounding boxes '
                   'with classes and unique id of vehicles, and strong '
                   'audio labels featuring vehicle types and indicating '
                   'off-screen sounds.'),
      source_audio=None
    )


class MavdTraffic(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='MAVD-traffic',
      mapping_id='MAVDtraffic',
      url='https://github.com/pzinemanas/MAVD-traffic',
      subset_of=None,
      description=('ban noise monitoring in Montevideo city, Uruguay'),
      source_audio=None
    )


class IdmtTraffic(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='IDMT-traffic',
      mapping_id='IDMTtraffic',
      url='https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html',
      subset_of=None,
      description=('17,506 2-second long stereo audio excerpts of recorded '
                   'vehicle passings as well as different background '
                   'sounds alongside streets.'),
      source_audio=None
    )


class UrbanSound8k(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='UrbanSound8k',
      mapping_id='UrbanSound8K',
      url='https://urbansounddataset.weebly.com/urbansound8k.html',
      subset_of=None,
      description=('8732 labeled sound excerpts (<=4s) of urban sounds '
                   'from 10 classes.'),
      source_audio=None
    )


class TutSoundEvents2016(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='TUT Sound Events 2016',
      mapping_id='TUTSoundEvents2016',
      url='https://zenodo.org/records/45759',
      subset_of=None,
      description=('TUT Sound events 2016, development dataset consists of '
                   '22 audio recordings from two acoustic scenes: '
                   'Home and Residential area.'),
      source_audio='TUT Acoustic Scenes: "home" and "residential area"'
    )


class TutSoundEvents2017(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='TUT Sound Events 2017',
      mapping_id='TUTSoundEvents2017',
      url='https://zenodo.org/records/400516',
      subset_of=None,
      description=('TUT Sound events 2017, development dataset consists of ',
                   '24 audio recordings from a single acoustic scene: Street.'),
      source_audio='TUT Acoustic Scenes: "street"'
    )


class MaestroSynthetic(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='MAESTRO Synthetic – Multiple Annotator Estimated STROng labels',
      mapping_id='MAESTROsynthetic',
      url='https://research.tuni.fi/machinelistening/datasets/',
      subset_of=UrbanSound8k(),
      description=('20 synthetic audio files created using Scaper, '
                   'each of them 3 minutes long'),
      source_audio='UrbanSound8k'
    )


class Sonyc(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='SONY-UST',
      mapping_id='SONYC',
      url='https://zenodo.org/records/3966543',
      subset_of=None,
      description=('2.5 hours of stereo audio recordings of 4718 vehicle '
                   'passing events captured with both high-quality sE8 '
                   'and medium-quality MEMS microphones.'),
      source_audio=None
    )


class Archeo(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='ARCHEO',
      mapping_id='Archeo',
      url='https://ieeexplore.ieee.org/document/9248467',
      subset_of=None,
      description=('A Dataset for Sound Event Detection in Areas of '
                   'Touristic Interest'),
      source_audio=None
    )


class TauSpatialSoundEvents2020(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='TAU NIGENS Spatial Sound Events 2020',
      mapping_id='TAUNIGENSSpatialSoundEvents2020',
      url='https://paperswithcode.com/dataset/tau-nigens-spatial-sound-events-2020',
      subset_of=Nigens(),
      description='Spatil sound-scene recordings',
      source_audio='NIGENS general events database'
    )


class TutRareSoundEvents(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='TUT Rare Sound Events 2017',
      mapping_id='TUTRareSoundEvents',
      url='https://zenodo.org/records/401395',
      subset_of=None,
      description=(
        'mixtures of rare sound events (classes baby cry, gun shot, '
        'glass break) with background audio.'),
      source_audio=('Backgrounds: TUT Acoustic scenes 2016',
                    'Foreground: Freesound')
    )


class ChimeHome(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='CHiME-Home',
      mapping_id='CHiMEHome',
      url='https://paperswithcode.com/dataset/chime-home',
      subset_of=None,
      description='6.8 hours of domestic environment audio recordings',
      source_audio=None
    )


class Realised(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='ReaLISED: Real-Life Indoor Sound Event Dataset',
      mapping_id='ReaLISED',
      url='https://www.mdpi.com/2079-9292/11/12/1811',
      subset_of=None,
      description='Real labeled indoor audio event recordings.',
      source_audio=None
    )


class DesedReal(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='DESED',
      mapping_id='DESEDReal',
      url='https://project.inria.fr/desed/',
      subset_of=Audioset(),
      description=('Dataset designed to recognize sound event classes in '
                   'domestic environments.'),
      source_audio='AudioSet'
    )


class Starss22(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='STARSS22: Sony-TAu Realistic Spatial Soundscapes 2022 dataset',
      mapping_id='Starss22',
      url='https://zenodo.org/records/6387880',
      subset_of=Starss23(),
      description=('Multichannel recordings of sound scenes in various'
                   ' rooms and environments.'),
      source_audio=None
    )


class Starss23(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='STARSS23: Sony-TAu Realistic Spatial Soundscapes 2023',
      mapping_id='Starss23',
      url='https://zenodo.org/records/7880637',
      subset_of=None,
      description=('Multichannel recordings of sound scenes in various'
                   ' rooms and environments.'),
      source_audio=None
    )


class Mats(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='MATS – Multi-Annotator Tagged Soundscapes',
      mapping_id='MATS',
      url='https://research.tuni.fi/machinelistening/datasets/',
      subset_of=None,
      description=('Strong annotations for 3930 audio files of ',
                   'TAU Urban Acoustic Scenes 2019'),
      source_audio='TAU Urban Acoustic Scenes 2019'
    )


class Nonspeech7k(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='Nonspeech7k',
      mapping_id='Nonspeech7k',
      url='https://zenodo.org/records/6967442',
      subset_of=None,
      description='Human nonspeech sound events.',
      source_audio=None
    )


class AnimalSound(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='AnimalSound',
      mapping_id='AnimalSound',
      url='https://github.com/YashNita/Animal-Sound-Dataset',
      subset_of=None,
      description='Animal sounds',
      source_audio=None
    )

class Nigens(DatasetInformer):
  def __init__(self) -> None:
    super().__init__(
      name='NIGENS (Neural Information Processing group GENeral sounds)',
      mapping_id='Nigens',
      url='https://zenodo.org/records/2535878',
      subset_of=None,
      description=('1017 wav files of various lengths (between 1s and 5mins),',
                   ' in total comprising 4h:46m of sound material'),
      source_audio=None
    )
