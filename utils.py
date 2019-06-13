from matplotlib import pyplot as plt
from IPython import display as ipdisplay
from string import Formatter, ascii_letters
import progressbar

from dataclasses import dataclass
from typing import Any

import time
import regex as re
import random
import subprocess


def play_songs(song_txt):
    """
        Opens generated songs
    """
    songs = load_songs(song_txt)
    if not songs:
        print('No songs found. Try training the model longer' +
              ' or use a larger dataset')
    for song in songs:
        song_name = save_to_abc(song)
        failed = abc_to_wav(song_name)

        if not failed:
            return play_wav_snippet(song_name.replace('abc', 'wav'))
    print('Found no valid songs. Try training for longer')


def load_songs(gen_text):
    """
        Matches pattern to find song in generated text
    """

    pattern = '\n\n(.*?)\n\n'
    res = re.findall(pattern, gen_text, overlapped=True, flags=re.DOTALL)
    songs = [song for song in res]

    print(f'Found {len(songs)} likely songs')
    return songs


def save_to_abc(song: 'str', filename='') -> str:
    filename = generate_random_name() if not filename else f'{filename}.abc'

    with open(filename, 'w') as f:
        f.write(song)

    return filename


def generate_random_name():
    """
        Generates random file name
    """
    chars = list(ascii_letters)
    random.shuffle(chars)

    return ''.join(random.choices(chars, k=5)) + '.abc'


def abc_to_wav(song_file):
    """
        Converts the file format from abc to wav
    """
    player = 'data/abc_player'

    return subprocess.check_call(f'{player} {song_file}')


def play_wav_snippet(audio_file):
    """
        Loads the audio file in a player
    """
    return ipdisplay.Audio(audio_file)


@dataclass
class Plotter:
    """
        Periodic Plotter
    """
    sec: float = 1
    x_label: str = ''
    y_label: str = ''
    scale: Any = None
    start_time: Any = time.time()

    def plot(self, data):
        """
            Outputs periodic graphic representation of labels
        """
        if time.time() - self.start_time > self.sec:
            plt.cla()
        if not self.scale:
            plt.plot(data)
        else:
            getattr(plt, self.scale)(data)

        plt.xlabel(self.xlabel)
        plt.y_label(self.y_label)
        ipdisplay.clear_output(wait=True)
        ipdisplay.display(plt.gcf())

        self.start_time = time.time()


def create_progress_text(msg):
    """
        Creates label for the progress bar widget
    """
    keys = [key[1] for key in Formatter().parse(msg)]
    ids = {key: float('nan') for key in keys if key}

    return progressbar.FormatLabel(msg, ids)


def create_progress_bar(label=None):
    """
        Returns a progressbar instance
    """
    if not label:
        label = progressbar.FormatLabel('')

    pgrbar = progressbar.ProgressBar(widgets=[
        progressbar.Percentage(),
        progressbar.Bar(),
        progressbar.AdaptiveETA(),
        ' ',
        label
    ])

    return pgrbar


progress_bar_widgets = [
    'Training: ', progressbar.Percentage(),
    ' ',
    progressbar.Bar(marker='#',
                    left='[',
                    right=']'
                    ),
    ' ',
    progressbar.ETA(), ' '
]


def get_progress_bar(process='Training'):
    """
        Creates a progressbar widget for a display of
        a running process.

        Parameters
        ----------
        process: str
            Name of the process
    """
    widget = progress_bar_widgets[:]
    widget[0] = process if process else widget[0]

    return progressbar.ProgressBar(widgets=widget)
