#!/usr/bin/env python3

import re
import subprocess
import argparse
import sys
from dumbquotes import dumbquote
from fuzzysearch import find_near_matches as fuzzysearch
from phonemizer.backend import EspeakBackend
import os

from typing import Optional, TypeVar
from pathlib import Path
from shutil import (
    unpack_archive,
    copytree,
    rmtree,
    which
)

from rich.console import Console
from rich.pretty import Pretty
from rich.panel import Panel
from rich.table import Table
import rich.progress
from rich.progress import (
    BarColumn,
    DownloadColumn,
    TimeRemainingColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn
)
from vosk import Model, KaldiRecognizer, SetLogLevel

# Local imports
from model.models import (
    models_small,
    models_large,
    model_languages,
    get_language_features
)

'''
    Globals
'''

__version__ = '0.7.0'
__author__ = 'jgilmore'

PathLike = TypeVar('PathLike', Path, str, None)
vosk_url = "https://alphacephei.com/vosk/models"
vosk_link = f"[link={vosk_url}]this link[/link]"
# Default to ffmpeg in PATH
ffmpeg = 'ffmpeg'
con = Console()

'''
    Utility Function Declarations
'''


def path_exists(path: PathLike) -> Path:
    """Utility function to check if a path exists. Used by argparse.

    :param path: File path to verify
    :return: The tested file path if it exists
    """
    if Path(path).exists():
        return Path(path)
    else:
        raise FileNotFoundError(f"The path: <{path}> does not exist")


def verify_language(language: str) -> str:
    """Verifies that the selected language is valid.

    Used to confirm that the language passed via argparse is valid and also supported
    in the list of downloadable model files if the download option is selected.

    :param language: Model language
    :return: The language string if it is a supported language
    """

    found = False
    code = ''

    if language.lower() in model_languages.values():
        code = language.lower()
        found = True
    if not found and language.title() in model_languages.keys():
        code = model_languages[language.title()]
        found = True

    if not language:
        con.print("[bold red]ERROR:[/] Language option appears to be empty")
        sys.exit(1)
    elif not found:
        con.print("[bold red]ERROR:[/] Invalid language or langauge code entered. Possible options:")
        print("\n")
        con.print(Panel(Pretty(model_languages), title="Supported Languages & Codes"))
        print("\n")
        sys.exit(2)
    else:
        return code


def verify_download(language: str, model_type: str) -> str:
    """Verifies that the selected language can be downloaded by the script.

    If the download option is selected, this function verifies that the language
    model and size are supported by the script.

    :param language: Language of the model to download.
    :param model_type: Type of model (small or large).
    :return: String name of the model file to download if supported.
    """

    lang_code = verify_language(language)
    name = ''
    found = False
    other = 'small' if model_type == 'large' else 'large'

    if model_type == 'small':
        for line in models_small:
            if lang_code in line:
                name = line
                break
    elif model_type == 'large':
        for line in models_large:
            if lang_code in line:
                name = line
                break

    # If the specified model wasn't found, check for a different size
    if not name and model_type == 'small':
        for line in models_large:
            if lang_code in line:
                found = True
                break
    elif not name and model_type == 'large':
        for line in models_small:
            if lang_code in line:
                found = True
                break

    if not name and found:
        con.print(
            f"[bold yellow]WARNING:[/] The selected model cannot be downloaded for '{language}' "
            f"in the specified size '{model_type}'. However, a '{other}' model was found. "
            f"You can re-run the script and choose {other}, or attempt to "
            f"download a different model manually from {vosk_link}."
        )
        sys.exit(3)
    elif not name:
        con.print(
            f"[bold red]ERROR:[/] The selected model cannot be downloaded for '{language}' "
            f"in size {model_type}. You can try and download a different model manually "
            f"from {vosk_link}."
        )
        sys.exit(33)

    return name


def parse_config() -> dict:
    """Parses the toml config file.

    :return: A dictionary containing the config file contents.
    """

    if (config := Path.cwd().joinpath('defaults.toml')).exists():
        with open(config, 'r') as fp:
            lines = fp.readlines()
        defaults = {k: v for k, v in [l.strip("\n").replace("'", "").split('=') for l in lines if '#' not in l]}
        return defaults
    else:
        con.print("[bold red]ERROR:[/] Could not locate [blue]defaults.toml[/] file. Did you move or delete it?")
        print("\n")
        return {}


'''
    Function Declarations
'''


def parse_args():
    """
    Parses command line arguments.

    :return: A tuple containing the audiobook path, metadata file path, and user-defined metadata values
    """

    model_name = ''
    download = ''

    parser = argparse.ArgumentParser( formatter_class=argparse.RawTextHelpFormatter,
        description='''
        Splits a single monolithic mp3 audiobook file into multiple files for use in machine learning.
        Works in three steps:
        1. Uses vosk to generate an SRT file, which has timestamps for every word in the file.
        2. Uses fuzzy matching to merge the text ebook file with SRT file, generating a csv file
           with the results. This can be reviewed for accuracy and hand edited to improve results.
        3. Splits the mp3 audiobook file into pieces based on the csv file. Stores text->audio
           metadata in metadata-all.csv. Appends to the end, if it already exists.

        Only works on mp3 files. Audio data is copied, not re-encoded. Text merging is done phonetically.
        ebook text is assumed to be found as "mp3filename.txt"
        csv and srt files (the intermediate storage between steps) are also named after the mp3 file.
        output mp3 files are numbered, and the metadata-all.csv file has the corresponding ebook text.
        '''
    )
    parser.add_argument('audiobook', nargs='?', metavar='AUDIOBOOK_PATH',
                        type=path_exists, help='Path to audiobook file. Positional argument. Required')
    parser.add_argument('--textfile', '-tf', nargs='?', metavar='TEXTFILE_PATH',
                        type=path_exists, dest='textfile',
                        help='path to text file. Only needed to force a different name')
    parser.add_argument('--srtfile', '-sf', nargs='?', metavar='TIMECODES_PATH',
                        type=Path, dest='srtfile',
                        help='path to generated srt timecode file (if ran previously in a different directory)')
    parser.add_argument('--csvfile', '-cf', nargs='?', metavar='CSVCODES_PATH',
                        type=Path, dest='csvfile',
                        help='path to generated csv slicing file (if ran previously in aV different directory)')
    parser.add_argument('--stop-after', '-s', nargs=1, metavar='srt|csv|split',
                        type=str, dest='stop',
                        help='if set, stop after creating that file.')
    parser.add_argument('--language', '-l', dest='lang', nargs='?', default='en-us',
                        metavar='LANGUAGE', type=verify_language,
                        help='model language to use (en-us provided). See the --download_model parameter.')
    parser.add_argument('--model', '-m', dest='model_type', nargs='?', default='small',
                        type=str, choices=['small', 'large'],
                        help='model type to use if multiple models are available. Default is small.')
    parser.add_argument('--list_languages', '-ll', action='store_true', help='List supported languages and exit')
    parser.add_argument('--download_model', '-dm', choices=['small', 'large'], dest='download',
                        nargs='?', default=argparse.SUPPRESS,
                        help='download the model archive specified in the --language parameter')

    args = parser.parse_args()
    config = parse_config()

    if args.list_languages:
        con.print(Panel(Pretty(model_languages), title="Supported Languages & Codes"))
        print("\n")
        con.print(
            "[yellow]NOTE:[/] The languages listed are supported by the "
            "[bold green]--download_model[/]/[bold green]-dm[/] parameter (either small, large, or both). "
            f"You can find additional models at {vosk_link}."
        )
        print("\n")
        sys.exit(0)

    if 'download' in args:
        if args.lang == 'en-us':
            con.print(
                "[bold yellow]WARNING:[/] [bold green]--download_model[/] was used, but a language was not set. "
                "the default value [cyan]'en-us'[/] will be used. If you want a different language, use the "
                "[bold blue]--language[/] option to specify one."
            )

        download = 'small' if args.download not in ['small', 'large'] else args.download
        model_name = verify_download(args.lang, download)

    # If the user chooses to download a model, and it was verified
    if download:
        model_type = download
    # If the user passes a value via CLI (overrides config)
    elif 'model_type' in args:
        model_type = args.model_type
    # If the config file contains a value
    elif config['default_model']:
        # Verify valid model size
        if (model_type := config['default_model']) not in ('small', 'large'):
            con.print(
                f"[bold red]ERROR:[/] Invalid model size in config file: '{model_type}'. "
                "Defaulting to 'small'."
            )
            model_type = 'small'
    else:
        model_type = 'small'

    # If the user passes a language via CLI
    if 'lang' in args:
        language = args.lang
    # Check for a value in the config file
    elif 'default_language' in config:
        language = verify_language(config['default_language'])
    else:
        language = 'en-us'

    # Set ffmpeg from system PATH or config file
    global ffmpeg
    if 'ffmpeg_path' in config and config['ffmpeg_path'] != 'ffmpeg':
        if Path(config['ffmpeg_path']).exists():
            ffmpeg = Path(config['ffmpeg_path'])
        elif (ffmpeg := which('ffmpeg')) is not None:
            con.print(
                "[yellow]NOTE:[/] ffmpeg path in [blue]defaults.toml[/] does not exist, but "
                f"was found in system PATH: [green]{ffmpeg}[/]"
            )
            ffmpeg = 'ffmpeg'
        else:
            con.print("[bold red]CRITICAL:[/] ffmpeg path in [blue]defaults.toml[/] does not exist")
            sys.exit(1)
    elif (ffmpeg := which('ffmpeg')) is not None:
        ffmpeg = 'ffmpeg'
    else:
        con.print("[bold red]CRITICAL:[/] ffmpeg was not found in config file or system PATH. Aborting")
        sys.exit(1)

    if not args.textfile:
        args.textfile = args.audiobook.with_suffix('.txt')
        if not args.textfile.exists():
            # When textfile is specified on the command line, argsparser makes sure it exists.
            con.print(f"[bold red]CRITICAL:[/] Text file {args.textfile} was not found. Aborting")
    if not args.srtfile:
        args.srtfile = args.audiobook.with_suffix('.srt')
    if not args.csvfile:
        args.csvfile = args.audiobook.with_suffix('.csv')
    if args.stop:
        stop=args.stop[0]
    else:
        stop=None

    return args.audiobook, args.textfile, args.srtfile, args.csvfile, stop, language, model_name, model_type


def build_progress(bar_type: str) -> Progress:
    """Builds a progress bar object and returns it.

    :param bar_type: Type of progress bar.
    :return: a Progress object
    """

    text_column = TextColumn(
        "[bold blue]{task.fields[verb]}[/] [bold magenta]{task.fields[noun]}",
        justify="right"
    )

    if bar_type == 'chapterize':
        progress = Progress(
            text_column,
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn()
        )
    elif bar_type == 'download':
        progress = Progress(
            text_column,
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn()
        )
    elif bar_type == 'file':
        progress = Progress(
            text_column,
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn()
        )
    else:
        raise ValueError("Unknown progress bar type")

    return progress


def print_table(list_dicts: list[dict]) -> None:
    """Formats a list of dictionaries into a table. Currently only used for timecodes.

    :param list_dicts: List of dictionaries to format
    :return: None
    """

    table = Table(
        title='[bold magenta]Parsed Timecodes for Chapters[/]',
        caption='[red]EOF[/] = End of File'
    )
    table.add_column('Start')
    table.add_column('End')
    table.add_column('Chapter')

    merge_rows = []
    for item in list_dicts:
        row = []
        for v in item.values():
            row.append(v)
        merge_rows.append(row)

    if len(merge_rows[-1]) != 3:
        merge_rows[-1].append('EOF')
    for i, row in enumerate(merge_rows):
        table.add_row(f"[green]{str(row[0])}", f"[red]{str(row[2])}", f"[bright_blue]{str(row[1])}")

    con.print(table)


def convert_to_wav(audiobook_path: PathLike) -> Path:
    """
    Convert input file to lossless wav format. Currently unused, but might be useful for
    legacy versions of vosk.

    :param audiobook_path: Input .mp3 file to convert
    :return: Path to .wav file
    """

    wav_file = audiobook_path.with_suffix('.wav')
    con.print("[magenta]Converting file to wav...[/]")
    result = subprocess.run([
        str(ffmpeg), '-i', audiobook_path, '-ar', '16000', '-ac', '1', wav_file
    ])

    print(f"Subprocess result: {result.returncode}")

    return wav_file


def download_model(name: str) -> None:
    """Downloads the specified language model from vosk (if available).

    :param name: Name of the model found on the vosk website
    :return: None (void)
    """

    try:
        import requests
        from requests.exceptions import ConnectionError as ReqConnectionError
    except ImportError:
        con.print(
            "[bold red]CRITICAL:[/] requests library is not available, and is required for "
            "downloading models. Run [bold green]pip install requests[/] and re-run the script."
        )
        sys.exit(18)

    full = f'{vosk_url}/{name}.zip'
    out_base = Path('__file__').parent.absolute() / 'model'
    out_zip = out_base / f'{name}.zip'
    out_dir = out_base / name

    if out_dir.exists():
        con.print("[bold yellow]it appears you already have the model downloaded. Sweet![/]")
        return

    progress = build_progress(bar_type='download')
    with requests.get(full, stream=True, allow_redirects=True) as req:
        if req.status_code != 200:
            raise ReqConnectionError(
                f"Failed to download the model file: {full}. HTTP Response: {req.status_code}"
            )

        size = int(req.headers.get('Content-Length'))
        chunk_size = 50 if 'small' in name else 300
        task = progress.add_task("", size=size, noun=name, verb='Downloading')
        progress.update(task, total=size)
        with open(out_zip, 'wb') as dest_file:
            with progress:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    dest_file.write(chunk)
                    progress.update(task, advance=len(chunk))

    try:
        unpack_archive(out_zip, out_dir)
        if out_dir.exists():
            con.print("[bold green]SUCCESS![/] Model downloaded and extracted successfully")
            print("\n")
            out_zip.unlink()
            child_dir = out_dir / name
            # If it extracts inside another directory, copy up and remove extra
            if child_dir.exists():
                child_dir.rename(Path(f"{child_dir}-new"))
                child_dir = copytree(Path(f"{child_dir}-new"), out_base / f"{name}-new")
                rmtree(out_dir)
                child_dir.rename(Path(out_dir))
        elif out_zip.exists() and not out_dir.exists():
            con.print(
                "[bold red]ERROR:[/] Model archive downloaded successfully, but failed to extract. "
                "Manually extract the archive into the model directory and re-run the script."
            )
            sys.exit(4)
        else:
            con.print(
                "[bold red]CRITICAL:[/] Model archive failed to download. The selected model "
                f"might not be supported by the script, or is unavailable. Follow {vosk_link} "
                "to download a model manually.\n"
            )
            sys.exit(5)
    except Exception as e:
        con.print(f"[bold red]CRITICAL:[/] Failed to unpack or rename the model: [red]{e}[/red]")
        sys.exit(29)


def split_file(audiobook_path: PathLike,
               timecodes: list[dict]) -> int:

    """Splits a single .mp3 file into chapterized segments.

    :param audiobook_path: Path to original .mp3 audiobook
    :param timecodes: List of start/end markers for each chapter
    :return: An integer status code
    """

    file_stem = audiobook_path.stem
    # Set the log path for output. If it exists, generate new filename
    log_path = audiobook_path.parent.joinpath('ffmpeg_log.txt')
    if log_path.exists():
        with open(log_path, 'a+') as fp:
            fp.writelines([
                '********************************************************\n',
                'NEW LOG START\n'
                '********************************************************\n\n'
            ])
    metadata_path = audiobook_path.parent.joinpath('metadata-all.csv')
    if metadata_path.exists():
        # Append to end of existing file
        # Which means we need to get the counter_offset
        with open(metadata_path,'r') as f:
            for line in f:
                pass
        counter_offset = int(line.split('|')[0])
        if counter_offset == 0:
            con.print(f"[bold red]ERROR:[/] failed to read offset from last line of existing [red]{metadata_path}[/red]\n")
            sys.exit(8)
    else:
        counter_offset = 0
    metadata = open(metadata_path,'a')

    command = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'info', '-i', f'{str(audiobook_path)}']
    stream = ['-c', 'copy']

    progress = build_progress(bar_type='chapterize')
    with progress:
        task = progress.add_task('', total=len(timecodes), verb='Splitting', noun='Audiobook...')
        for counter, times in enumerate(timecodes, start=1):
            command_copy = command.copy()
            if 'start' in times:
                command_copy[5:5] = ['-ss', str(times['start']) + 'ms' ]
            if 'end' in times:
                command_copy[7:7] = ['-to', str(times['end']) + 'ms' ]
            file_path = audiobook_path.parent.joinpath(f"{counter+counter_offset}.mp3")

            command_copy.extend([*stream, f'{file_path}'])

            try:
                with open(log_path, 'a+') as fp:
                    fp.write('----------------------------------------------------\n\n')
                    subprocess.run(command_copy, stdout=fp, stderr=fp)
            except Exception as e:
                con.print(
                    f"[bold red]ERROR:[/] An exception occurred writing logs to file: "
                    f"[red]{e}[/red]\nOutputting to stdout..."
                )
                subprocess.run(command_copy, stdout=subprocess.STDOUT)

            # Reset the list but keep reference
            command_copy = None

            # Write out metadata
            metadata.write(f"{counter+counter_offset}|{times['text']}\n")

            progress.update(task, advance=1)
    return counter_offset + len(timecodes)


def generate_timecodes(audiobook_path: PathLike, out_file: PathLike, language: str, model_type: str) -> Path:
    """Generate chapter timecodes using vosk Machine Learning API.

    This function searches for the specified model/language within the project's 'models' directory and
    uses it to perform a speech-to-text conversion on the audiobook, which is then saved in a subrip (srt) file.

    If more than 1 model is present, the script will attempt to guess which one to use based on input.

    :param audiobook_path: Path to input audiobook file
    :param language: Language used by the parser
    :param model_type: The type of model (large or small)
    :return: Path to timecode file
    """

    sample_rate = 16000
    model_root = Path(r"model")

    # If the timecode file already exists, exit early and return path
    if out_file.exists() and out_file.stat().st_size > 10:
        con.print("[bold green]SUCCESS![/] An existing srt timecode file was found")

        return out_file

    try:
        if model_path := [d for d in model_root.iterdir() if d.is_dir() and language in d.stem]:
            con.print(f":white_heavy_check_mark: Local ML model found. Language: '{language}'\n")
            # If there is more than 1 model, infer the proper one from the name
            if len(model_path) > 1:
                con.print(
                    f"[yellow]Multiple models for '{language}' found. "
                    f"Attempting to use the {model_type} model[/yellow]"
                )
                if model_type == 'small':
                    model_path = [d for d in model_path if 'small' in d.stem][0]
                else:
                    model_path = [d for d in model_path if 'small' not in d.stem][0]
            else:
                model_path = model_path[0]
    except IndexError:
        con.print(
            "[bold yellow]WARNING:[/] Local ML model was not found (did you delete it?) "
            "or multiple models were found and the proper one couldn't be inferred.\n"
            "The script will attempt to download an online model, which "
            "isn't always reliable. Fair warning."
        )
        model_path = None

    SetLogLevel(-1)
    model = Model(lang=language, model_path=str(model_path))
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    try:
        # Convert the file to wav (if needed), and stream output to file
        # Start by getting the length (in seconds) of the mp3 file and converting that to wav file
        # size in bytes, for the progress bar.
        length = subprocess.run(["ffprobe", '-show_entries', 
                                 'format=duration', '-i', audiobook_path], 
                                text=True, capture_output=True).stdout
        (_, length, _) = length.splitlines()
        (_, length) = length.split('=')
        # Length of .wav file is 2 bytes per sample, 16K samples per second.
        length=int(float(length) * 16000 * 2)
                                

        with rich.progress.wrap_file(subprocess.Popen([str(ffmpeg), "-loglevel", "quiet", "-i",
                               audiobook_path,
                               "-ar", str(sample_rate), "-ac", "1", "-f", "s16le", "-"],
                              stdout=subprocess.PIPE).stdout, length) as stream:
            with open(out_file, 'w+') as fp:
                fp.writelines(rec.SrtResult(stream, words_per_line = 1 ))

        con.print("[bold green]SUCCESS![/] Timecode file created\n")
    except Exception as e:
        con.print(f"[bold red]ERROR:[/] Failed to generate timecode file with vosk: [red]{e}[/red]\n")
        sys.exit(7)

    return Path(out_file)


class merge_srt:
    """Correct text from speach recognition with actual book text.

    This merges the text from the book into the srt file to create a cannonical
    and reliable interpretation of the audio file. Parts that don't match anything are
    assumed to be fluff/filler/music/disclaimer/voice actor credits, etc. that can be safely 
    discarded.

    This works via the following:
    read entire SRT file in, with spaces, to make a long string of the whole book.
        Keep the timing information in a seperate structure.
        Since the fuzzy search will be returning start/stop characters, we need to easily find the
        timecode for the start of the word based on the character of any part of the word.

    Fuzzy match sentences from the book to that long string, with a moving window.
        Quoted text is ALWAYS a different sentence. Paragraph (empty lines) are also sentence dividers.        
    store results as start/stop + text groups.
        Assuming there's silence between the words, the start/stop times should be halfway between
        the words.
    grab as many sentences as max_phenomes allows
        This can be fuzzy, as shorter sequences will be between dialogue or parts of dialogue,
        while longer ones can just be excluded later if they're really that long.
    output
        format is wav or mp3 files with names of <numeric ID>.<mp3/wav>, and a metadata file with
        <numeric id of wav/mp3>|<text spoken>
    """

    def __init__(self, srt_path: PathLike, text_path: PathLike, csv_path: PathLike):
        """ Merge the SRT data with ebook data to create the list of start/stop times with text

        Since text is from the ebook file, not the speach recognition, this can then be used to 
        train TTS and for other AI training purposes.
        """
        # First, check if the .csv file already exists. If so, just read it in to fill out 
        # self.slicelist, and return.
        self.slicelist = []
        if text_path.with_suffix('.csv').is_file():
            con.print("An existing csv file was found with directions on where to slice")
            self.read_csv(text_path.with_suffix('.csv'))
            con.print("[bold green]SUCCESS![/] csv file was read in, and was valid.")
            return


        # Read the srt file produced by vosk. This file is ONE WORD per srt record, and
        # is used because it contains the offsets in the audio file for the begining AND END of each
        # word.
        self.espeak = EspeakBackend('en-us')
        self.srt_text = ""
        self.srt_offsets = []
        self.srt_times = []
        self.srt_offset = 0
        self.srt_gtext = []
        self.read_srt(srt_path)

        # logfile is our actual output, and contains a record for each matching text, and each bit
        # of text skipped either in the srt file (the speach recognition file) and the text file.
        # These records are words that aren't in the audiobook, and that aren't in the ebook, 
        # respectively.
        self.bookonlytext = 0
        self.srtonlytext = 0
        self.goodtext = 0
        self.logfile = open(csv_path,'w')
        self.read_text(text_path)
        self.logfile.close()
        if self.goodtext < self.bookonlytext + self.srtonlytext:
            con.print("[bold red]ERROR![/] Merge failed more than succeded! Wrong text file? Empty text file?")
            con.print("In any case, a high failure-to-match rate probably means you really don't want to slice.")
            con.print(f"{self.bookonlytext} fragments from the ebook file were not able to be matched.")
            con.print(f"{self.srtonlytext} fragments from the srt file were not able to be matched.")
            con.print(f"{self.goodtext} fragments were successfully matched between the ebook and srt files.")
            con.print("In any case, a high failure-to-match rate probably means you really don't want to slice.")
            con.print(f"Removing the inadequate csv file {self.logfile.name}")
            os.remove(self.logfile.name)
            sys.exit(10)




    def read_csv(self, csv_file: PathLike):
        """ Read the csv file into a self.slicelist """
        with open(csv_file,'r') as csv:
            for i, line in enumerate(csv):
                if line[0] != 'G':
                    continue
                try:
                    (_, start, end, text) = line.rstrip().split('|')
                except ValueError:
                    print(f"valueerror on csv file line {i+1}")
                    raise
                self.slicelist += [{ "start": int(start), "end": int(end), "text": text }]


    def read_srt(self, srt_path: PathLike):
        counter = 1
        progress = build_progress("file")
        with open(srt_path, 'r') as srt:
            # Read the srt file in by records
            task = progress.add_task('', total=os.fstat(srt.fileno()).st_size, verb='Phonemizing', noun='srt file..')
            line = "\n"
            position = srt.tell()
            progress.start()
            progress.start_task(task)
            while line != "":
                progress.update(task, advance = srt.tell() - position)
                position = srt.tell()

                # First line of each record is always a strictly increasing integer counter
                line = srt.readline()
                if line == '':
                    break
                if int(line) != counter:
                    con.print(f"[bold red]CRITICAL:[/] counter didn't equal {counter} at line {self.srtnewlines}")
                    sys.exit(1)
                counter += 1

                # Second line is timestamp in the format "00:00:10,440 --> 00:00:11,310"
                line = srt.readline()
                start = self.srt_time_to_ms(line[0:13])
                stop = self.srt_time_to_ms(line[17:])

                # Third line is a single word (because we told vosk to do it that way)
                line = srt.readline()
                if " " in line:
                    con.print( f"[bold red]CRITICAL:[/] words may not contain a space! at line {srt.newlines}")
                    sys.exit(1)
                self.srt_times += [(start, stop)]
                self.srt_offsets += [len(self.srt_text)]
                self.srt_gtext += [line]
                line = self.to_phenomes(line)
                self.srt_text += line

                # Seperated by a blank line.
                line = srt.readline()
                if line != "" and line != "\n":
                    con.print(f"[bold red]CRITICAL:[/] synchronization error at line {srt.newlines}: unexpected non-blank line, got {line}")
                    sys.exit(1)
            progress.stop()


    def srt_time_to_ms(self, time :str) -> int:
        """ convert str timestamp "00:00:00,000" to ms 

        Note that this timestamp format is defined as part of the srt format, but character encoding
        somehow isn't.
        """
        """
        00:00:00.000
        012345678901
        """
        hours = int(time[0:2])
        minutes = int(time[3:5]) + hours * 60
        seconds = int(time[6:8]) + minutes * 60
        milliseconds = int(time[9:12]) + seconds * 1000
        return milliseconds


    def to_phenomes(self, text :str) -> str:
        """ convert text to phenomes in IPA 

        This works via handing the text off to espeak via a pipe, and reading the result.
        Relies on buffering by lines, and espeak translating to phenomes by lines.
        Which turns out to not be true.

        So the only reliable way to do this is spawn a new espeak process for EVERY seperate word
        of the srt file and every word of the ebook file.
        """
        return self.espeak.phonemize([text])[0].rstrip().lstrip()+" "


    def to_ms(self, offset :int) -> int:
        """ Convert an offset into the srt into a start time 

        from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value#
        Start time is halfway between the start of the current word 
        and the end of the previous word
        """
        from bisect import bisect_left

        pos = bisect_left(self.srt_offsets, offset)
        if pos == 0:
            return self.srt_times[pos][0]
        if pos == len(self.srt_offsets):
            return self.srt_times[pos-1][1]
        before = self.srt_offsets[pos - 1]
        after = self.srt_offsets[pos]
        if after - offset >= offset - before:
            # We were handed an offset close to the begining of a word
            pos = pos - 1

        # This is the "normal" case. 
        # Return a spot halfway between this word and the previous one.
        return self.srt_times[pos - 1][1] + (self.srt_times[pos][0] -  self.srt_times[pos - 1][1]) // 2


    def quotesplit(self, text :str ) -> list:
        """ Split a paragraph string into speach/non speach chunks

        Keeps quotes with speach parts.
        """
        output = []
        text = text.lstrip().rstrip()
        text = dumbquote(text)
        while text != "":
            index = text.find('"')
            # if current chunk starts with a quote:
            if index == 0:
                i = text.find('"',1)
                if i != -1:
                    i += 1
                    output.append(text[0:i])
                    text = text[i:]
                else:
                    output.append(text)
                    break
            elif index == -1:
                output.append(text)
                break
            else:
                output.append(text[0:index])
                text = text[index:]
            text = text.lstrip()
        return output

    def read_text(self, text_path :PathLike):

        # Begin processing ebook file by reading it into chunks
        progress = build_progress("file")
        with open(text_path,'r') as text:
            task = progress.add_task('', total = os.fstat(text.fileno()).st_size, verb='merging', noun='text')
            progress.start()
            progress.start_task(task)
            # Read all paragraphs, split them into chunks.
            # After this point, paragraphs are marked by double spaces.
            for line in text:
                progress.update(task, advance = len(line))
                for chunk in self.quotesplit(dumbquote(line.rstrip())):
                    self.write_text(chunk)
            progress.stop()


    def log(self, logtype :str, start :int, stop :int, text :[str] ) -> None:
        """ write a standardized record out to the logfile.

        After some review, this record needs to include the text (phonems) that failed to match,
        the text (grapheme) that it was generated from, the start/stop times if available, etc.

        This is mostly to allow manual editing of this file later to include/exclude matches.

        The "good" records also need to be saved as a dict to hand off to the file slicer.
        """
        if logtype == 'G':
            self.goodtext += 1
        elif logtype == 'B':
            self.bookonlytext += 1
        elif logtype == 'S':
            self.srtonlytext += 1
        elif logtype == 'M':
            pass
        else:
            con.print("log called with unknown type!")
            sys.exit(1)

        t = text.pop()
        for line in text:
            self.logfile.write(f"# {line}\n")
        self.logfile.write(f"{logtype}|{start}|{stop}|{t}\n")
        if logtype == 'G':
            self.slicelist += [{ "start": int(start), "end": int(stop), "text": t }]


    def write_text(self, text :str) -> None:
        """ Write text given out to the csv file

        Text should come from the chapter, not the SRT (speak recognition) and is matched with the
        SRT to determine start/stop times in the audio file.

        Format is pipe seperated, and is type, start, stop, text
        where "type" is "G"ood, "S"rt text discarded, or "B"ook text discarded
        SRT text discarded has valid timestamps, Book text discarded has start=stop
        """
        ptext = self.to_phenomes(text)

        # Fuzzysearch is EXPENSIVE, and the real bottleneck in this process.
        # To limit that, limit the length of the text we're searching inside to just the start
        # of the ebook, not the whole thing. (fuzzysearch doesn't stop after finding the first
        # match, AND we really don't want to skip the whole damned book because "chapter 3"
        # accidentally matched chapter 39.

        # For longer paragraphs, only search the first bit of the book
        # It'll match right away or not at all. (Search double the paragraph length)
        if len(ptext) > 80:
            end = self.srt_offset+len(ptext)+len(ptext)
        else:
            # For shorter paragraphs, only search the first couple K. (Assumes the audiobook has
            # at most 2K worth of garbage at the front/at the begining of each chapter)
            # String slicing doesn't get indexErrors, so we don't need to limit this.
            end = self.srt_offset+2000

        discardedmatch = False
        for match in fuzzysearch(ptext, 
                               self.srt_text[self.srt_offset:end],
                               max_l_dist = len(ptext) // 4 ):
            start = match.start + self.srt_offset
            end = match.end + self.srt_offset
            startms = self.to_ms(start)
            endms = self.to_ms(end)
            # Discard later poor matches. match.dist <= max_l_dist,
            # so this is max 1/4 * 1K
            if match.start < 1000 * (1 - match.dist / len(ptext)):
                # write skipped SRT text
                if match.start != 0:
                    #TODO: Log the original text from the .srt file as well.
                    self.log("S",self.to_ms(self.srt_offset), 
                         startms,
                         [self.srt_text[self.srt_offset:start]])
                self.log("G",startms, endms,
                                  [ptext, 
                                   self.srt_text[start:end], 
                                   text] )
                self.srt_offset = end
                return
            else:
                self.log("M", startms, endms, [self.srt_text[start:end], ptext] )
        # No good matches, discard the text.
        ms = self.to_ms(self.srt_offset)
        self.log("B", ms, ms, [ptext, text] )



def verify_count(audiobook_path: PathLike, expected :int) -> None:
    """Verify that the expected number of files were generated.

    Compares the number of files split from the audiobook to ensure it matches the length of the generated
    timecodes.

    :param audiobook_path: Path to audiobook file
    :param timecodes: List of dictionaries containing chapter type, start, and end times
    :return: None (void)
    """

    file_count = sum(1 for x in audiobook_path.parent.glob('*.mp3') if x.stem != audiobook_path.stem)
    if file_count >= expected:
        con.print(f"[bold green]SUCCESS![/] Audiobook split into {file_count} files\n")
    else:
        con.print(
            f"[bold yellow]WARNING:[/] {file_count} files were generated "
            f"which is less than the expected {expected}\n"
        )


def main():
    """
    Main driver function.

    :return: None
    """

    con.rule("[cyan]Starting script[/cyan]")
    con.print("[magenta]Preparing splitting magic[/magenta] :zap:...")

    # Check python version
    if not sys.version_info >= (3, 10, 0):
        con.print("[bold red]CRITICAL:[/] Python version must be 3.10.0 or greater to run this script\n")
        sys.exit(20)

    # Destructure tuple
    audiobook_file, text_file, srt_file, csv_file, stopme, lang, model_name, model_type = parse_args()
    if not str(audiobook_file).endswith('.mp3'):
        con.print("[bold red]ERROR:[/] The script only works with .mp3 files (for now)")
        sys.exit(9)

    # Download model if option selected
    if model_name and lang:
        con.rule(f"[cyan]Downloading '{lang} ({model_type})' Model[/cyan]")
        con.print("[magenta]Preparing download...[/magenta]")
        download_model(model_name)

    # Generate timecodes from mp3 file
    con.rule("[cyan]Using Vosk to Generate timecodes as .srt[/cyan]")
    srt_file = generate_timecodes(audiobook_file, srt_file, lang, model_type)

    if stopme == 'srt':
        sys.exit(0)

    # merge correct text from ebook file against srt file to generate slicelist
    slicelist = merge_srt(srt_file, text_file, csv_file).slicelist

    if stopme == 'csv':
        sys.exit(0)

    # Split the file
    con.rule("[cyan]Splitting File[/cyan]")
    expected = split_file(audiobook_file, slicelist)

    # Count the generated files and compare to timecode dict to ensure they match
    verify_count(audiobook_file, expected)


if __name__ == '__main__':
    main()
