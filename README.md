<a href="https://github.com/jgilmore/Audiobook-To-AI-Training-data"><img alt="GitHub release (latest SemVer)" src="https://img.shields.io/github/v/release/patrickenfuego/Chapterize-Audiobooks"><a/>
<a href="https://github.com/jgilmore/Audiobook-To-AI-Training-data"><img alt="python" src="https://img.shields.io/badge/python-v3.10%2B-blue"><a/>
<a href="https://github.com/jgilmore/Audiobook-To-AI-Training-data"><img src="https://img.shields.io/badge/platform-win | linux | mac-eeeeee"><a/>

# Audiobook-To-Ai-Training-data

Split a single, monolithic mp3 audiobook file into chunks using Machine Learning and ffmpeg.
The vosk-generated .srt file is merged with the supplied ebook text to insure recognition was
accurate, and thus generate samples suitable for use for training other AI's for TTS or voice
recognition

This documentation is from chapterize, and has only been minimally updated. There will be old
information here.

---

## Table of Contents

- [Audiobook-To-AI-Training-data](#chapterize-audiobooks)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
    - [Machine Learning](#machine-learning)
    - [Models](#models)
    - [Metadata Parsing](#metadata-parsing)
    - [Cue files](#cue-files)
    - [Configuration File](#configuration-file)
  - [Dependencies](#dependencies)
    - [ffmpeg](#ffmpeg)
  - [Supported Languages and Models](#supported-languages-and-models)
  - [Usage](#usage)
    - [Examples](#examples)
    - [Docker](#docker)
  - [Improvement](#improvement)
    - [Language Support](#language-support)
  - [Known Issues](#known-issues)
    - [Access Denied Error on Windows](#access-denied-error-on-windows)

---

## About

This is a "simple" command line utility that will chop your mp3 audiobooks into smaller chewable chunks for you. No longer will you have to dissect a waveform, manually compare it to the ebook text, and use audacity or something to cut it up and create your training data.

I made this (and forked patrickenfuego's work to do so) to chop my OWN audiobooks up, as I've read several that are available on YouTube for free. Training an AI to imitate my voice is a work in progress still though, particularly with regard to character's voices.

### Models

> **NOTE**: Downloading models requires the `requests` library

The small United States English (`en-us`) machine learning model is already provided with this project. However, various other languages and model sizes are available to download directly from the script - it will unpack the archive for you, too, and no additional effort should be required. See [Supported Languages and Models](#supported-languages-and-models) for a comprehensive list.

Models should be saved within the project's `model` directory as this is where the script looks for them. If you want to save multiple models or sizes, no problem - just don't change the name of the unpacked model archive as the script uses portions of it to determine which model to use based on the combination of arguments passed.

If there is a model conflict, the script will perform a best effort guess to determine which one to use. If that fails, an error will be thrown.

### CSV files

CSV files can be created to make editing chapter markers and start/stop timecodes easier. This is especially useful if the machine learning speech-to-text conversion misses a section, timecodes require tweaking, or the matching didn't quite work out.

IF YOU WANT TO SANITIZE YOUR DATA A LITTLE MORE, this is where you should interviene. The CSV file contains both the book text AND the phonems that were generated from it and failed to match, allowing you to easily see and correct things. Understanding of IPA (The international phonetic alphabet) is helpful, but you can mostly guess around the failures anyway.

Set the --stop-after to "csv" and edit it by hand if needed.

CSV files are always generated within the same directory as the audiobook itself, but there are arguments which allow you to specify a custom path to a csv file if it is inconvenient to keep them paired with the audiobook.


### Configuration File

Included with this project is a `defaults.toml` file which you can use to specify configurations that you use frequently. This removes the need to pass certain CLI argument each time you run the script and provides options for specifying file paths (such as the path to `ffmpeg` if you don't feel like setting environment variables - see below).

Here is the base template included with the project:

```toml
# Uncomment lines starting with '#' to set config options, or modify an existing line.
# No spaces before/after the '='!
#
# Default model language
default_language='english'
# Default model size
default_model='small'
# Defaults to the system's PATH. You can specify a full path in single '' quotes
ffmpeg_path='ffmpeg'
# Change this to True if you always want the script to generate a cue file
generate_cue_file='False'
```

---

## Dependencies

- [ffmpeg](https://ffmpeg.org/)
- [ffprobe](https://ffmpeg.org/)
- [python](https://www.python.org/downloads/) 3.10+
  - Packages:
    - [rich](https://github.com/Textualize/rich)
    - [vosk](https://github.com/alphacep/vosk-api)
    - [requests](https://requests.readthedocs.io/en/latest/) (if you want to download models)

To install python dependencies, open a command shell and type the following:

> **NOTE**: If you're on Linux, you might need to use `pip3` instead

```bash
# Using the requirements file (recommended)
pip install -r requirements.txt
# Manually installing packages
pip install vosk rich requests
```

### ffmpeg

It is recommended that you add ffmpeg to your system PATH so you don't have to run the script from the same directory. How you do this depends on your Operating System; consult your OS documentation (if you aren't familiar with the process, it's super easy. Just Google it).

Here is a quick example for Windows using PowerShell (it can be done via GUI, too):

```powershell
# Whatever the path is to your ffmpeg install
$ffmpeg = 'C:\Users\SomeUser\Software\ffmpeg.exe'
$newPath = $env:PATH + ";$ffmpeg"
[Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
# Now close and reopen PowerShell to update
```

Here is a quick example using bash:

```bash
# Set this equal to wherever ffmpeg is
ffmpeg="/home/someuser/software/ffmpeg"
# If you're using zsh, replace with .zshrc
echo "export PATH=${ffmpeg}:${PATH}" >> ~/.bashrc
# Source the file to update
source ~/.bashrc
```

If you don't want to deal with all that, you can add the path of ffmpeg to the `defaults.toml` file included with the repository - copy and paste the full path and set it equal to the `ffmpeg_path` option **using single quotes** `''`:

```toml
# Specfying the path to ffmpeg manually
ffmpeg_path='C:\Users\SomeUser\Software\ffmpeg.exe'
# If ffmpeg is added to PATH, leave the file like this
ffmpeg_path='ffmpeg'
```

---

## Supported Languages and Models

> **NOTE**: You can set a default language and model size in the defaults.toml file included with the repository

The `vosk-api` provides models in several languages. By default, only the small 'en-us' model is provided with this repository, but you can download additional models in several languages using the script's `--download_model`/`-dm` parameter, which accepts arguments `small` and `large` (if nothing is passed, it defaults to `small`); if the model isn't English, you must also specify a language using `--language`/`-l` parameter. See [Usage](#usage) for more info.

Not all models are supported, but you can download additional models manually from [the vosk website](https://alphacephei.com/vosk/models) (and other sources listed on the site). Simply replace the existing model inside the `/model` directory with the one you wish to use.

The following is a list of models which can be downloaded using the `--download_model` parameter of the script:

> You can use either the **Language** or **Code** fields to specify a model

| **Language**    | **Code** | **Small** | **Large** |
|-----------------|----------|-----------|-----------|
| English (US)    | en-us    | ✓         | ✓         |
| English (India) | en-in    | ✓         | ✓         |
| Chinese         | cn       | ✓         | ✓         |
| Russian         | ru       | ✓         | ✓         |
| French          | fr       | ✓         | ✓         |
| German          | de       | ✓         | ✓         |
| Spanish         | es       | ✓         | ✓         |
| Portuguese      | pt       | ✓         | ✓         |
| Greek           | el       | ✕         | ✓         |
| Turkish         | tr       | ✕         | ✓         |
| Vietnamese      | vn       | ✓         | ✕         |
| Italian         | it       | ✓         | ✓         |
| Dutch           | nl       | ✓         | ✕         |
| Catalan         | ca       | ✓         | ✕         |
| Arabic          | ar       | ✕         | ✓         |
| Farsi           | fa       | ✓         | ✓         |
| Filipino        | tl-ph    | ✕         | ✓         |
| Kazakh          | kz       | ✓         | ✓         |
| Japanese        | ja       | ✓         | ✓         |
| Ukrainian       | uk       | ✓         | ✓         |
| Esperanto       | eo       | ✓         | ✕         |
| Hindi           | hi       | ✓         | ✓         |
| Czech           | cs       | ✓         | ✕         |
| Polish          | pl       | ✓         | ✕         |

The model used for speech-to-text the conversion is fairly dependent on the quality of the audio. The model included in this repo is meant for small distributions on mobile systems, as it is the only one that will fit in a GitHub repository. If you aren't getting good results, you might want to consider using a larger model (if one is available).

---

## Usage

```
usage: audiobook-to-AI-Training-data.py [-h] [--textfile [TEXTFILE_PATH]] [--srtfile [TIMECODES_PATH]] [--csvfile [CSVCODES_PATH]] [--stop-after srt|csv|split]
                                        [--language [LANGUAGE]] [--model [{small,large}]] [--list_languages] [--download_model [{small,large}]]
                                        [AUDIOBOOK_PATH]


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


positional arguments:
  AUDIOBOOK_PATH        Path to audiobook file. Positional argument. Required

options:
  -h, --help            show this help message and exit
  --textfile [TEXTFILE_PATH], -tf [TEXTFILE_PATH]
                        path to text file. Only needed to force a different name
  --srtfile [TIMECODES_PATH], -sf [TIMECODES_PATH]
                        path to generated srt timecode file (if ran previously in a different directory)
  --csvfile [CSVCODES_PATH], -cf [CSVCODES_PATH]
                        path to generated csv slicing file (if ran previously in aV different directory)
  --stop-after srt|csv|split, -s srt|csv|split
                        if set, stop after creating that file.
  --language [LANGUAGE], -l [LANGUAGE]
                        model language to use (en-us provided). See the --download_model parameter.
  --model [{small,large}], -m [{small,large}]
                        model type to use if multiple models are available. Default is small.
  --list_languages, -ll
                        List supported languages and exit
  --download_model [{small,large}], -dm [{small,large}]
                        download the model archive specified in the --language parameter

                        
```

### Examples

> **NOTE**: Each argument has a shortened alias. Most examples use the full argument name for clarity, but it's often more convenient in practice to use the aliases

Of particular interest is the "--stop-after" paremeter. Strongly recommend using this to avoid slicing your files until you've reviewed the CSV and confirmed that it meets expectations.


```powershell
# Set model to use German as the language (requires a different model, see above)
PS > python .\chapterize_ab.py 'C:\path\to\audiobook\file.mp3' --language 'de'
```

```bash
# Download a different model (Italian large used here as an example)
~$ python3 ./chapterize_ab.py '/path/to/audiobook/file.mp3' --download_model 'large' --language 'italian'
```

---

### Docker

This makes the assumption that you are familiar with building and running [Docker](https://docs.docker.com/manuals/). There is a Dockerfile that you can build and run this solution with.

Build with the following:

```powershell
docker build -t chapterizeaudiobooks .
```

With a volume setup, you can run this solution like so:

```powershell
docker run --rm -v "path/on/your/machine/to/audiobooks:/audiobooks" chapterizeaudiobooks "/audiobooks/file.mp3"
```

### Language Support

So far, support for this project is primarily targeted toward English audiobooks only; I've added some German content, but I'm by no means a fluent speaker and there are a lot of gaps.

If you want to contribute an exclusion list and chapter markers for other languages (preferably vosk supported languages), please do! Open a pull request or send them to me in a GitHub issue and I'll gladly merge them into the project. I'd like to make this project multi-lingual, but I can't do it without your help.

---

## Known Issues

### Access Denied Error on Windows

Every once in a while when downloading a new model on Windows, it will throw an "Access Denied" exception after attempting to rename the extracted file. This isn't really a permissions issue, but rather a concurrency one. I've found that closing any app or Explorer window that might be related to Audiobook-To-AI-Training-data usually fixes this problem. This seems to be a somewhat common issue with Python on Windows when renaming/deleting/moving files.
