#02/2020
#rtroulak
#rtroulak@protonmail.com

import librosa
import librosa.display
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

"""
"""

## load file, we have 4 test files lute.wav, flute.wav, piano.wav and pianoBig.wav in google drive
folder = "audiosamples"
filename = "lute.wav"
file = folder + "/" + filename

# file parameters
offset = 0 # offset: starting reading after this time (in seconds)
duration = 30 # duration: to avoid big audio files

# chromagram predefines
sr = 44100# sr: sampling rate
hop_length = 4096# hop_length: (frame size)

#  templateGram predefines
w = 30 # w: filter size

"""# Chromagram function:
"""

def chromaGram(sr, file):
    # reads audio file. Parameters is the startng second (offset) and the total duration
    y, sr = librosa.load(file, sr=sr, offset=offset, duration=duration)

    # split harmonic content and percussive content to keep only harmonic for our analysis
    harmonic, percussive = librosa.effects.hpss(y)
    chromagram = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)

    #plot the chromagram 
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(chromagram, sr=sr, x_axis="frames",  y_axis="chroma")
    plt.title("Chroma Features")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return chromagram

"""# Template Function
"""

def templateCreate():
    template = {}
    majors = ["A","Bb","B","C","Db","D","Eb","E","F","F#","G","Ab"]
    minors = ["Am","Bbm","Bm","Cm","Dbm","Dm","Ebm","Em","Fm","F#m","Gm","Abm"]
    # template with the first chord A and Am
    template_a = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    template_am = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    i = 0

    for chord in majors:
        template[chord] = template_a[12 - i:] + template_a[:12 - i]
        i += 1

    for chord in minors:
        template[chord] = template_am[12 - i:] + template_am[:12 - i]
        i += 1

    # template for no chords
    template_noChords = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    template["NC"] = template_noChords

    return template

"""# templateGram functions
"""

def templateGram(C):
    frames = C.shape[1]

    # initialize tables chords and vectors
    template = templateCreate()
    chords = list(template.keys())
    chroma_vectors = np.transpose(C)
    H = []

    for n in np.arange(frames):
        cr = chroma_vectors[n]
        sims = []

        for chord in chords:
            t = template[chord]
            # calculate cos sim, add weight for non chord and for chords
            if chord == "NC":
                sim =  np.dot(cr, t) / (np.linalg.norm(cr) * np.linalg.norm(t)) * 0.7
            else:
                sim =  np.dot(cr, t) / (np.linalg.norm(cr) * np.linalg.norm(t))
            sims += [sim]
        H += [sims]
    H = np.transpose(H)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(H, sr=sr, x_axis="frames")
    plt.title("templateGram Simple")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return H

"""# Smooth templateGram
"""

def smoothing(s):
    w = 15
    news = [0] * len(s)
    for k in np.arange(w, len(s) - w):
        m = mode([s[i] for i in range(k - w // 2, k + w // 2 + 1)])[0][0]
        news[k] = m
    return news

def smoothedTemplateGram(H):
    chords = H.shape[0]
    H1 = []

    for n in np.arange(chords):
        H1 += [smoothing(H[n])]

    H1 = np.array(H1)
    
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(H1, sr=sr, x_axis="frames")
    plt.title("templateGram Smoothed")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return H1

"""# Chord Sequence and Label return
"""

# give the sequence of chords based on template array
def chordSequence(H):
    template = templateCreate()
    chords = list(template.keys())
    frames = H.shape[1]
    H = np.transpose(H)
    R = []

    for n in np.arange(frames):
        index = np.argmax(H[n])
        if H[n][index] == 0.0:
            chord = "NC"
        else:
            chord = chords[index]

        R += [chord]

    return R
# translate chrod on labels to represent on main procedure
def toChords(input):
    string = ""
    for r in input:
        if r == "NC": #if is not chord prin space
            string += " "
        else:
            string += " " + r
    return string

"""# Main Body of Script
"""

print("Script for Chord representation and recognition")
print("File Name:", file)
print("Offset and Duration:", offset, "to", offset + duration)


#chromaprin
chroma = chromaGram(sr,file)


#Chords without smoothing, gram and chord string
chord = templateGram(chroma)
result = chordSequence(chord)
print("Chords without smoothing:")
print(toChords(result))


#Chords with smoothing, gram and chord string
chordSmoothed = smoothedTemplateGram(chord)
resultSmoothed = chordSequence(chordSmoothed)
print("Chords with smoothing:")
print(toChords(resultSmoothed))
