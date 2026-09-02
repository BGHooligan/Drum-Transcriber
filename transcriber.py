from tkinter import filedialog
import playsound3 as ps3
import os
import shutil
import numpy as np
import soundfile
import librosa
import customtkinter as ctk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import mido

root = ctk.CTk()
frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=60, fill='both', expand=True)
scroll_frame = ctk.CTkScrollableFrame(root, width=400, height=400)
label = ctk.CTkLabel(frame, text='Select your Drum Stem', font=('Roboto', 10))
label.pack(pady=12, padx=10)
win = 0.075
cluster_samples = {}
cluster_labels = {}
onset_times = []
labels = []
eps = 2.299
bpm = 0
y = None
sr = None

MIDI_DRUMS = {
    "None": -6,
    "Bass Drum": 36,
    "Snare": 38,
    "Clap": 39,
    "Closed Hi-Hat": 42,
    "Pedal Hi-Hat": 44,
    "Open Hi-Hat": 46,
    "Low Floor Tom": 41,
    "High Floor Tom": 43,
    "Low Mid Tom": 47,
    "High Mid Tom": 48,
    "High Tom": 50,
    "Ride Cymbal": 51,
    "Crash Cymbal": 49,
    "Cowbell": 56,
}

def file():
    global y, sr, win
    filename = filedialog.askopenfilename(filetypes=(('Audio Files', '*.wav *.mp3'), ('All Files', '*.*')))
    if not filename:
        return
    y, sr = librosa.load(filename)
    win = float(wintxt.get())
    main()

button = ctk.CTkButton(frame, text="Open File", command=lambda: file())
button.pack()
winframe = ctk.CTkFrame(frame)
winframe.pack(pady=5)
winlbl = ctk.CTkLabel(winframe, text="Onset Window:", font=('Roboto', 12, 'normal'))
winlbl.grid(row=0, column=0, padx=(2,0), pady=2)
wintxt = ctk.CTkEntry(winframe, width=100)
wintxt.insert(0, str(win))
wintxt.grid(row=0, column=1, padx=(2,2), pady=2)

def extraction():
    global onset_times, labels, cluster_samples, y, sr, win, bpm
    bpm = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(bpm[0][0])
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames)
    onset_times = librosa.samples_to_time(onset_samples, sr=sr)
    sounds = []
    mfccs = []
    window = int(win*sr)
    for onset in onset_samples:
        start = onset
        end = onset + window
        if end <= len(y):
            hit = y[start:end]
        else:
            hit = y[start:len(y)]
        
        # Normalize and extract MFCCs
        hit = hit / (np.max(np.abs(hit)) + 1e-6)
        sounds.append(hit)
        mfcc = extract_mfcc(hit, sr)
        mfccs.append(mfcc)
        root.update_idletasks()

    return sounds, mfccs 


def main():
    global labels, cluster_samples, eps
    sounds, mfccs = extraction()
    labels = compare(mfccs, eps)
    cluster_samples = get_cluster_samples(labels, sounds)
    cluster_assignment_ui()

def get_cluster_samples(labels, sounds):
    cluster_dict = {}
    for cluster_id, sound in zip(labels, sounds):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = sound
    return cluster_dict

#Extract mfccs with dynamic n_fft mapping
def extract_mfcc(audio_segment, sr):
    n_fft = min(512, len(audio_segment) - 1)
    if n_fft < 64:
        n_fft = 64
    return librosa.feature.mfcc(y=audio_segment, sr=sr, n_fft=n_fft, n_mfcc=13)

#Compare mfccs using DBSCAN
def compare(mfccs, eps):
    features = np.array([mfcc.mean(axis=1) for mfcc in mfccs])
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features) #Normalize

    dbscan = DBSCAN(eps=eps, min_samples=1)
    cluster_labels = dbscan.fit_predict(features_norm) 
    
    return cluster_labels

def cluster_assignment_ui():
    global cluster_samples, cluster_labels, eps
    if frame.winfo_exists() != 0:
        for widget in frame.winfo_children():
            widget.destroy()
        frame.destroy()
    
    for widget in scroll_frame.winfo_children():
            widget.destroy()
    scroll_frame.pack()

    title = ctk.CTkLabel(scroll_frame, text="Assign Drum Types to Clusters", font=('Roboto', 14, 'bold'))
    title.pack(pady=10)

    eps_frame = ctk.CTkFrame(scroll_frame)
    eps_frame.pack(pady=5)
    eps_lbl = ctk.CTkLabel(eps_frame, text="EPS:", font=('Roboto', 12, 'normal'))
    eps_lbl.grid(row=0, column=0, padx=(2,0), pady=2)
    eps_txt = ctk.CTkEntry(eps_frame, width=100)
    eps_txt.insert(0, str(eps))
    eps_txt.grid(row=0, column=1, padx=(2,2), pady=2)
    eps_conf = ctk.CTkButton(eps_frame, text="Recluster", command= lambda: recluster(eps_txt.get()))
    eps_conf.grid(row=0, column=2, padx=(0,2), pady=2)

    assignments = {}
    for cluster_id in sorted(cluster_samples.keys()):
        cluster_frame = ctk.CTkFrame(scroll_frame, border_width=2, border_color="gray")
        cluster_frame.pack(pady=10, padx=10, fill='x')
        
        cluster_label = ctk.CTkLabel(cluster_frame, text=f"Cluster {cluster_id + 1}", font=('Roboto', 12, 'bold'))
        cluster_label.pack(pady=2)

        play_btn = ctk.CTkButton(cluster_frame, text="▶ Play Sample", command=lambda clusterid=cluster_id: play(clusterid), width=120)
        play_btn.pack(pady=2)

        dropdown1 = ctk.CTkOptionMenu(cluster_frame,values=list(MIDI_DRUMS.keys()),width=200)
        dropdown1.set(list(MIDI_DRUMS.keys())[0])
        dropdown1.pack()

        dropdown2 = ctk.CTkOptionMenu(cluster_frame,values=list(MIDI_DRUMS.keys()),width=200)
        dropdown2.set(list(MIDI_DRUMS.keys())[0])
        dropdown2.pack()

        dropdown3 = ctk.CTkOptionMenu(cluster_frame,values=list(MIDI_DRUMS.keys()),width=200)
        dropdown3.set(list(MIDI_DRUMS.keys())[0])
        dropdown3.pack(pady=(0,5))
        
        assignments[cluster_id] = [dropdown1, dropdown2, dropdown3]

    conf_frame = ctk.CTkFrame(scroll_frame)
    conf_frame.pack(pady=5)
    file_txt = ctk.CTkEntry(conf_frame, width=100, font=('Roboto', 12))
    file_txt.insert(0, "drums.mid")
    file_txt.grid(row=0, column=0, pady=20)
    confirm_btn = ctk.CTkButton(conf_frame, text="Generate MIDI", command=lambda: confirm(assignments, file_txt.get()), width=100, height=40, font=('Roboto', 12))
    confirm_btn.grid(row=0, column=1, pady=20)
    

def play(cluster_id):
    temp = f"temp_cluster_{cluster_id}.wav"
    soundfile.write(temp, cluster_samples[cluster_id], sr)
    ps3.playsound(temp)
    os.remove(temp)

def recluster(epsvalue):
    global eps
    eps=float(epsvalue)
    main()

def confirm(assignments, output):
    global cluster_labels, labels, onset_times, sr
    
    cluster_midis = {}
    for cluster_id, dropdowns in assignments.items():
        temp = []
        for x in range(3):
            drum = MIDI_DRUMS[dropdowns[x].get()]
            if drum > 0:
                temp.append(MIDI_DRUMS[dropdowns[x].get()])
        cluster_midis[cluster_id] = temp

    midi_labels = []
    for label in labels:
        temp = []
        for note in cluster_midis.get(label):
            temp.append(note)
        midi_labels.append(temp)
    
    generate_midi(midi_labels, output)
    
    for widget in scroll_frame.winfo_children():
        widget.destroy()
    
    root.quit()

def generate_midi(midi_labels, output):
    global onset_times, bpm
    if not output.lower().endswith(".mid"):
        output += ".mid"
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    tps = mid.ticks_per_beat * bpm / 60
    duration = int(0.1 * tps)
    previous = 0

    events = []

    for onset_time, notes in zip(onset_times, midi_labels):
        current = int(onset_time * tps)

        for note in notes:
            events.append((current, 'note_on', note, 100))
            events.append((current + duration, 'note_off', note, 0))

    events.sort(key=lambda event: event[0])

    previous = 0
    for current, message_type, note, velocity in events:
        dtime = current - previous
        track.append(mido.Message(message_type, channel=9, note=note, velocity=velocity, time=dtime))

        previous = current
    mid.save(output)
    print(f"MIDI file saved to {output}")

root.mainloop()
