# Drum-Transcriber
A simple, single python file program that attempts to transcribe a Drum Stem into Midi. Works best on drum kits. Works by using onset detection to find drum hits, comparing and clustering those hits using MFCCs and DBSCAN, then the user classifies the cluster, and then finally creating the final midi file.
