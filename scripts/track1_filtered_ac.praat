# Filtered-AC sensitivity comparison for recording 11, track 1.
# Tested with standalone Praat 6.6.30; NOT supported by Parselmouth's Praat 6.1.38.
# Usage: Praat.exe --utf8 --no-pref-files --run track1_filtered_ac.praat INPUT_WAV OUTPUT_PITCH
# Full original timeline; no resynthesis, amplitude scaling, or silence removal.
form: "Track 1 filtered autocorrelation comparison"
    sentence: "Input wav", "track_1_16k.wav"
    sentence: "Output pitch", "track1_filtered_ac.Pitch"
endform

Read from file: input_wav$
# time step, floor, top, candidates, very accurate, attenuation at top,
# silence threshold, voicing threshold, octave cost, octave-jump cost,
# voiced/unvoiced cost. Top=600 is the comparison setting, not raw ceiling.
To Pitch (filtered autocorrelation): 0.01, 50, 600, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
Save as binary file: output_pitch$
q05 = Get quantile: 0, 0, 0.05, "Hertz"
q50 = Get quantile: 0, 0, 0.50, "Hertz"
q95 = Get quantile: 0, 0, 0.95, "Hertz"
frames = Get number of frames
writeInfoLine: "frames=", frames
appendInfoLine: "q05_Hz=", q05
appendInfoLine: "q50_Hz=", q50
appendInfoLine: "q95_Hz=", q95
