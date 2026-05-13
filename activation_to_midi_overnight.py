import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

activation_path = "runs/twinkle_overnight_from_201/videos/piano_activation.npy"
out_mid = "overnight_activation.mid"

x = np.load(activation_path)
print("Loaded:", activation_path)
print("shape:", x.shape, "min:", x.min(), "max:", x.max())

pressed = x > 0.5

# 88键钢琴：A0=21 到 C8=108
base_midi = 21

# 视频/控制频率大概是 10 fps / 10 Hz
fps = 10
bpm = 120
ticks_per_beat = 480
seconds_per_tick = 60 / bpm / ticks_per_beat
ticks_per_frame = int((1 / fps) / seconds_per_tick)

mid = MidiFile(ticks_per_beat=ticks_per_beat)
track = MidiTrack()
mid.tracks.append(track)
track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))

active_notes = set()
events = []

for t in range(pressed.shape[0]):
    current_notes = set(np.where(pressed[t])[0] + base_midi)

    for note in sorted(active_notes - current_notes):
        events.append((t * ticks_per_frame, Message("note_off", note=int(note), velocity=0)))

    for note in sorted(current_notes - active_notes):
        events.append((t * ticks_per_frame, Message("note_on", note=int(note), velocity=90)))

    active_notes = current_notes

end_tick = pressed.shape[0] * ticks_per_frame
for note in sorted(active_notes):
    events.append((end_tick, Message("note_off", note=int(note), velocity=0)))

events.sort(key=lambda e: e[0])
last_tick = 0
for tick, msg in events:
    msg.time = max(0, tick - last_tick)
    track.append(msg)
    last_tick = tick

mid.save(out_mid)
print("Saved:", out_mid)
print("Total note events:", len(events))
