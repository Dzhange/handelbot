import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

activation_path = "runs/twinkle_demo_1cam/videos/piano_activation.npy"
out_mid = "trained_activation.mid"

x = np.load(activation_path)  # shape: (160, 88)
print("Loaded:", activation_path)
print("shape:", x.shape, "min:", x.min(), "max:", x.max())

# 大于 0.5 认为这个键被按下
pressed = x > 0.5

# 88 键钢琴一般从 MIDI note 21(A0) 到 108(C8)
base_midi = 21

# video 是 10 fps，训练参数 control_freq=10，所以每个 step 约 0.1 秒
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

    # 原来按着、现在没按了 → note_off
    for note in sorted(active_notes - current_notes):
        events.append((t * ticks_per_frame, Message("note_off", note=int(note), velocity=0)))

    # 原来没按、现在按下了 → note_on
    for note in sorted(current_notes - active_notes):
        events.append((t * ticks_per_frame, Message("note_on", note=int(note), velocity=90)))

    active_notes = current_notes

# 结束时关闭所有还在响的音
end_tick = pressed.shape[0] * ticks_per_frame
for note in sorted(active_notes):
    events.append((end_tick, Message("note_off", note=int(note), velocity=0)))

# 转成 MIDI delta time
events.sort(key=lambda e: e[0])
last_tick = 0
for tick, msg in events:
    msg.time = max(0, tick - last_tick)
    track.append(msg)
    last_tick = tick

mid.save(out_mid)
print("Saved:", out_mid)
print("Total note events:", len(events))
