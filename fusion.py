from collections import defaultdict, Counter

def fuse(voice_turns, face_sightings):
    scores = defaultdict(Counter)
    for turn in voice_turns:
        speaker = turn["speaker"]
        for face in face_sightings:
            if turn["start"] <= face["time"] <= turn["end"]:
                scores[speaker][face["person"]] += 1

    mapping = {}
    for speaker, counts in scores.items():
        mapping[speaker] = counts.most_common(1)[0][0] if counts else "UNKNOWN"

    enriched = []
    for turn in voice_turns:
        item = dict(turn)
        item["person"] = mapping.get(turn["speaker"], "UNKNOWN")
        enriched.append(item)
    return mapping, enriched
