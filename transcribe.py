from faster_whisper import WhisperModel
import sys
import json

output_path = ""

def transcribe(path,size="small"):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    print("Transcribing...")

    segments, info = model.transcribe(
        path,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )

    segments = list(segments)

    # debug purposes only, remove for final push

    save_json(segments,info)
    save_txt(segments)

def save_json(segments,info):
    segment_list =[]
    for segment in segments:
            clean_text = segment.text.strip()
            segment_list.append({
                 "start":segment.start,
                 "end":segment.end,
                 "text":clean_text})
            
    output_dict = {
         "schema_version": 1,
         "language": info.language,
         "duration": info.duration,
         "segments": segment_list
    }
    
    with open("./output/output.json","w",encoding="utf-8") as json_file:
            json.dump(output_dict,json_file,indent=4,ensure_ascii=False)

def save_txt(segments):
    text_lines = []
    for segment in segments:
        clean_text = segment.text.strip()
        text_lines.append(clean_text)

    with open("./output/output.txt","w",encoding="utf-8") as text_file:
        text_file.write("\n".join(text_lines))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Error: Missing required argument. Usage: python transcribe.py <file_path>")
    else:
        transcribe(sys.argv[1],"small")


