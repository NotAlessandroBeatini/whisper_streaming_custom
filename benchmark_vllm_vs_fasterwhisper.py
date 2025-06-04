import time
import argparse
import os
from whisper_online import FasterWhisperASR, VllmASR, load_audio

parser = argparse.ArgumentParser(description="Benchmark FasterWhisper vs vLLM backend")
parser.add_argument("audio_path", help="Path to 16kHz mono wav file")
parser.add_argument("--runs", type=int, default=3, help="Number of runs for averaging")
parser.add_argument("--model", default="large-v2", help="Whisper model size for FasterWhisper")
parser.add_argument("--lan", default="en", help="Language")
parser.add_argument("--vllm_url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), help="vLLM server base URL")
args = parser.parse_args()

audio = load_audio(args.audio_path)

print("Benchmarking FasterWhisper...")
fw = FasterWhisperASR(modelsize=args.model, lan=args.lan)
fw_times = []
for _ in range(args.runs):
    start = time.time()
    fw.transcribe(audio)
    fw_times.append(time.time() - start)
print(f"FasterWhisper avg: {sum(fw_times)/len(fw_times):.3f}s over {args.runs} runs")

print("Benchmarking vLLM backend...")
vllm = VllmASR(lan=args.lan, base_url=args.vllm_url)
vllm_times = []
for _ in range(args.runs):
    start = time.time()
    vllm.transcribe(audio)
    vllm_times.append(time.time() - start)
print(f"vLLM avg: {sum(vllm_times)/len(vllm_times):.3f}s over {args.runs} runs")
