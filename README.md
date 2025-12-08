# F5E-TTS Codebase

This repository is a research fork of F5-TTS built to reproduce the results in `F5E-TTS`. It keeps the flow-matching transformer pipeline and extends it with phonetic posteriorgram (PPG) conditioning, vector-quantized codebooks, and related ablations. Only local, script-based workflows are supported; upstream conveniences (pip package, Docker images, Gradio UI) are **not maintained here** even if some code remains.

## Layout
- `src/f5_tts/train/` – main training scripts, dataset prep scripts.
- `src/f5_tts/model/` – core models, flow-matching trainer, utilities.
- `src/f5_tts/configs/` – Hydra configs.
- `src/f5_tts/eval/` – batch inference for evaluation sets and objective metrics.

## Environment
- Python 3.10 recommended.
- Install locally (editable):
  ```bash
  pip install -e .
  ```
- Extras for evaluation:
  ```bash
  pip install -e .[eval]
  ```
- Configure Accelerate before multi-GPU runs:
  ```bash
  accelerate config
  ```

## Required Assets (place locally and update paths if needed)
- Vocoder checkpoint (default path in configs): `pretrained_models/vocos-mel-24khz`.
- PPG model and stats (used when `use_ppg: True` in configs), e.g. `pretrained_models/ppg/33.pt`, its `train.yaml`, and associated `phn_center.npy` / `ce_layer.pkl`.
- TTS checkpoints: training outputs are saved under `ckpts/...`; inference expects explicit `--ckpt_file` paths.

## Data Preparation
Dataset preprocessors write Arrow data to `data/{DATASET}_{tokenizer}` with `raw.arrow` (or `mel.arrow`), `duration.json`, and `vocab.txt`. Edit paths inside the scripts, then run, for example:
```bash
# Emilia
python src/f5_tts/train/datasets/prepare_emilia_v2.py
# LibriTTS
python src/f5_tts/train/datasets/prepare_libritts.py
```

## Training
Write a Hydra config under `src/f5_tts/configs/` that matches your dataset, tokenizer, and features (PPG, codebook, etc.).
```bash
# Example: train a small model with PPG inputs and the codebook on LibriTTS
accelerate launch src/f5_tts/train/train.py --config-name example.yaml
```

## Inference
Use the local script; the upstream pip entrypoints are not supported here.
```bash
python src/f5_tts/infer/infer_cli.py \
  --model F5TTS_v1_Base \            # matches a config in src/f5_tts/configs
  --ckpt_file /path/to/model_XXXX.safetensors \
  --ref_audio path/to/ref.wav \
  --ref_text "Transcription of the reference audio" \
  --gen_text "Target text to synthesize" \
  --vocoder_name vocos \             # or bigvgan if you provide it
  --ode_method euler --nfe_step 32 \
  --cfg_strength 2 --sway_sampling_coef -1
```
Tips:
- Provide `--vocab_file` if you trained with a custom tokenizer.
- If using PPG features, ensure PPG paths in your config/checkpoint are valid.

## Evaluation
Place test sets under `data/` (Seed-TTS, LibriSpeech test-clean, etc.) and update paths in scripts if needed.
```bash
# Generate evaluation audio
accelerate launch src/f5_tts/eval/eval_infer_batch.py --config-name <CONFIG> --nfe 16 --ckpt /path/to/ckpt

# Metrics examples
python src/f5_tts/eval/eval_seedtts_testset.py --eval_task wer --lang en --gen_wav_dir <DIR> --gpu_nums 8
python src/f5_tts/eval/eval_librispeech_test_clean.py --eval_task sim --gen_wav_dir <DIR> --librispeech_test_clean_path <PATH>
python src/f5_tts/eval/eval_utmos.py --audio_dir <DIR> --ext wav
```
Download the ASR/embedding models referenced in `src/f5_tts/eval/README.md` before running metrics.

## Reminders
- No maintained support for pip installation as a library, Docker images, or Gradio UI.
- Keep dataset, checkpoint, vocoder, and PPG paths consistent with your chosen Hydra config.
