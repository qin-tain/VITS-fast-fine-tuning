import argparse
import json
import os

import torch
import torchaudio
import whisper


def transcribe_one(audio_path, model):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return lang, result.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Please enable GPU in order to run Whisper!")

    args_token = {
        "C": "zh",
        "J": "ja",
        "E": "en",
    }
    lang2token_map = {
        "zh": "[ZH]",
        "ja": "[JA]",
        "en": "[EN]",
    }
    lang2token = {
        (a_t := args_token[alu]): lang2token_map[a_t]
        for al in args.languages
        if (alu := al.upper()) in args_token
    }

    model = whisper.load_model(args.whisper_size)
    parent_dir = "./custom_character_voice/"
    speaker_names = list(os.walk(parent_dir))[0][1]
    speaker_annos = []
    solved_files_info = []  # to save more information
    total_files = sum(len(files) for r, d, files in os.walk(parent_dir))
    # resample audios
    # 2023/4/21: Get the target sampling rate
    with open("./configs/finetune_speaker.json", "r", encoding="utf-8") as f:
        hps = json.load(f)
    target_sr = hps["data"]["sampling_rate"]
    processed_files = 0
    for speaker in speaker_names:
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # try to load file as audio
            if wavfile.startswith("processed_"):
                continue
            try:
                print(f"**solving: {wavfile}")
                wav, sr = torchaudio.load(
                    parent_dir + speaker + "/" + wavfile,
                    frame_offset=0,
                    num_frames=-1,
                    normalize=True,
                    channels_first=True,
                )
                wav = wav.mean(dim=0).unsqueeze(0)
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=target_sr
                    )(wav)
                if wav.shape[1] / sr > 20:
                    print(f"{wavfile} too long, ignoring\n")
                    continue  # add to ignore:)
                save_file_name = f"processed_{i}.wav"
                save_path = parent_dir + speaker + "/" + save_file_name
                torchaudio.save(save_path, wav, target_sr, channels_first=True)
                # transcribe text
                lang, text = transcribe_one(save_path, model)
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue
                # save more information
                solved_files_info.append((speaker, wavfile, save_file_name, lang, text))

                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)

                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")
            except Exception:
                continue

    # # clean annotation
    # import argparse
    # import text
    # from utils import load_filepaths_and_text
    # for i, line in enumerate(speaker_annos):
    #     path, sid, txt = line.split("|")
    #     cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
    #     cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    #     speaker_annos[i] = path + "|" + sid + "|" + cleaned_text
    # write into annotation
    if len(speaker_annos) == 0:
        print(
            "Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links."
        )
        print(
            "this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported."
        )
    with open("short_character_anno.txt", "w", encoding="utf-8") as f:
        for line in speaker_annos:
            f.write(line)
    with open("short_character_info.txt", "w", encoding="utf-8") as f:
        for item in solved_files_info:
            f.write("|".join(item) + "\n")

    # import json
    # # generate new config
    # with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    #     hps = json.load(f)
    # # modify n_speakers
    # hps['data']["n_speakers"] = 1000 + len(speaker2id)
    # # add speaker names
    # for speaker in speaker_names:
    #     hps['speakers'][speaker] = speaker2id[speaker]
    # # save modified config
    # with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
    #     json.dump(hps, f, indent=2)
    # print("finished")


if __name__ == "__main__":
    main()
