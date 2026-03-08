## Aurelius Relation Aware Evaluation

It takes four steps to run both the general evaluation and relation aware evaluation.

1. All pre-trained PANNs model can be downloaded via [GoogleDrive](https://drive.google.com/drive/folders/1y-Ys6mGyTP2erdBeovuxZSvXl6EvEkLX?usp=sharing), which is used for audio event detection and classification from the generated
audio. The relation aware evaluation depends the PANNs model analyzing result.

2. Prepare the dataset to evaluate. Basically the dataset contains a json file (e.g., `aurelius_test.json`, see [TextAudioPairGen](TextAudioPairGen/README.md)) recording each test case's text prompt, corresponding reference audio path.

3. edit `config.yaml` for configuration. Specifically,
  ```yaml
    refaudio_dir: aurelius_test
    predaudio_dir: predaudio_dir
    refaudio_data_filename: aurelius_test/aurelius_test.json
    result_save_dir: result_save_dir
    predaudio_key: methodname #ref_audio.wav is reference, the generated audio is referred by: ref_audio_methodname.wav
  ```
4. run `python main.py` to get the evaluation result, which will be dumped to `result_save_dir`.

  ```python
    python main.py
  ```
