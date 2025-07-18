import audiofiles_switching.simpleaudio as simpleaudio

wav_obj = simpleaudio.WaveObject.from_wave_file("/home/takamichi-lab-pc05/ドキュメント/B4/concone_synthesizerV/concone_synthesizerV_sakiAI_trial1_MixDown.wav")
play_obj = wav_obj.play()
play_obj.wait_done()