def play_songs(song_txt):
    """
        Opens generated songs
    """
    songs = load_songs(song_txt)
    if not songs:
        print('No songs found. Try training the model longer' +
              ' or use a larger dataset')
    for song in songs:
        base_name = convert_to_abc(song)
        ret = abc_to_wav(base_name + '.abc')

        if not ret:
            return play_wav_snippet(ret + '.wav')
    print('Found no valid songs. Try training for longer')
