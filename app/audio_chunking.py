from pydub import AudioSegment

def split_audio_into_chunks(audio_path, chunk_length_ms, overlap_ms):
    """
    Splits an audio file into chunks with specified length and overlap.

    Args:
        audio_path (str): Path to the audio file.
        chunk_length_ms (int): Length of each chunk in milliseconds.
        overlap_ms (int): Overlap between chunks in milliseconds.

    Returns:
        list: List of AudioSegment objects representing the chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        chunks.append(chunk)
        start += chunk_length_ms - overlap_ms
        if start >= len(audio):
            break
    return chunks

def merge_word_lists(word_lists, offsets):
    """
    Merges multiple word lists with timestamp adjustments.

    Args:
        word_lists (list of list of dict): Each inner list contains dicts with 'word' and 'timestamp' keys.
        offsets (list of float): Offset in seconds to add to timestamps for each word list.

    Returns:
        list: Merged and sorted list of dicts with 'word' and 'timestamp'.
    """
    merged = []
    for word_list, offset in zip(word_lists, offsets):
        for word_dict in word_list:
            adjusted_timestamp = word_dict['timestamp'] + offset
            merged.append({'word': word_dict['word'], 'timestamp': adjusted_timestamp})
    merged.sort(key=lambda x: x['timestamp'])
    return merged