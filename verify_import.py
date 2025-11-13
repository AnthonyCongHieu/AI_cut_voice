try:
    from faster_whisper import WhisperModel
    print("Import successful")
    # Attempt to instantiate without loading a model
    try:
        model = WhisperModel("dummy_model", local_files_only=True)
        print("WhisperModel instantiation successful")
    except Exception as e:
        print(f"WhisperModel instantiation failed: {e}")
except ImportError as e:
    print(f"Import error: {e}")