from .transcribe.transcribe import Transcribe
from .clip.clip import Clip

# from .resize.resize import Resize
__all__ = ["Transcribe", "Clip", "Resize"]
# __all__ = ["TranscribeAndClip", "Resize"]


# now the user can do:
# from clipsai import TranscribeAndClip

# # Example usage
# transcribe_clip = TranscribeAndClip()
# result = transcribe_clip.run(request_data)

