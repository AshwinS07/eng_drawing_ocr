# Minimal replacement for removed stdlib imghdr (Python 3.13+)
import mimetypes

def what(file, h=None):
    if isinstance(file, (str, bytes)):
        kind = mimetypes.guess_type(file)[0]
        if kind and kind.startswith("image/"):
            return kind.split("/")[1]
    return None
