from exiftool import ExifToolHelper

# pyexiftool's default arguments plus large file support: without it exiftool
# does not read the QuickTime metadata (e.g. CameraSerialNumber) of files
# over 4 GB, such as long GoPro takes.
EXIFTOOL_ARGS = ['-G', '-n', '-api', 'LargeFileSupport=1']


def ExifTool() -> ExifToolHelper:
    """
    ExifToolHelper with large file support; use as `with ExifTool() as et:`.
    """
    return ExifToolHelper(common_args=EXIFTOOL_ARGS)


def get_videos_metadata(video_paths, keys=['QuickTime:CameraSerialNumber', 'QuickTime:Model']):
    results = dict()
    with ExifTool() as et:
        for meta in et.get_metadata(video_paths):
            result = dict()
            for key in keys:
                result[key] = meta[key]
            results[meta['SourceFile']] = result
    return results
