from exiftool import ExifToolHelper

def get_videos_metadata(video_paths, keys=['QuickTime:CameraSerialNumber', 'QuickTime:Model']):
    results = dict()
    with ExifToolHelper() as et:
        for meta in et.get_metadata(video_paths):
            result = dict()
            for key in keys:
                result[key] = meta[key]
            results[meta['SourceFile']] = result
    return results
