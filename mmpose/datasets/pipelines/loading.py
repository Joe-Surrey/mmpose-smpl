import lmdb
from pathlib import Path
import cv2
import mmcv
import pickle
from ..builder import PIPELINES



@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self, to_float32=False,  color_type='color', channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order)

        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        results['img'] = img
        return results


@PIPELINES.register_module()
class LoadImageFromlmdb:
    """Loading pngs from lmdb.
    image_file must be in the format shardname%w_or_n%file%frame

    """

    def __init__(self):
        print(f"Making new object {id(self)}")
        self.db = None
        self.shard_name = None


    def __call__(self, results):
        """Loading image from file."""

        shard_name, w_or_n, video_file, frame = results['image_file'].split("%")
        key = pickle.dumps(Path(shard_name).stem + "%" + w_or_n + "%" + video_file + "%" + frame)

        try:
            with self.db.begin() as txn:
                value = txn.get(key=key)
                img = cv2.imdecode((pickle.loads(value)[1]), cv2.IMREAD_COLOR)
        except TypeError as e:
            print(f"Failed on: {results['image_file']}")
            raise e

        if img is None:
            raise ValueError(f'Fail to read frame {frame} of {(w_or_n,video_file)} in {shard_name}')

        results['img'] = img
        return results

    def unload(self):
        """Unload the lmdb"""

        if not self.db is None:
            print("closing")
            self.db.close()
            self.db = None

    def load(self,results):
        """Load the lmdb"""

        shard_name, w_or_n, video_file, frame = results['image_file'].split("%")

        print(f"Using {shard_name}, {id(self)}")
        self.db = lmdb.open(
            path=shard_name + ".lmdb",
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )

        self.shard_name = shard_name

@PIPELINES.register_module()
class LoadImageFromMP4:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.readers = {}

    def __call__(self, results):
        """Loading image from file."""
        video_file, frame = results['image_file'].split("%")
        if video_file not in self.readers:
            self.readers[video_file] = mmcv.VideoReader(video_file)

        # print(results)

        img = self.readers[video_file].get_frame(int(frame))

        #if img is None:
        #    raise ValueError('Fail to read frame {} of {}'.format(frame, image_file))

        results['img'] = img
        return results