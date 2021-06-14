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
        video_file = results['image_file']
		if video_file not in self.readers:
			self.readers[video_file] = mmcv.VideoReader(video_file)

		img = self.readers[video_file].get_frame(int(results['frame']))

        img = mmcv.imread(image_file, self.color_type, self.channel_order)

        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        results['img'] = img
        return results
