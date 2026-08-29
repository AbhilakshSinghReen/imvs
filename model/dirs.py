from os.path import dirname, join


segmentors_src_dir = dirname(__file__)
segmentors_dir = dirname(segmentors_src_dir)
backend_dir = dirname(segmentors_dir)
eso_label_dir = dirname(backend_dir)
data_dir = join(eso_label_dir, "data")
datasets_dir = join(data_dir, "datasets")
debug_dir = join(data_dir, "debug")
