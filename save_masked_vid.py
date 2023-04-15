import image
import pipeline as pl
from rtmdet import RTMDet

device = 'cpu'
pipeline = pl.create_pipeline()
model = RTMDet(device)
image.save_masked_vid(model, pipeline)
