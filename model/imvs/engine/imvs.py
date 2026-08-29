from model.imvs.engine.sma import SMA
from model.imvs.engine.vmt import VMT


class IMVS:
    def __init__(self, use_cuda: bool = False) -> None:
        self.sma = SMA(use_cuda)
        self.vmt = VMT(use_cuda)
    
    def infer(self, images, scribbles):
        preprocessed_image = self.sma.preprocess_slice(images[0])

        if scribbles is not None:    
            # preprocessed_scribbles = self.sma.preprocess_mask(scribbles)
            
            background_scribbles = scribbles == False
            foreground_scribbles = scribbles == True

            sma_output = self.sma.interactive_segment(preprocessed_image, [background_scribbles, foreground_scribbles], "task1")
            return sma_output
        
        sma_output = self.vmt.propagate(preprocessed_image, "task1")
        return sma_output
