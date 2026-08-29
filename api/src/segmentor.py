from os import makedirs
from os.path import join

from nibabel import load as nib_load
import numpy as np

from .dirs import debug_dir
from .utils import (
    get_volume_absolute_path,
    get_segmentation_path_from_volume_path,
    save_debug_images,
)


class Segmentor:
    def __init__(self, model_name, use_cuda) -> None:
        from src.models.model import Model

        self.model = Model(use_cuda)

        self.debug_save_latest_dir = join(debug_dir, "latest", "segmentor")
        makedirs(self.debug_save_latest_dir, exist_ok=True)
    
    def load_volume(self, volume_path):
        volume_path = volume_path.replace("\\","/")
        volume_absolute_path = get_volume_absolute_path(volume_path)
        
        nii_img = nib_load(volume_absolute_path)
        nii_data = nii_img.get_fdata()

        self.axial_slices = [None] * nii_data.shape[2]
        self.windowed_axial_slices = [None] * nii_data.shape[2]
        self.preprocessed_axial_slices = [None] * nii_data.shape[2]
        self.axial_model_outputs = [None] * nii_data.shape[2]
        self.postprocessed_axial_model_outputs = [None] * nii_data.shape[2]
        for z in range(nii_data.shape[2]):
            axial_slice = nii_data[:, :, z]

            self.axial_slices[z] = axial_slice

            batched_slice = self.model.preprocess_slice(axial_slice)
            self.preprocessed_axial_slices[z] = batched_slice

            if hasattr(self.model, 'window_slice'):
                self.windowed_axial_slices[z] = self.model.window_slice(axial_slice)
        
        self.loaded_volume_data = nii_data
        self.loaded_volume_affine = nii_img.affine
        self.loaded_volume_shape = nii_data.shape

        self.loaded_volume_path = volume_path
        self.loaded_volume_absolute_path = volume_absolute_path
        self.segmentation_path = get_segmentation_path_from_volume_path(self.loaded_volume_path)
        self.segmentation_absolute_path = get_segmentation_path_from_volume_path(self.loaded_volume_absolute_path)
    
    def segment_single_slice(
            self,
            volume_path,
            slice_type,
            slice_index,
            task_id
        ):
        if slice_type != "axial":
            return
        
        if volume_path != getattr(self, 'loaded_volume_path', None):
            self.load_volume(volume_path)

        model_output = self.model.segment(self.preprocessed_axial_slices[slice_index], task_id)

        self.axial_model_outputs[slice_index] = model_output

        postprocessed_output_masks, index_to_label_map = self.model.postprocess(
            model_output,
            # np.zeros((512, 512), dtype=bool),
            output_size=(self.loaded_volume_shape[0], self.loaded_volume_shape[1])
        )

        save_debug_images(
            self.debug_save_latest_dir,
            {
                'slice': self.axial_slices[slice_index],
                # 'windowed_slice': self.windowed_axial_slices[slice_index],
                # 'preprocessed_slice': self.preprocessed_axial_slices[slice_index],
                # 'model_output': model_output,
                # 'postprocessed_model_output': postprocessed_model_output,
            }
        )

        tracked_output_masks = postprocessed_output_masks
        output_masks = [postprocessed_output_masks, tracked_output_masks]

        return output_masks, index_to_label_map
    
    def interactive_segment_single_slice(
            self,
            volume_path,
            slice_type,
            slice_index,
            segment_masks,
            task_id
        ):
        if slice_type != "axial":
            return
        
        if volume_path != getattr(self, 'loaded_volume_path', None):
            self.load_volume(volume_path)
        
        # if getattr(self, 'axial_model_outputs', [None] * 1024)[slice_index] is None:
        #    print("Redirecting ...")
        #    return self.segment_single_slice(volume_path, slice_type, slice_index, f"{task_id}---redirect")
        
        preprocessed_segment_masks = [self.model.preprocess_mask(mask) for mask in segment_masks]
                
        model_output = self.model.interactive_segment(
            self.preprocessed_axial_slices[slice_index],
            preprocessed_segment_masks,
            task_id
        )

        self.axial_model_outputs[slice_index] = model_output

        postprocessed_output_masks, index_to_label_map = self.model.postprocess(
            model_output,
            # np.zeros((512, 512), dtype=bool),
            output_size=(self.loaded_volume_shape[0], self.loaded_volume_shape[1])
        )
        
        # save_debug_images(
        #     self.debug_save_latest_dir,
        #     {
        #         'slice': self.axial_slices[slice_index],
        #         'windowed_slice': self.windowed_axial_slices[slice_index],
        #         'received_model_prev_output': model_prev_output,
        #         'model_prev_output': self.axial_model_outputs[slice_index],
        #         'scribbles_fg_arr': scribbles_fg_arr,
        #         'scribbles_bg_arr': scribbles_bg_arr,
        #         'postprocessed_model_prev_output': self.postprocessed_axial_model_outputs[slice_index],
        #         'preprocessed_slice': self.preprocessed_axial_slices[slice_index],
        #         'preprocessed_scribbles_fg_arr': preprocessed_scribbles_fg_arr,
        #         'preprocessed_scribbles_bg_arr': preprocessed_scribbles_bg_arr,
        #         'model_output': model_output,
        #         'postprocessed_model_output': postprocessed_model_output,
        #     }
        # )

        tracked_output_masks = postprocessed_output_masks
        output_masks = [postprocessed_output_masks, tracked_output_masks]

        return output_masks, index_to_label_map
