from copy import deepcopy
from datetime import datetime
from os import listdir, makedirs
from os.path import dirname, join
from yaml import dump as yaml_dump, FullLoader as yaml_FullLoader, load as yaml_load

from cv2 import resize
import numpy as np
from segmentation_models_pytorch import UnetPlusPlus
import torch
from torchvision import transforms
from torch.optim import Adam as AdamOptimizer
from torch.nn import BCELoss
from torch.nn.functional import interpolate

from model.utils import save_debug_images, print_value_meta, resize_np_bool_arr


class SMA:
    def __init__(self, use_cuda: bool = False) -> None:
        self.name = "SMA
        self.description = ""
        self.input_dims = (256, 256)

        self.weights_dir = join(dirname(__file__), "weights")
        self.debug_dir = join(dirname(dirname(dirname(dirname(dirname(dirname(__file__)))))), "data", "debug")
        self.debug_save_latest_dir = join(self.debug_dir, "latest", "models", "sma")
        self.model_params_file_path = join(dirname(__file__), "params.yaml")

        self.load_params()

        self.using_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.using_cuda else torch.device("cpu")
        self.dtype = torch.cuda.FloatTensor if self.using_cuda else torch.Tensor

        self.last_loaded_checkpoint_path = self.get_latest_checkpoint()
        self.last_saved_checkpoint_path = self.last_loaded_checkpoint_path

        self.model = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_depth=4,
            encoder_weights="imagenet",
            decoder_channels=(256, 128, 64, 32),
            in_channels=1,
            classes=1,
            activation=None,
            aux_params=None
        )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.last_loaded_checkpoint_path))

        self.optimizer = AdamOptimizer(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )
        self.criterion = BCELoss()

        self.preprocessing_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=self.input_dims)
        ])

        if use_cuda and not self.using_cuda:
            print("CUDA is not available to Torch, using CPU instead.")

        makedirs(self.debug_save_latest_dir, exist_ok=True)

    def get_latest_checkpoint(self) -> str:
        all_checkpoints = listdir(self.weights_dir)
        if ".gitkeep" in all_checkpoints:
            all_checkpoints.remove(".gitkeep")
        
        if len(all_checkpoints) == 0:
            print("No checkpoints found in the weights directory.")

        return join(self.weights_dir, all_checkpoints[0])
    
    def load_params(self):
        with open(self.model_params_file_path, 'r') as model_params_file:
            model_params = yaml_load(model_params_file, Loader=yaml_FullLoader)

        formatted_datetime = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        file_update_comments = f"# Parameters last read at: {formatted_datetime}"

        with open(self.model_params_file_path, 'w') as model_params_file:
            model_params_file.writelines(file_update_comments)
            model_params_file.write("\n")
            yaml_dump(
                model_params,
                model_params_file,
                default_flow_style=False
            )

        self.params = model_params
        
        return model_params
    
    def save_weights(self):
        pass

    def log(self, string):
        print(string)
    
    #region Preprocessing
    def window_slice(self, image):
        window_level = self.params['window']['level']
        window_width = self.params['window']['width']

        lower_bound = window_level - window_width // 2
        upper_bound = window_level + window_width // 2

        clipped_image = np.clip(image, lower_bound, upper_bound)

        windowed_image = ((clipped_image - lower_bound) / (upper_bound - lower_bound)) * 255
        windowed_image = windowed_image.astype(np.uint8)

        return windowed_image
    
    def preprocess_slice(self, image):
        windowed_image = self.window_slice(image)
        windowed_image = windowed_image / 255

        transformed_image = self.preprocessing_transform(windowed_image)
        transformed_image = transformed_image.type(self.dtype)

        batched_image = transformed_image.unsqueeze(0)
        batched_image = batched_image.to(self.device)

        return batched_image

    def preprocess_mask(self, mask):
        cv2_image = mask.astype(np.uint8) * 255
        resized_cv2_image = resize(cv2_image, self.input_dims)
        resized_mask = (resized_cv2_image > 0)
        return resized_mask

    def postprocess(self, output, output_size=None):
        if output is None:
            output_size = self.input_dims
        
        output = output[0][0]
        output = output.ge(0.5)
        output = interpolate(output.unsqueeze(0).unsqueeze(0).float(), size=output_size, mode="nearest")
        output = output.bool().squeeze(0).squeeze(0)

        if self.using_cuda:
            output = output.cpu()
        
        output = output.numpy()

        # return output

        index_to_label_map = {
            "0": "background",
            "1": "liver",
        }

        background_mask = np.zeros_like(output)

        all_masks = [
            background_mask,
            output,
        ]
        all_masks = [resize_np_bool_arr(mask, output_size) for mask in all_masks]
        all_masks_np = np.stack(all_masks, axis=0)

        return all_masks_np, index_to_label_map
    #endregion

    #region Core
    def compute_loss(
            self,
            target_prediction,
            current_prediction,
            original_model,
            new_model
        ):
        target_prediction = target_prediction.detach().clone()
        target_prediction = target_prediction.to(self.device)

        #current_prediction = current_prediction.detach().clone()
        current_prediction = current_prediction.to(self.device)

        classification_loss = self.criterion(current_prediction, target_prediction)
        
        return classification_loss
        # classification_loss_sum = torch.sum(classification_loss.detach() * current_prediction)

        # regularization_loss = 0
        # for p1, p2 in zip(original_model.parameters(), new_model.parameters()):
        #     regularization_loss += torch.norm(p1 - p2)
        # regularization_loss *= self.params['regularization_factor']

        # log_string = "Model.compute_loss: "
        # log_string += f"classification_loss_sum={round(classification_loss_sum, 8)} | "
        # log_string += f"regularization_loss={round(regularization_loss, 8)}"
        # self.log(log_string)

        # total_loss = classification_loss_sum + regularization_loss

        # return loss, total_loss

    def segment(self, preprocessed_image, task_id):
        self.load_params()

        output = self.model.forward(preprocessed_image)
        output = torch.sigmoid(output)

        save_debug_images(
            self.debug_save_latest_dir,
            {
                'model_output': output,
            }
        )

        return output

    def interactive_segment(
            self,
            preprocessed_image,
            preprocessed_scribbles_masks,
            task_id
        ):
        self.load_params()

        new_model = deepcopy(self.model)
        new_optimizer = self.optimizer = AdamOptimizer(
            new_model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )

        initial_prediction = self.segment(preprocessed_image, "internal_redirect")

        preprocessed_bg_mask = preprocessed_scribbles_masks[0]
        preprocessed_fg_mask = preprocessed_scribbles_masks[1]
        
        target_prediction = initial_prediction.detach().clone()
        mask_num_rows, mask_num_cols = preprocessed_fg_mask.shape
        for i in range(mask_num_rows):
            for j in range(mask_num_cols):
                if preprocessed_bg_mask[i][j]:
                    target_prediction[0][0][i][j] = 0
                elif preprocessed_fg_mask[i][j]:
                    target_prediction[0][0][i][j] = 1
        
        target_prediction = target_prediction.to(self.device)
        
        save_debug_images(
                self.debug_save_latest_dir,
                {
                    'target_prediction': target_prediction,
                    'preprocessed_scribbles_fg_arr': preprocessed_fg_mask,
                    'preprocessed_scribbles_bg_arr': preprocessed_bg_mask,
                }
        )

        save_debug_images(
            self.debug_save_latest_dir,
            {
                'multi_label_target_prediction': target_prediction,
                'multi_label_preprocessed_scribbles_masks': preprocessed_scribbles_masks,
            }
        )
        
        
        #print(new_model.is_cuda)
        #print(self.model.is_cuda)
        #print(target_prediction.is_cuda)
        
        #print(new_model.is_cuda)

        

        current_iteration = 0
        sum_constraint_error = 1e5
        
        best_loss = 1e5
        best_new_prediction = None

        while \
          sum_constraint_error > self.params['break_constraint_error'] \
          and current_iteration < self.params['max_iterations']:
            new_optimizer.zero_grad()

            new_prediction = new_model.forward(preprocessed_image)
            new_prediction = torch.sigmoid(new_prediction)

            #print(type(target_prediction))
            #print(target_prediction.dtype)
            #print(target_prediction.shape)
            #print(type(new_prediction))
            #print(new_prediction.dtype)
            #print(new_prediction.shape)
            #print(type(preprocessed_fg_mask))
            #print(preprocessed_fg_mask.dtype)
            #print(preprocessed_fg_mask.shape)
            #print(type(preprocessed_bg_mask))
            #print(preprocessed_bg_mask.dtype)
            #print(preprocessed_bg_mask.shape)

            constraint_error = self.compute_loss(
                target_prediction,
                new_prediction,
                self.model,
                new_model
            )
            #print()
            #print("constraint_error")
            #print(constraint_error)

            constraint_error.backward()
            new_optimizer.step()
            
            save_debug_images(
                self.debug_save_latest_dir,
                {
                    # f'model_output_iteration-{current_iteration}': new_prediction,
                }
            )

            if best_new_prediction is None:
                best_new_prediction = new_prediction

            if best_loss > constraint_error.item() and current_iteration > 0:
                print("Best prediction updated")
                best_loss = constraint_error.item()
                best_new_prediction = new_prediction

            log_string = "Model.interactive_segment: "
            log_string += f"current_iteration={current_iteration} | "
            log_string += f"loss={round(constraint_error.item(), 8)}"
            self.log(log_string)
            
            sum_constraint_error = torch.sum(constraint_error)
            current_iteration += 1

        self.model = new_model

        save_debug_images(
            self.debug_save_latest_dir,
            {
                'model_output': new_prediction,
            }
        )

        return best_new_prediction
    #endregion
