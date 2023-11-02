import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Callable

import torch
from accelerate.utils import send_to_device

logger = logging.getLogger(__name__)

class DataType(Enum):
    ScalarTensor = 0  # used for the example info and game status, such as the reward of a single step,
    Tensor1D = 1  # used for the sequence of tokens, token's score, token's position, etc.
    Tensor2D = 2  # used for the embedding of tokens, the hidden state of a single step, etc.
    Tensor3D = 3  # used for the attention matrix.
    Tensor4D = 4  # used for the attention matrix, head as an independent dimension.
    Tensor5D = 5  # used for activation matrix.

    # The composition of a batch is not converted to a torch.Tensor:
    IntValue = 10  # used for the length of the sequence, etc.
    BoolValue = 11  # used for the game status, etc.
    FloatValue = 12  # other values

    Info = 20  # used for the text of the sequence, etc.


TensorDataType = [DataType.ScalarTensor, DataType.Tensor1D, DataType.Tensor2D, DataType.Tensor3D, DataType.Tensor4D, DataType.Tensor5D]

@dataclass
class DataStructure:
    datatype: DataType
    name: str
    pin_on_cpu: bool = False
    dimension_names: List[str] = None
    padding_dimensions: List[int] = None  # For a 4D tensor [3, 4], means we need to padding the 3rd and 4th dimension
    need_padding_mask: bool = False
    need_log: bool = False

    def __hash__(self) -> int:
        return hash(self.datatype) + hash(self.name)

    def __eq__(self, other) -> bool:
        return self.datatype == other.datatype and self.name == other.name

    @property
    def is_tensor(self) -> bool:
        return self.datatype in TensorDataType

    def check(self, value, offset=0) -> bool:
        if self.datatype in TensorDataType:
            if isinstance(value, torch.Tensor):
                if self.datatype == DataType.ScalarTensor:
                    if offset == 0: # When ItemTensorSet
                        assert (len(value.shape) == 1 and value.shape[0] == 1) or (len(value.shape) == 0)
                    else: # When BatchTensorSet
                        assert len(value.shape) == 1
                elif self.datatype == DataType.Tensor1D:
                    assert len(value.shape) == 1 + offset
                elif self.datatype == DataType.Tensor2D:
                    assert len(value.shape) == 2 + offset
                elif self.datatype == DataType.Tensor3D:
                    assert len(value.shape) == 3 + offset
                elif self.datatype == DataType.Tensor4D:
                    assert len(value.shape) == 4 + offset
                elif self.datatype == DataType.Tensor5D:
                    assert len(value.shape) == 5 + offset
            else:
                raise ValueError(f"Data type error: {self.name} is not a tensor, but {type(value)}")
            
        elif self.datatype == DataType.IntValue:
            if offset == 0:
                assert isinstance(value, int)
            else:
                assert isinstance(value, list) and all([isinstance(v, int) for v in value])
        elif self.datatype == DataType.BoolValue:
            if offset == 0:
                assert isinstance(value, bool)
            else:
                assert isinstance(value, list) and all([isinstance(v, bool) for v in value])
        elif self.datatype == DataType.FloatValue:
            if offset == 0:
                assert isinstance(value, float)
            else:
                assert isinstance(value, list) and all([isinstance(v, float) for v in value])

        return True


class TensorSet(dict):
    """
        TensorSet is a dict of tensors, which is used to store the features of a single example.
    """
    _data_structure_dict: Dict[str, DataStructure] = {}  # the data structure of the tensor set
    __check_data_type_offset = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        elif name in self._data_structure_dict:
            return None
        else:
            raise AttributeError(f"Unknown data structure: {name}")

    def __setattr__(self, name: str, value):
        if name == "_data_structure_dict":
            self.__dict__[name] = value
        elif name in self._data_structure_dict:
            data_structure = self._data_structure_dict[name]
            if data_structure.check(value, self.__check_data_type_offset):
                if data_structure.is_tensor and data_structure.pin_on_cpu:
                    self[name] = value.cpu() # auto send to cpu
                else:
                    self[name] = value
        else:
            super().__setattr__(name, value)

    def name_structure_value_iter(self) -> Iterator:
        for name, data_structure in self._data_structure_dict.items():
            yield name, data_structure, self.get(name)

    def __repr__(self):
        # print the data structure in a dict view, including the data type and the shape of the tensor
        description = [f"{self.__class__.__name__}(\n"]
        for name, data_structure in self._data_structure_dict.items():
            description.append(f"\t{name}: {data_structure.datatype.name}(")
            if name in self.keys():
                dim_description = []
                if isinstance(self[name], torch.Tensor) and data_structure.dimension_names is not None:
                    for dim_name, dim_size in zip(data_structure.dimension_names, self[name].shape):
                        dim_description.append(f"{dim_name}={dim_size}")
                else:
                    dim_description.append(f"value={self[name]}")
            else:
                dim_description = []
                description += "None"
            description.append(", ".join(dim_description) + ")\n")
        description.append(")")
        return "".join(description)

    def send_to_device(self, device):
        for name, data_structure, value in self.name_structure_value_iter():
            if data_structure.is_tensor and value is not None and data_structure.pin_on_cpu == False:
                self[name] = value.to(device)
        return self


class ItemTensorSet(TensorSet):
    """
        ItemTensorSet is a dict of tensors, which is used to store the features of a single example.
    """
    __check_data_type_offset = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BatchTensorSet(TensorSet):
    """
        BatchTensorSet is a dict of tensors, which is used to store the features of a batch of examples.
    """
    __check_data_type_offset = 1
    __ItemTensorSetClass = None

    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def __setattr__(self, name: str, value):
        if name in self._data_structure_dict:
            data_structure = self._data_structure_dict[name]
            if data_structure.check(value, self.__check_data_type_offset):
                self[name] = value
                if data_structure.is_tensor and data_structure.need_padding_mask and self.get(f"{name}_mask") is None:
                    logger.warning(f"[BatchTensorSet] You're trying to manually set a tensor data that requires a padding mask.")
        else:
            super().__setattr__(name, value)

    def keep_items(self, indices: List[int]):
        for name, data_structure, value in self.name_structure_value_iter():
            if len(indices) == 0:
                self[name] = None
            elif value is not None:
                if data_structure.is_tensor:
                    self[name] = torch.index_select(value, 0, torch.tensor(indices))
                else:  # For non-tensor types, assuming they are lists
                    self[name] = [self[name][i] for i in range(self.batch_size) if i in indices]
        self.batch_size = len(indices)

    def delete_items(self, indices: List[int]):
        re_indices = [i for i in range(self.batch_size) if i not in indices]
        self.keep_items(re_indices)

    def split_into_items(self) -> List[ItemTensorSet]:
        item_tensor_sets = []
        for i in range(self.batch_size):
            item_tensor_set = self.__ItemTensorSetClass()
            for name, data_structure, value in self.name_structure_value_iter():
                
                if value is not None:
                    # Undo padding by the info from padding_dimensions and their corresponding dimension_names
                    if data_structure.padding_dimensions is not None:
                        slices = [slice(0, value.size()[dim]) for dim in range(len(value.size()))]
                        for padding_dimension_idx in data_structure.padding_dimensions:
                            undo_padding_ref_dimension_name = data_structure.dimension_names[padding_dimension_idx]
                            undo_padding_cutoff_position = self[name].size()[padding_dimension_idx]

        
                    item_tensor_set[name] = value[i]
                    # if data_structure.is_tensor:
                    #     item_tensor_set[name] = value[i]
                    # else:
                    #     item_tensor_set[name] = value[i] # TODO need to check
            item_tensor_sets.append(item_tensor_set)
        return item_tensor_sets


def calculate_target_dimensions(
    item_tensor_sets: List[ItemTensorSet],
    data_structure_dict: Dict[str, DataStructure]
) -> Dict[str, List[int]]:
    dims = defaultdict(list)
    target_dims = {}

    # Collecting dimensional information
    for item_tensor in item_tensor_sets:
        for key, data_structure, value in item_tensor.name_structure_value_iter():
            if data_structure.is_tensor:
                dims[key].append(list(value.size()))

    # Calculate the target dimension
    for key, shapes in dims.items():
        max_shape = [max(dim) for dim in zip(*shapes)]
        padding_dims = data_structure_dict[key].padding_dimensions
        if padding_dims is not None:
            for dim_idx in padding_dims:
                max_shape[dim_idx] = max(shapes, key=lambda x: x[dim_idx])[dim_idx]
        target_dims[key] = max_shape

    return target_dims

def pad_item_tensor_sets(item_tensors: List[ItemTensorSet], target_dims: Dict[str, List[int]], data_structure_dict) -> Dict[str, torch.Tensor]:
    data_dict = {}
    for key, target_shape in target_dims.items():
        tensor_list = [item_tensor[key] for item_tensor in item_tensors]
        data_structure = data_structure_dict[key]
        
        if data_structure.datatype == DataType.ScalarTensor:
            data_dict[key] = torch.cat(tensor_list)
            
        elif data_structure.is_tensor:
            padding_dims = data_structure.padding_dimensions
            if padding_dims is None:
                padding_dims = []

            padded_tensors = []
            padding_masks = [] if data_structure.need_padding_mask else None

            for tensor in tensor_list:
                tensor_shape = list(tensor.size())
                
                # Create a padding mask filled with ones
                if data_structure.need_padding_mask:
                    padding_mask = torch.ones(*target_shape, dtype=torch.bool)
                    
                # pad_size = [(target_shape[dim] - tensor_shape[dim]) for dim in range(len(tensor_shape))]
                slices = [slice(0, tensor_shape[dim]) for dim in range(len(tensor_shape))]
                
                # Prepare padded tensor
                padded_tensor = torch.zeros(*target_shape, dtype=tensor.dtype)
                padded_tensor[tuple(slices)] = tensor
                
                # Update the padding mask
                if data_structure.need_padding_mask:
                    padding_mask[tuple(slices)] = 1
                    padding_masks.append(padding_mask)
                
                padded_tensors.append(padded_tensor)
            
            data_dict[key] = torch.stack(padded_tensors)
            
            # Add padding masks to data_dict
            if data_structure.need_padding_mask:
                data_dict[f"{key}_mask"] = torch.stack(padding_masks)
                
        else:
            data_dict[key] = tensor_list

    return data_dict

def from_item_tensors(BatchTensorSetClass, item_tensors: List[ItemTensorSet]) -> BatchTensorSet:
    data_structure_dict= BatchTensorSetClass._data_structure_dict
    target_dims = calculate_target_dimensions(item_tensors,data_structure_dict)
    data_dict = pad_item_tensor_sets(item_tensors, target_dims,data_structure_dict)
    batch = BatchTensorSetClass(batch_size = len(item_tensors), **data_dict)

    return batch


class DataStructureFactory:
    def __init__(self):
        self._data_structure_dict: Dict[str, DataStructure] = {}

    def register(self, data_structure: DataStructure):
        self._data_structure_dict[data_structure.name] = data_structure
        if data_structure.need_padding_mask:
            padding_mask_data_structure = DataStructure(data_structure.datatype, 
                                                        f"{data_structure.name}_mask", 
                                                        data_structure.pin_on_cpu, 
                                                        data_structure.dimension_names, 
                                                        data_structure.padding_dimensions, 
                                                        False)

            self._data_structure_dict[f"{data_structure.name}_mask"] = padding_mask_data_structure

    def __call__(self, name: str) -> DataStructure:
        return self._data_structure_dict.get(name)

    def new_item_tensor_set_class(self, name, data_structure_names: List[DataStructure]):
        """
        Generate a new TensorSet subclass with predefined DataStructures.
        :param name: The name of the new class.
        :param fields: A dictionary containing the names and DataStructures of the predefined fields.
        """
        data_structure_dict = {}
        for ds_name in data_structure_names:
            data_structure = self._data_structure_dict.get(ds_name)
            if data_structure is None:
                raise ValueError(f"Unknown data structure: {ds_name}, please register it first.")
            data_structure_dict[ds_name] = data_structure

        CustomTensorSet = type(
            name,  # class name
            (ItemTensorSet,),  # base classes
            {'_data_structure_dict': data_structure_dict}  # class attributes
        )
        return CustomTensorSet

    def new_batch_tensor_set_class(self, name, data_structure_names: List[DataStructure], ItemTensorSetClass):
        """
        Generate a new TensorSet subclass with predefined DataStructures.
        :param name: The name of the new class.
        :param fields: A dictionary containing the names and DataStructures of the predefined fields.
        """
        data_structure_dict = {}
        for ds_name in data_structure_names:
            data_structure = self._data_structure_dict.get(ds_name)
            if data_structure is None:
                raise ValueError(f"Unknown data structure: {ds_name}, please register it first.")
            data_structure_dict[ds_name] = data_structure
            if data_structure.need_padding_mask:
                data_structure_dict[f"{ds_name}_mask"] = self._data_structure_dict.get(f"{ds_name}_mask")

        CustomTensorSet = type(
            name,  # class name
            (BatchTensorSet,),  # base classes
            {
                '_data_structure_dict': data_structure_dict,
                '__ItemTensorSetClass': ItemTensorSetClass
            }# class attributes
        )
        
        return CustomTensorSet

    def new_tensor_set_class(self, name, data_structure_names: List[DataStructure]):
        ItemTensorSetClass = self.new_item_tensor_set_class(f"Item{name}", data_structure_names)
        BatchTensorSetClass = self.new_batch_tensor_set_class(f"Batch{name}", data_structure_names, ItemTensorSetClass)

        return ItemTensorSetClass, BatchTensorSetClass

F = DataStructureFactory()

# Some default data structures
F.register(DataStructure(DataType.Info, "id", need_log=True))
F.register(DataStructure(DataType.Info, "text", need_log=True))
F.register(DataStructure(DataType.Info, "token_word_position_map"))

F.register(DataStructure(DataType.IntValue, "original_seq_length", need_log=True))
F.register(DataStructure(DataType.IntValue, "squence_length", need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "golden_label", pin_on_cpu=True, need_log=True))

F.register(DataStructure(DataType.ScalarTensor, "original_logits", pin_on_cpu=True))
F.register(DataStructure(DataType.ScalarTensor, "original_pred_label", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "original_prob", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "original_acc", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "original_loss", pin_on_cpu=True))

F.register(DataStructure(DataType.ScalarTensor, "perturbed_logits", pin_on_cpu=True))
F.register(DataStructure(DataType.ScalarTensor, "perturbed_pred_label", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "perturbed_prob", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "perturbed_acc", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "perturbed_loss", pin_on_cpu=True))

F.register(DataStructure(DataType.ScalarTensor, "delta_logits", pin_on_cpu=True))
F.register(DataStructure(DataType.ScalarTensor, "delta_prob", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "delta_acc", pin_on_cpu=True, need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "delta_loss", pin_on_cpu=True))

F.register(DataStructure(DataType.ScalarTensor, "step_reward", need_log=True))
F.register(DataStructure(DataType.IntValue, "game_step"))
F.register(DataStructure(DataType.ScalarTensor, "episode_reward", need_log=True))

F.register(DataStructure(DataType.ScalarTensor, "q_loss", need_log=True))

F.register(DataStructure(DataType.ScalarTensor, "if_done", need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "if_success", need_log=True))
F.register(DataStructure(DataType.ScalarTensor, "if_timeout", need_log=True))

F.register(DataStructure(DataType.Tensor1D, "input_ids", dimension_names=["squence_length"], padding_dimensions=[0]))
F.register(DataStructure(DataType.Tensor1D, "attention_mask", dimension_names=["squence_length"], padding_dimensions=[0]))
F.register(DataStructure(DataType.Tensor1D, "special_tokens_mask", dimension_names=["squence_length"], padding_dimensions=[0]))
F.register(DataStructure(DataType.Tensor1D, "token_type_ids", dimension_names=["squence_length"], padding_dimensions=[0]))
F.register(DataStructure(DataType.Tensor1D, "position_ids", dimension_names=["squence_length"], padding_dimensions=[0]))


if __name__ == "__main__":
    # test
    import random

    logging.basicConfig(level=logging.WARNING)
    
    F.register(DataStructure(DataType.Tensor2D, "token_embeddings", dimension_names=["squence_length", "embedding_size"], padding_dimensions=[0]))
    F.register(DataStructure(DataType.Tensor4D, "attention_matrix", dimension_names=["head", "head_dim", "squence_length1", "squence_length2", ], padding_dimensions=[2, 3]))
    F.register(DataStructure(DataType.Tensor5D, "activation_matrix", dimension_names=["head", "head_dim", "squence_length1", "squence_length2", "squence_length3"], padding_dimensions=[2, 3, 4]))


    tensor_set_blueprint = ["original_seq_length", "input_ids", "token_embeddings", "attention_matrix", "activation_matrix", "reward"]

    ItemTransformerTensorSet = F.new_item_tensor_set_class("TransformerTensorSet", tensor_set_blueprint)
    BatchTransformerTensorSet = F.new_batch_tensor_set_class("BatchTransformerTensorSet", tensor_set_blueprint)

    item_tensor_sets = []
    batch_size = 4

    for i in range(batch_size):
        random_sequence_length = random.randint(2, 10)
        item_tensor_set = ItemTransformerTensorSet()
        item_tensor_set.original_seq_length = random_sequence_length
        item_tensor_set.input_ids = torch.tensor([i for i in range(random_sequence_length)])
        item_tensor_set.token_embeddings = torch.randn(random_sequence_length, 4)
        item_tensor_set.attention_matrix = torch.randn(3, 4, random_sequence_length, random_sequence_length)
        item_tensor_set.activation_matrix = torch.randn(3, 4, random_sequence_length, random_sequence_length, random_sequence_length)
        item_tensor_set.reward = torch.randn(1)
        item_tensor_sets.append(item_tensor_set)
        print(item_tensor_set)

    batch_tensor_set = from_item_tensors(BatchTransformerTensorSet, item_tensor_sets)
    print(batch_tensor_set)

    transformer_tensor_set = BatchTransformerTensorSet(batch_size)
    batch_size = 4
    transformer_tensor_set.original_seq_length = [random.randint(2, 10) for _ in range(batch_size)]
    transformer_tensor_set.input_ids = torch.randn(batch_size, 3)
    transformer_tensor_set.token_embeddings = torch.randn(batch_size, 3, 4)  # [batch_size, seq_len, embedding_size]
    transformer_tensor_set.attention_matrix = torch.randn(batch_size, 3, 3, 4, 4)  # [batch_size, head, head_dim, seq_len, seq_len]
    # transformer_tensor_set.activation_matrix = torch.randn(batch_size, 3, 3, 4, 4, 4) # [batch_size, head, head_dim, seq_len, seq_len, seq_len]
    transformer_tensor_set.reward = torch.randn(batch_size)

    print(transformer_tensor_set)

    print("delete two items:")
    transformer_tensor_set.delete_items([0, 2])
    print(transformer_tensor_set)
    print("batch_size:",transformer_tensor_set.batch_size)

    print("delete two items:")
    transformer_tensor_set.delete_items([0, 1])
    print(transformer_tensor_set)
    print("batch_size:",transformer_tensor_set.batch_size)

