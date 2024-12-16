# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################

# Makes HQQ 4-bit and 2-bit (axis=1) compatbile with BitBlas: https://github.com/microsoft/BitBLAS

# Only works with: float16, axis=1.
# Only tested on 3090/4090 gpus.

import torch
import os
from torch import float16, Tensor

from ..core.quantize import HQQLinear
from ..core.peft import HQQLinearLoRA
from ..core.utils import cleanup

BITBLAS_TARGET = None
BITBLAS_DATABASE_PATH = None
BITBLAS_PROPAGATE_WEIGHTS = False

try:
    import bitblas  # noqa: F401
    BITBLAS_AVAILABLE = True
except Exception:
    BITBLAS_AVAILABLE = False

BITBLAS_INSTALL_HINT = "bitblas not installed. Please install via `pip install bitblas`."

def import_bitblas():
    # print("import_bitblas() called")
    global BITBLAS_DATABASE_PATH, BITBLAS_TARGET

    # guard against bitblas pip whl incompatible env`
    import bitblas

    bitblas.set_log_level("INFO")

    if BITBLAS_TARGET is None:
        from .bitblas_target_detector import patched_auto_detect_nvidia_target

        bitblas.auto_detect_nvidia_target = patched_auto_detect_nvidia_target
        BITBLAS_TARGET = patched_auto_detect_nvidia_target(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
        os.environ["TVM_TARGET"] = f"{BITBLAS_TARGET}"
        print(f"BITBLAS_TARGET {BITBLAS_TARGET}")

    if BITBLAS_DATABASE_PATH is None:
        # from importlib.metadata import version

        from bitblas.cache import get_database_path

        # bitblas_version = version(distribution_name="bitblas")
        # gptqmodel_version = version(distribution_name="gptqmodel")

        BITBLAS_DATABASE_PATH = get_database_path()

@torch.library.custom_op("hqq::matmul_bitblas", mutates_args=())
def matmul_bitblas(
    x: Tensor, W_q: Tensor, scale: Tensor, zero: Tensor, out_features: int, eng_tag: str
) -> Tensor:
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = HQQLinearBitBlas.ENG_CACHE[eng_tag](x, W_q, scale=scale, zeros=zero)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


@torch.library.register_fake("hqq::matmul_bitblas")
def matmul_bitblas_fake(
    x: Tensor, W_q: Tensor, scale: Tensor, zero: Tensor, out_features: int, eng_tag: str
) -> Tensor:
    return torch.empty(
        [x.shape[0], x.shape[1], out_features], device=W_q.device, dtype=scale.dtype
    )


class HQQLinearBitBlas(torch.nn.Module):
    BIT_TO_DTYPE = {
        4: "uint4",
        2: "uint2",
        1: "uint1",
    }
    DEFAULT_BATCHSIZE = [1]
    ENG_CACHE = {}
    # Creating a matmul config takes a lot of time, so we cache them to be re-used based on the matrix shapes / quant-setting

    def __init__(self, hqq_layer):
        super().__init__()

        import_bitblas()
        self.target = BITBLAS_TARGET
        
        self.bias = (
            hqq_layer.bias.data.clone() if (hqq_layer.bias is not None) else None
        )
        self.group_size = hqq_layer.meta["group_size"]
        self.nbits = hqq_layer.meta["nbits"]
        self.axis = hqq_layer.meta["axis"]
        self.shape = hqq_layer.meta["shape"]
        self.compute_dtype = torch.float16
        self.device = hqq_layer.device
        self.in_features = self.shape[1]
        self.out_features = self.shape[0]

        self.W_q = hqq_layer.unpack()
        self.zero = hqq_layer.meta["zero"]
        self.scale = hqq_layer.meta["scale"]

        # Reshape for group_size is None
        if self.group_size is None:
            self.group_size = 128
            self.W_q = self.W_q.reshape([-1, self.group_size])
            self.scale = self.reshape_meta_axis1(
                self.scale, self.group_size, self.shape
            )
            self.zero = self.reshape_meta_axis1(self.zero, self.group_size, self.shape)

        self.meta_shape_bitblas = (
            self.out_features,
            self.in_features // self.group_size,
        )
        self.meta_shape_hqq = (
            (self.in_features * self.out_features) // self.group_size,
            1,
        )  # axis=1

        self.eng_tag = (
            str(self.shape) + "_" + str(self.nbits) + "_" + str(self.group_size)
        )

        # matmul eng
        matmul_config = bitblas.MatmulConfig(
            M=HQQLinearBitBlas.DEFAULT_BATCHSIZE,
            N=self.out_features,
            K=self.in_features,
            A_dtype="float16",
            W_dtype=HQQLinearBitBlas.BIT_TO_DTYPE[self.nbits],
            accum_dtype="float16",  # float32 ?
            out_dtype="float16",
            layout="nt",
            with_bias=False,
            group_size=self.group_size,
            with_scaling=True,
            with_zeros=True,
            zeros_mode="original",
            # fast_decoding=True,
        )

        if self.eng_tag not in HQQLinearBitBlas.ENG_CACHE:
            HQQLinearBitBlas.ENG_CACHE[self.eng_tag] = self._get_or_create_bitblas_operator(
                config=matmul_config, enable_tuning=True
            )

        self.matmul_eng = HQQLinearBitBlas.ENG_CACHE[self.eng_tag]

        self.W_q = self.matmul_eng.transform_weight(self.W_q.reshape(self.shape)).to(self.device)
        self.zero = self.zero.view(self.meta_shape_bitblas)
        self.scale = self.scale.view(self.meta_shape_bitblas)
        torch.cuda.empty_cache()

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=self.target)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                print(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul
    
    @torch.no_grad()
    def reshape_meta_axis1(self, meta_tensor, new_group_size, shape):
        meta_tensor = meta_tensor.repeat([1, shape[1]]).reshape(shape)
        meta_tensor = torch.mean(
            meta_tensor.reshape([-1, new_group_size]), axis=1, keepdim=True
        )
        return meta_tensor

    @staticmethod
    def check(hqq_layer):
        is_valid = True
        is_valid = is_valid and hqq_layer.meta["nbits"] in [4, 2, 1]
        is_valid = is_valid and hqq_layer.meta["axis"] in [1]
        is_valid = is_valid and hqq_layer.meta["group_size"] in [None, 32, 64, 128, 256]
        is_valid = is_valid and hqq_layer.compute_dtype in [float16]
        return is_valid

    ###################### Forward/matmul ######################

    # @torch.jit.ignore()
    # def matmul(self, x: Tensor) -> Tensor:
    # 	origin_x_size = x.size()
    # 	x = x.reshape(-1, origin_x_size[-1])
    # 	c = self.matmul_eng(x, self.W_q, scale=self.scale, zeros=self.zero)
    # 	new_shape = origin_x_size[:-1] + (self.out_features,)
    # 	c = c.reshape(new_shape)
    # 	return c

    def matmul(self, x: Tensor) -> Tensor:
        #torch.cuda.set_device(self.device) #Need this with multi-gpu but it breaks with torch.compile
        return matmul_bitblas(
            x.to(self.device), self.W_q, self.scale, self.zero, self.out_features, self.eng_tag
        )

    # TODO without matmul
    def dequantize(self) -> Tensor:
        return self.matmul(
            torch.eye(self.in_features, dtype=self.compute_dtype, device=self.device)
        )[: self.in_features].t()

    def forward(self, x: Tensor) -> Tensor:
        out = self.matmul(x)
        if self.bias is not None:
            out += self.bias
        return out


###################### Patching ######################
def patch_linearlayers(model, fct, patch_param=None):
    model.base_class.patch_linearlayers(
        model,
        fct,
        {lin_tag: patch_param for lin_tag in model.base_class.get_linear_tags()},
    )


def patch_hqq_to_bitblas(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, HQQLinear):
        hqq_layer = layer
    if isinstance(layer, HQQLinearLoRA):
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    if HQQLinearBitBlas.check(hqq_layer) is False:
        print("Skipping BitBLas conversion for ", hqq_layer.name)
        return layer

    hqq_bitblas_layer = HQQLinearBitBlas(hqq_layer)

    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return hqq_bitblas_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = hqq_bitblas_layer

    return layer


def replace_with_bitblas(model):
    patch_linearlayers(model, patch_hqq_to_bitblas)
    cleanup()
