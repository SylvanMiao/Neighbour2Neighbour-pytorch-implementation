import torch
import torch.nn.functional as F

# 全局种子计数器，确保每次调用 get_generator 使用不同种子
operation_seed_counter = 0


def get_generator():
    """返回 CUDA 随机数生成器，每次调用种子自增。"""
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def generate_mask_pair(img):
    """生成一对互补的 2x2 邻域子采样掩码。

    将图像的每个 2x2 块按 8 种预设配对方式随机选择一种，
    两个掩码在每个块中各自选中 2 个不同像素，互为补集。
    """
    n, c, h, w = img.shape
    n_blocks = n * h // 2 * w // 2
    total_pixels = n_blocks * 4

    mask1 = torch.zeros(size=(total_pixels,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(total_pixels,), dtype=torch.bool, device=img.device)

    # 2x2 块内 4 个位置的 8 种邻域配对：
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)

    rd_idx = torch.zeros(size=(n_blocks,), dtype=torch.int64, device=img.device)
    torch.randint(low=0, high=8, size=(n_blocks,),
                  generator=get_generator(), out=rd_idx)

    # 查表得到每个块选中的配对，加上块偏移得到全局像素索引
    rd_pair_idx = idx_pair[rd_idx] #size (n_blocks,2) 这里是fancy index
    rd_pair_idx += torch.arange(start=0, end=total_pixels, step=4,
                                dtype=torch.int64, device=img.device).reshape(-1, 1)

    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    """根据掩码从原图中提取子图像。

    通过 space_to_depth 将 2x2 块展开为通道维，再按 mask 选取像素，
    重组为 (H/2, W/2) 的子图。
    """
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h // 2, w // 2,
                           dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def space_to_depth(x, block_size):
    """将空间块重排到通道维度（等价于 pixel_unshuffle）

    用 unfold 实现：每个 block_size×block_size 的空间块展开后堆叠到通道维。
    """
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)