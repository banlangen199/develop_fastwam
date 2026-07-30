from __future__ import annotations

import torch

from fastwam.datasets.lerobot.robot_video_dataset import DreamTargetAdapter


def _adapter() -> DreamTargetAdapter:
    return DreamTargetAdapter(
        dataset_dirs=[],
        cfg={
            "enabled": False,
            "feature_dims": {"sam": 4, "dino": 3},
            "token_grids": {"sam": [2, 3], "dino": [2, 2]},
        },
    )


def test_flat_sam_chw_is_converted_to_spatial_tokens() -> None:
    adapter = _adapter()
    chw = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)

    actual = adapter._as_token_matrix(chw.flatten(), "sam")
    expected = chw.permute(1, 2, 0).reshape(6, 4)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()


def test_sam_chw_tensor_is_converted_to_spatial_tokens() -> None:
    adapter = _adapter()
    chw = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)

    actual = adapter._as_token_matrix(chw, "sam")
    expected = chw.permute(1, 2, 0).reshape(6, 4)

    torch.testing.assert_close(actual, expected)


def test_existing_token_matrix_layout_is_unchanged() -> None:
    adapter = _adapter()
    tokens = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)

    actual = adapter._as_token_matrix(tokens, "dino")

    torch.testing.assert_close(actual, tokens)


def test_depth_patch_size_eight_preserves_two_view_depth_pixels() -> None:
    primary = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    wrist = primary + primary.numel()
    combined = torch.cat([primary, wrist], dim=1)

    patches = DreamTargetAdapter._patchify_image_target(
        combined,
        patch_size=8,
        modality="depth",
    )

    assert patches.shape == (512, 64)
    restored = torch.nn.functional.fold(
        patches.transpose(0, 1).unsqueeze(0),
        output_size=(128, 256),
        kernel_size=8,
        stride=8,
    ).squeeze(0).squeeze(0)
    torch.testing.assert_close(restored, combined)
