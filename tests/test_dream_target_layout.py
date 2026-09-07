from __future__ import annotations

import torch

from fastwam.datasets.lerobot.robot_video_dataset import DreamTargetAdapter


def _adapter() -> DreamTargetAdapter:
    return DreamTargetAdapter(
        dataset_dirs=[],
        cfg={
            "enabled": False,
            "feature_dims": {"sam": 4, "dino": 3},
            "token_grids": {"sam": [2, 3], "dino": [2, 2], "dyn": [4, 4]},
            "dyn": {
                "patch_size": 2,
                "motion_threshold_px": 1.0,
                "score_threshold": 0.5,
            },
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


def test_depth_target_is_a_two_view_dense_map() -> None:
    primary = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    wrist = primary + primary.numel()
    actual = _adapter()._concat_camera_targets(primary, wrist, "depth")

    assert actual.shape == (128, 256)
    torch.testing.assert_close(actual[:, :128], primary)
    torch.testing.assert_close(actual[:, 128:], wrist)


def test_dynamic_motion_vectors_become_strict_binary_patch_labels() -> None:
    adapter = _adapter()
    motion = torch.zeros(16, 2)
    motion[:2] = torch.tensor([2.0, 0.0])
    motion[4:6] = torch.tensor([2.0, 0.0])
    motion[2:4] = torch.tensor([0.5, 0.0])
    motion[6:8] = torch.tensor([0.5, 0.0])

    mask = adapter._dynamic_to_patch_mask(motion, patch_size=2)

    assert mask.shape == (4, 1)
    assert set(mask.flatten().tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(mask.flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_dense_dynamic_scores_are_binarized_after_patch_pooling() -> None:
    adapter = _adapter()
    scores = torch.zeros(4, 4)
    scores[:2, :2] = 0.75
    scores[:2, 2:] = 0.25

    mask = adapter._dynamic_to_patch_mask(scores, patch_size=2)

    torch.testing.assert_close(mask.flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
