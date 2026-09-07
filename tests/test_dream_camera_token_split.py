import torch

from fastwam.models.wan22.dream_fastwam import DenseDreamDecoder, DreamQueryExpert


def _decoder(target_shape, target_layout, patch_size=None):
    decoder = DenseDreamDecoder(
        modality="test",
        latent_dim=8,
        target_shape=target_shape,
        target_layout=target_layout,
        patch_size=patch_size,
        decoder_dim=8,
        decoder_ffn_dim=16,
        num_layers=1,
        num_heads=2,
    )
    return decoder.eval()


def test_two_view_grid_decoder_isolates_camera_latents():
    torch.manual_seed(0)
    decoder = _decoder((2, 4, 3), "grid_feature")
    primary = torch.randn(1, 2, 8)
    wrist = torch.randn(1, 2, 8)

    base = decoder.forward_two_view(primary, wrist)
    changed_wrist = decoder.forward_two_view(primary, wrist + 10.0)
    changed_primary = decoder.forward_two_view(primary + 10.0, wrist)

    torch.testing.assert_close(base[:, :, :2], changed_wrist[:, :, :2])
    torch.testing.assert_close(base[:, :, 2:], changed_primary[:, :, 2:])
    assert not torch.allclose(base[:, :, 2:], changed_wrist[:, :, 2:])
    assert not torch.allclose(base[:, :, :2], changed_primary[:, :, :2])


def test_two_view_token_decoder_restores_horizontal_patch_order():
    torch.manual_seed(1)
    decoder = _decoder((8, 3), "token_feature")
    primary = torch.randn(1, 2, 8)
    wrist = torch.randn(1, 2, 8)

    base = decoder.forward_two_view(primary, wrist).reshape(1, 2, 4, 3)
    changed_wrist = decoder.forward_two_view(primary, wrist + 10.0).reshape(1, 2, 4, 3)

    torch.testing.assert_close(base[:, :, :2], changed_wrist[:, :, :2])
    assert not torch.allclose(base[:, :, 2:], changed_wrist[:, :, 2:])


def test_two_view_image_decoder_restores_dense_map_and_isolates_cameras():
    torch.manual_seed(2)
    decoder = _decoder((4, 8), "image", patch_size=2)
    primary = torch.randn(1, 2, 8)
    wrist = torch.randn(1, 2, 8)

    base = decoder.forward_two_view(primary, wrist)
    changed_wrist = decoder.forward_two_view(primary, wrist + 10.0)

    assert base.shape == (1, 4, 8)
    torch.testing.assert_close(base[:, :, :4], changed_wrist[:, :, :4])
    assert not torch.allclose(base[:, :, 4:], changed_wrist[:, :, 4:])


def test_depth_image_layout_keeps_legacy_decoder_parameter_shapes() -> None:
    legacy = _decoder((512, 64), "token_feature")
    image = _decoder((128, 256), "image", patch_size=8)

    legacy_state = legacy.state_dict()
    image_state = image.state_dict()
    assert legacy_state.keys() == image_state.keys()
    assert {
        key: tuple(value.shape) for key, value in legacy_state.items()
    } == {
        key: tuple(value.shape) for key, value in image_state.items()
    }
    image.load_state_dict(legacy_state, strict=True)


def test_configured_modality_shapes_split_into_equal_camera_regions():
    specs = [
        ((128, 256), "image", 256),
        ((392, 1), "token_feature", 196),
        ((16, 32, 768), "grid_feature", 256),
        ((16, 32, 256), "grid_feature", 256),
    ]
    for target_shape, target_layout, expected_per_view in specs:
        decoder = _decoder(
            target_shape,
            target_layout,
            patch_size=8 if target_layout == "image" else None,
        )
        primary_indices, wrist_indices = decoder._two_view_query_indices(device=torch.device("cpu"))
        assert primary_indices.numel() == expected_per_view
        assert wrist_indices.numel() == expected_per_view
        all_indices = torch.cat([primary_indices, wrist_indices]).sort().values
        torch.testing.assert_close(
            all_indices,
            torch.arange(decoder.num_output_tokens),
        )


def _expert(camera_token_split):
    return DreamQueryExpert(
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        dream_query={
            "modalities": ["depth"],
            "future_offsets": [16, 32],
            "n_depth": 18,
            "camera_token_split": camera_token_split,
        },
        dream_expert={
            "hidden_dim": 8,
            "ffn_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "attn_head_dim": 8,
        },
        dream_decoder={
            "decoder_dim": 8,
            "decoder_ffn_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "attn_head_dim": 8,
            "depth": {
                "enabled": True,
                "target_layout": "token_feature",
                "target_shape": [8, 3],
            },
        },
    )


def test_camera_split_keeps_checkpoint_parameter_keys_and_shapes():
    legacy = _expert(None)
    split = _expert([9, 9])
    legacy_state = legacy.state_dict()
    split_state = split.state_dict()

    assert legacy_state.keys() == split_state.keys()
    assert {
        key: tuple(value.shape) for key, value in legacy_state.items()
    } == {
        key: tuple(value.shape) for key, value in split_state.items()
    }
    split.load_state_dict(legacy_state, strict=True)


def test_camera_split_validates_all_enabled_modality_token_counts():
    try:
        _expert([8, 9])
    except ValueError as error:
        assert "must sum" in str(error)
    else:
        raise AssertionError("Expected an invalid camera token split to fail.")
