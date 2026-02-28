def test_model_init_does_not_export_legacy_attention():
    import tajalli.model as m

    assert not hasattr(m, "MultiHeadAttention")
    assert not hasattr(m, "apply_rope")
    assert not hasattr(m, "FeedForward")
