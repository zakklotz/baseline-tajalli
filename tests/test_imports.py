def test_imports_smoke():
    import tajalli  # noqa: F401
    import tajalli.nncore_bridge  # noqa: F401

    # This should succeed after:
    #   pip install -e ./.nn-core
    import nncore  # noqa: F401
