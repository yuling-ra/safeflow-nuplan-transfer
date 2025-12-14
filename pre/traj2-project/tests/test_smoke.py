def test_import():
    import traj2
    assert hasattr(traj2, "__all__")
