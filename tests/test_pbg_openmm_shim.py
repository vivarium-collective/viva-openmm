"""The pbg_openmm import name stays working (deprecated) after the rename."""
import warnings


def test_pbg_openmm_still_imports_and_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pbg_openmm  # noqa: F401
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_pbg_openmm_submodule_redirects_to_viva_openmm():
    import viva_openmm.topology as real
    import pbg_openmm.topology as shimmed
    assert shimmed is real            # meta-path finder aliases to the real module
