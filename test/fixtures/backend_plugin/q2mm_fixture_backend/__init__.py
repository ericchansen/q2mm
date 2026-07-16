"""Out-of-tree Q2MM backend plugin fixture (test-only).

This package is deliberately laid out to prove Q2MM's lazy, descriptor-first
plugin discovery:

* :mod:`q2mm_fixture_backend.descriptor` is a *lightweight* module that exposes
  only a JSON-safe manifest mapping (and an equivalent provider callable).  It
  imports nothing from :mod:`q2mm_fixture_backend.backend`.
* :mod:`q2mm_fixture_backend.backend` holds the actual backend implementation
  and is imported only when the manifest's ``factory`` import string is
  resolved by an explicit ``load_backend``/``BackendDescriptor.load``.

The distribution's ``q2mm.backends`` entry point targets the descriptor module,
so enumeration and cataloging never import the implementation.

This package is repository test code only.  It lives outside ``q2mm/`` and must
never ship inside the ``q2mm`` wheel or sdist.
"""
