
.. readme-intro

.. image:: https://github.com/catalyst-cooperative/ccai-entity-matching/workflows/tox-pytest/badge.svg
   :target: https://github.com/catalyst-cooperative/ccai-entity-matching/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://img.shields.io/codecov/c/github/catalyst-cooperative/ccai-entity-matching?style=flat&logo=codecov
   :target: https://codecov.io/gh/catalyst-cooperative/ccai-entity-matching
   :alt: Codecov Test Coverage

.. image:: https://img.shields.io/pypi/pyversions/catalystcoop.cheshire?style=flat&logo=python
   :target: https://pypi.org/project/catalystcoop.cheshire/
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

This repository performs entity resolution to match the FERC Form 1 and EIA data. Funded
by a grant from CCAI.

Installation
============
To install the software in this repository, clone it to your computer using git.
If you're authenticating using SSH:

.. code:: bash

   git clone git@github.com:catalyst-cooperative/ccai-entity-matching.git

Or if you're authenticating via HTTPS:

.. code:: bash

   git clone https://github.com/catalyst-cooperative/ccai-entity-matching.git

Then in the top level directory of the repository, create a ``conda`` environment
based on the ``environment.yml`` file that is stored in the repo:

.. code:: bash

   conda env create --name ferc1_eia_match --file environment.yml

To run the pre-commit hooks before you commit code run:

.. code:: bash

   pre-commit install

Note that the software in this repository depends on
`the dev branch <https://github.com/catalyst-cooperative/pudl/tree/dev>`__ of the
`main PUDL repository <https://github.com/catalyst-cooperative/pudl>`__,
and the ``setup.py`` in this repository indicates that it should be installed
directly from GitHub. This can be a bit slow, as ``pip`` (which in this case is
running inside of a ``conda`` environment) clones the entire history of the
repository containing the package being installed. How long it takes will depend on
the speed of your network connection. It might take ~5 minutes.

Thank You
=========

Thank you to `Climate Change AI <https://www.climatechange.ai/>`__. for awarding us a grant to conduct this work.
