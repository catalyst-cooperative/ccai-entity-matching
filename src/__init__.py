"""Perform entity resolution on the FERC1 and EIA datasets."""

import logging

import pkg_resources

# In order for the package modules to be available when you import the package,
# they need to be imported here somehow. Not sure if this is best practice though.
from ferc_eia_match import cli, dummy, extract, helpers, transform  # noqa: F401

__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "pudl@catalyst.coop"
__version__ = pkg_resources.get_distribution("catalystcoop.ferc_eia_match").version
__docformat__ = "restructuredtext en"
__description__ = "Perform entity resolution on the FERC1 and EIA datasets."
__long_description__ = """
This repository performs entity resolution to match the FERC Form 1 and EIA data. Funded
by a grant from Climate Change AI.
"""
__projecturl__ = "https://github.com/catalyst-cooperative/ferc_eia_match"
__downloadurl__ = "https://github.com/catalyst-cooperative/ferc_eia_match"

# Create a root logger for use anywhere within the package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
