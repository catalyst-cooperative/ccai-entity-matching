"""Perform entity resolution on the FERC1 and EIA datasets."""

import logging

import pkg_resources

# In order for the package modules to be available when you import the package,
# they need to be imported here somehow. Not sure if this is best practice though.
from ferc1_eia_match import blocking, helpers, inputs  # noqa: F401

__author__ = "Catalyst Cooperative"
__contact__ = "pudl@catalyst.coop"
__maintainer__ = "Catalyst Cooperative"
__license__ = "MIT License"
__maintainer_email__ = "pudl@catalyst.coop"
__version__ = pkg_resources.get_distribution("catalystcoop.ferc1_eia_match").version
__docformat__ = "restructuredtext en"
__description__ = "Perform entity resolution on the FERC1 and EIA datasets."
__long_description__ = """
This repository performs entity resolution to match the FERC Form 1 and EIA data. Funded
by a grant from Climate Change AI.
"""
__projecturl__ = "https://github.com/catalyst-cooperative/ferc1_eia_match"
__downloadurl__ = "https://github.com/catalyst-cooperative/ferc1_eia_match"

# Create a root logger for use anywhere within the package.
logging.basicConfig(level="INFO")
logger = logging.getLogger("ferc1_eia_match")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"

logger.addHandler(logging.NullHandler())
