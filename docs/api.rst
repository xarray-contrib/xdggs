.. _api:

#############
API reference
#############

Top-level functions
===================

.. currentmodule:: xdggs

.. autosummary::
   :toctree: generated/

   decode

Grid parameter objects
======================

.. autosummary::
   :toctree: generated

   HealpixInfo
   H3Info

.. currentmodule:: xarray

Parameters
==========
.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.dggs.grid_info
   Dataset.dggs.grid_info


Data inference
==============

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.dggs.cell_centers
   DataArray.dggs.cell_boundaries
   Dataset.dggs.cell_centers
   Dataset.dggs.cell_boundaries
