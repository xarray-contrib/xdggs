.. _api:

#############
API reference
#############

Top-level functions
===================

.. currentmodule:: xdggs

.. autosummary::
   :toctree: generated

   decode

Grid parameter objects
======================

.. autosummary::
   :toctree: generated

   DGGSInfo

   HealpixInfo
   H3Info

.. currentmodule:: xarray

Dataset
=======

Parameters
----------
.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_attribute.rst

   Dataset.dggs.grid_info
   Dataset.dggs.params

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   Dataset.dggs.decode
   Dataset.dggs.encode

Indexing
--------

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   Dataset.dggs.sel_latlon


Data inference
--------------

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   Dataset.dggs.cell_centers
   Dataset.dggs.cell_boundaries
   Dataset.dggs.assign_latlon_coords

DataArray
=========

Parameters
----------
.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_attribute.rst

   DataArray.dggs.grid_info
   DataArray.dggs.params

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   DataArray.dggs.decode
   DataArray.dggs.encode

Indexing
--------

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   DataArray.dggs.sel_latlon



Data inference
--------------

.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   DataArray.dggs.cell_centers
   DataArray.dggs.cell_boundaries
   DataArray.dggs.assign_latlon_coords

Plotting
--------
.. autosummary::
   :toctree: generated
   :template: autosummary/accessor_method.rst

   DataArray.dggs.explore

Tutorial
========

.. currentmodule:: xdggs

.. autosummary::
   :toctree: generated

   tutorial.open_dataset

Advanced API
============

Index
-----
.. autosummary::
   :toctree: generated

   register_dggs

Conventions
-----------
.. autosummary::
   :toctree: generated

   conventions.Convention
   conventions.register_convention
   conventions.DecoderError
