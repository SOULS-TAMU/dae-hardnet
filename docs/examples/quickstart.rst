Quick start: a differential equation
====================================

Learn ``dy/dt + y = 0`` from state and derivative observations. Run the complete
downloadable script after :doc:`../INSTALL`. Two epochs are an execution check,
not a converged fit.

Setup
-----

.. literalinclude:: quickstart.py
   :language: python
   :end-before: # 1. Data

1. Prepare observations
-----------------------

The arrays contain time, observed state, observed derivative and physical
parameters. This example has no physical parameters, so their array has zero
columns rather than zero samples.

.. literalinclude:: quickstart.py
   :language: python
   :start-at: # 1. Data
   :end-before: # 2. Problem

2. Define the problem
---------------------

Declare ``y(t)``, its first derivative, and a residual that must equal zero.

.. literalinclude:: quickstart.py
   :language: python
   :start-at: # 2. Problem
   :end-before: # 3. Projection

3. Prepare the projection
-------------------------

The Taylor neighborhood processes differential terms before projection
preparation. The batch size is fixed for the prepared resources.

.. literalinclude:: quickstart.py
   :language: python
   :start-at: # 3. Projection
   :end-before: # 4. Network

4. Define the network and train
-------------------------------

The first four observations are used for training; the remaining four are used
for validation. Named arguments make each data group's role explicit.

.. literalinclude:: quickstart.py
   :language: python
   :start-at: # 4. Network
   :end-before: # 5. Predict

5. Predict
----------

The public prediction method returns physical states and derivative/diagnostic
groups. Select its projected-state output:

.. literalinclude:: quickstart.py
   :language: python
   :start-at: # 5. Predict
   :end-before: # Check the declared physical/output contract

Verify and export
-----------------

These checks test the residual and export a physical-state graph. The repeated
input below respects the prepared batch size of two; deployment exposes one
instance. See :doc:`../INFERENCE` for the interpretation of the output graph.

.. literalinclude:: quickstart.py
   :language: python
   :start-after: # Check the declared physical/output contract and model transport.

Download the full script: :download:`quickstart.py <quickstart.py>`.
