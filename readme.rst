Samples
=======

A collection of lab-examples


Quick Start
-----------

1) Clone the Samples Repo
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

     git clone https://github.com/lab-ml/samples.git


2) Install the Requirements
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

     pip install -r requirements.txt

3) Run an Example
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

     make mnist_hyperparam_tuning


4) View on Lab Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~

Start the server using the following command. It will open the Lab Dashboard in your default browser.

**NodeJS is Required**

*NodeJS installation is required for Lab Dashboard. If NodeJS is not installed, you can downland and install* `here <https://nodejs.org/en/download/>`_.

.. code-block:: bash

     lab dashboard


Run Examples
------------


1) To run all the examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make all

2) To run all the Pytorch examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make pytorch


2.1) To run the GAN example
"""""""""""""""""""""""""""

.. code-block:: bash

    make gan

2.2) To run the RNN example
"""""""""""""""""""""""""""

.. code-block:: bash

    make rnn

2.3) To run the CIFR10 example
""""""""""""""""""""""""""""""

.. code-block:: bash

    make cifr10


2.4) To run all MNIST examples
"""""""""""""""""""""""""""""""""

.. code-block:: bash

    make mnist

.. note::

   *To run each MNIST example*

   .. code-block:: bash

    make mnist_configs
    make mnist_hyperparam_tuning
    make mnist_indexed_logs
    make mnist_latest
    make mnist_v1



3) To run all the SkLearn examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make sklearn



