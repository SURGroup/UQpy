import hms
import os

class HMS:
    """
    Run a hierarchical multiscale (HMS) model.
    This class leverages the hierarchical multiscale framework established by the Army Research Laboratory for
    evaluation of large multi-scale problems. The framework is described in references [3]_.
    **Input:**
    * **exec_prefix** ('str')
        String containing any commands that precede a call to the executable that runs the simulation
        Examples of this may include, e.g. 'mpirun -np 1 -machinefile machinefile', and other similar commands
    * **exec_path** ('str')
        String containing the path to the executable that runs the model
        If the executable exists in the user's PATH, this can just be a direct call to this executable (e.g. 'python'.
        If the executable does not exist in the user's PATH, this must be the full path to the executable file.
    * **resourceType** ('str')
        Type of computational resources being used by the model.
        Options: 'CPU'
    * **resourceAmount** ('int')
        Number of CPUs or other resources allocated to each individual model evaluation.
    * **hms_pointFileName** ('str')
        This is the template input file. It follows all the same formatting as the ``RunModel'' input template file
        specified by 'input_template'. See the ``RunModel'' class.
    * **hms_configFile** ('str')
        Provides the path to the HMS configuration file. For details of the HMS configuration file, see HMS
        documentation.
    * **mpi_rank** ('int')
        MPI Rank of the upper-scale model
    * **ncpus_upper** ('int')
        Number of CPUs used for running the upper-scale model.
    """

    def __init__(self, exec_prefix, exec_path, resourceType, resourceAmount, hms_pointFileName, hms_configFile,
                 mpi_rank=0, ncpus_upper=1):

        self.hms_pointFileName = hms_pointFileName
        self.hms_configFile = hms_configFile
        self.exec_prefix = exec_prefix
        self.exec_path = exec_path
        self.resourceAmount = resourceAmount
        self.mpi_rank = mpi_rank
        self.ncpus_upper = ncpus_upper
        self.hms_inputFilter = None
        self.hms_outputFilter = None
        self.hms_argument = None
        self.hms_modelPackage = None
        self.hms_returnPackage = None

        self.communicator = hms.BrokerLauncher().launch(self.hms_configFile, self.mpi_rank, self.ncpus_upper)

        if len(self.communicator) > 1:
            raise TypeError('\nUQpy: Only one HMS broker is supported.\n')
        else:
            self.communicator = self.communicator[0]

        if resourceType == 'cpu' or 'CPU':
            self.resourceType = hms.CPU
        else:
            raise TypeError('\nUQpy: HMS currently only supports CPU computing.\n')

        self.model = hms.Model(exec_prefix, exec_path, hms.StringVector([self.hms_pointFileName]), self.resourceType,
                               self.resourceAmount)

        self.qoi_list = []

    def run(self, hms_inputFilter, hms_outputFilter, hms_argument):
        """
        Run a single HMS model evaluation
        **Input:**
        * **hms_inputFilter** ('object' of ``InputFilter'' class)
            HMS requires a Python InputFilter class that is customized to the specific model. The InputFilter is used
            to process model arguments and write individual input files for each indexed model evaluation.
            A template InputFilter is provided with UQpy that writes each input file using the ``UQpy.RunModel''
            conventions. This template InputFilter requires only a small number of fields to be edited to customize
            the InputFilter for a specific application.
        * **hms_outputFilter** ('object' of ``OutputFilter'' class)
            HMS requires a Python OutputFilter class that is customized to the specific model. The OutputFilter is used
            to process model output files and return the values of model quantities of interest.
            A template OutputFilter is provided that reads output standard output (stdout) from the model evaluation.
            If the model creates more sophisticated output, e.g. text or binary output files, a custom OutputFilter
            must be written to read these files and extract the relavent quantities of interest.
        * **hms_argument** ('object' of ``Argument'' class)
            HMS requires a Python Arugment class that is customized to the specific model. The Argument class is used
            to internalize the values of variables in an HMS model evaluation, so that they can subsequently be passed
            into the InputFilter and an input file can be generated for that model evaluation.
            The 'hms_arguments' class must accept inputs corresponding to all input variables in the model as well as
            an index to track the model evaluations.
        """

        print('From inside RunModel')
        print(os.getcwd())

        self.hms_inputFilter = hms_inputFilter
        self.hms_outputFilter = hms_outputFilter
        self.hms_argument = hms_argument
        self.hms_modelPackage = hms.ModelPackage(self.model, self.hms_inputFilter, self.hms_outputFilter,
                                                 self.hms_argument)

        self.qoi_list.extend([None])

        self.communicator.send(self.hms_modelPackage)

    def receive(self):
        """
        Pull results of completed HMS model evaluations and return them to UQpy to populate the qoi_list.
        **Attributes**
        * **qoi_list** (`list`)
            A list containing the output quantities of interest
            In the HMS workflow, these output quantities of interest are extracted from the model output files by the
            OutputFilter.
        """
        self.hms_returnPackage = self.communicator.receive()

        for modelPackage in self.hms_returnPackage:
            index = modelPackage.getArgument().index
            self.qoi_list[index] = modelPackage.getValue().value