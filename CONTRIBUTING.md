# Contributing guidelines

### Coding styles ([PEP8](https://www.python.org/dev/peps/pep-0008/ "Style Guide for Python"))
___
We strive to make our code as community friendly as possible. In order to achieve that, our aim is to abibe by the code style guide PEP8 for python. This guide provides many conventions for Python languge and in the scope of UQpy, specific attention is given to the naming conventions provided. The contributor is adviced to give specific attention to these conventions that make the code, developer friendly as well as self-explanatory in most cases, thus reducing the need for code explanatory comments. To this end, single letter names are detered, as they are considered a left-over of old programming languages. Specific attention is given to to the following naming [conventions](https://pep8.org/#prescriptive-naming-conventions "Python naming conventions")

* Module names: Module names should comprise of only lowercase letters. If a module name is comprised of multiple words, underscores can be used to improve readability.

* Class names: For naming classes the **PascalCase** convention is followed, where the first letter of each word that comprises the name of the class is capitalized. In addition, abbreviations are detered to improve readability. For instance, the class name LatinHypercubeSampling is preferred to LHS.

* Variable names: In case of local variable names the **snake_case** convention is followed. Once again abbreviations or single letter names are detered. In case of constants, capitalized words separated by underscores are favoured (e.g. MAX_ITERATIONS).

* Functions: Similar to local variables, functions should abide to the **snake_case** naming convention. Prefer full names here as well. (e.g. evaluate_polynomial is preferred than ev_p)

### Code formatting
___
For the rest of the coding/styling conventions defined in PEP8, the contributor is encouraged to use the [Black](https://pypi.org/project/black/#:~:text=Black%20is%20the%20uncompromising%20Python,energy%20for%20more%20important%20matters. "coding formatter"), that takes care of most other PEP8 conventions automatically.  This is not yet a prerequite for new pull requests, but will be integrated to the next version of Continuous Integration tasks. To use it, utilize the following commands:

> pip install black
>
> black {uqpy_src_path}


Alternatively, in case of anaconda distributions

> conda install -c anaconda black
>
> black {uqpy_src_path}

### Docstrings
___
Before uploading a code make sure all new classes and methods added are accompanied by the required docstrings. The [PEP 257](https://www.python.org/dev/peps/pep-0257/ "Docstring conventions") guideline contains the specifications for creating and maintaining the respective code documentation.

### Documentation
___
The documentation of the project is two-fold. The first part comprises of a [ReadTheDocs](https://uqpyproject.readthedocs.io/en/latest/ "UQpy documentation") part that contains the docstrings, as well the theory that backs up the respective classes and is contained in the **docs/** folder of the repository. The second part regards Jupyter notebook examples that showcase the self-contained applications of the code and exists in the **example/** folder of the repo. In case of new feature contributions, the extention of the existing documentation will be required to include the respective theory and docstrings and at least one jupyter notebook example of the new feature. Pull requests that do not contain the required documentation will not be merged until all required examples are provided.


## Continous Integration
To ensure the quality of the code and its immediate distribution to all channels, a Continuous Integration pipeline is implemented in [Azure Pipelines](https://dev.azure.com/UQpy/UQpy/_build). The minimum requirements for a pull request to be acceptable are succesfull Pylint execution, successfull unit test execution and greater or equal to 80% code coverage on new code with **A** maintainability rating.  All of the above will be explained in detail below.

### Pylint
___
Pylint (along with **flake8**) is one of the most well-known linting packages for python. In the case of UQpy, pylint is utilized to detect the errors that the IDE was not able to recognize. For the time being all warnings, conventions, refactors and infos are disabled. The contributors are encouraged to run Pylint localy before creating a pull request. A  detailed configuration of Pylint with the Pycharm IDE can be found [here](https://www.jetbrains.com/help/pycharm/configuring-third-party-tools.html).

### Running and writing tests
___
Another vital part of the continuous integration process is the successfull test execution. Using the [Pytest](https://docs.pytest.org/en/stable/) framework, a set of unit test are developed that reside in the **tests/** folder of the repository. For all pull requests as mentioned above, there are two are main requirements in order to be accepted. This first requirement is the successfull test execution of all unit test of the repo. The second requirement is a minimum of 80% coverage on new code. The contributors are encouraged to check for successfull test execution localy before uploading their code and issuing a pull request. Detailed information on how to set up the [pytest framework](https://www.jetbrains.com/help/pycharm/pytest.html#pytest-bdd) and [code coverage](https://www.jetbrains.com/help/pycharm/code-coverage.html#run-with-coverage) in Pycharm IDE are provided in the respective links. Note that for Code Coverage in Pycharm the Professional Edition is required, which is also available for students and academic faculty.

### Sonarcloud & Quality gates
___
Sonarcloud is a static code analysis tool that ensures code quality and security. Integrated in the CI procedure, Sonarcloud analyzes the code uploaded in each pull request and detects tricky bugs and vulnerabilities that might lead to undefined behaviour of the code  and thus, impacting end-user. The current metrics of the UQpy code can be found [here](https://sonarcloud.io/dashboard?id=SURGroup_UQpy). The minimum requirements for a Sonarcloud analysis to be successfull is the default Quality Gate provided by the tool, which translates to minimum 80% coverage on new code, less than 3% duplicated code and **A** maintainability, reliability and security ratings. All the above can also be found in the project [Quality Gate](https://sonarcloud.io/organizations/jhusurg/quality_gates/show/9). 


### Branching stategies

For all contributors, **master** and **Development** branches are protected. As a result no-one can commit directly to these branches. For contributors that belong to the organization, the recommented procedure is to start the branches from **Development** branch. In order for the CI procedure to run automatically, two different naming strategies are followed. Either **feature/{name_of_feature}** in case this branch is used for the development of a non-existing feature, or **bugfix/{name_of_bugfix}** in case the respective branch is used to fix a bug on the existing code. Note that ALL pull requests must point to the Development branch. Any pull requests to the master branch will be closed and should be resubmitted to the Development branch. For external contributors, the suggested procedure is to create a fork of the existing repository. After the contribution is completed, a pull request should be issued to the Development branch of the main repository. In all cases, make sure that all the above CI requirements are satisfied for your pull request to acceptable.

