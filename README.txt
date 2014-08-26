Version 1.0.1
Eric Hansen, Per-Ola Norrby, Elaine Lime, Olaf Wiest
Contact: ericchansen@gmail.com

Python code to aid in optimizing molecular mechanics parameters.
Supports the MM3* force field, but many more will be added in future
updates.

I admit that much more documentation is needed, but I wanted to get
this online sooner rather than later.

Getting your hands dirty:
=========================
- No installation is necessary. calculate.py, evaluate.py, filetypes.py,
  gradient.py, loop.py, parameters.py, setup_logging.py, and simplex.py
  should all be in your Python path.
- logging.yaml, masses.yaml, parameters.yaml (you will generate this),
  steps.yaml, and weights.yaml should all be put into the subdirectory
  options if they aren't there already. Then again, the location of
  these files can be modified through command line arguments.
- You'll have to set up a substructure in a MM3* force field. Define all
  the parameters you want to optimize.
- Start by playing with parameters.py. Almost all of the Python scripts
  have help dialog. This script makes the aforementioned parameters.yaml,
  which is essentially a dictionary of the parameters you want to
  optimize (you may have defined many more in the substructure, but only
  want to optimize a subset at a time). The most important values are:
  mm3_col, mm3_row, and substr_name because these will be used to aid in
  reading and writing parameters from the FF.
- Next try using calculate.py. This script generates the output data,
  similar to what used to be found in par.ref and par.one. However,
  we barely need to read/write any files during the optimization.
  Instead, this data is simply stored in memory. Nonetheless, try
  printing some data using calculate.py --output.
  This is determined by the sort method in calculate.sort_datum. Check
  to ensure everything is lining up as it should.
- The order in which reference and FF data is calculated no longer
  matters. Instead, there is a sorting algorithm in calculate.sort_datum
  that handles the matching of data.
- If you did the last step, you'll notice that the 1st column of the
  output is a semi-arbitrary label. The 3rd column is the value of the
  data point. The 2nd column is the weight used in the penalty function.
  These weights are defined using a YAML file, which is in
  options/weights.yaml by default. Look at the various YAML files in
  options. They defined various important settings that you can change
  as you see fit.
- Next try using simplex.py or gradient.py to optimize a few parameters.
  These function as standalone scripts, as well as integrate nicely into
  the greater optimization loops.
- Try saving data using gradient.py or simplex.py. Pay attention to the
  help options under "Load/save options." The data is saved as a pickle.
- Here's an example of something to try. First, you may want to go into
  options/logging.yaml and change the root logger's level to DEBUG from
  INFO to get more information. Also, you will have to change the 
  arguments for calculate.py. Here, I am using the file c2.01.mae to
  compare charges.

  $ python gradient.py --ffpath data/mm3.fld -c "--dir data -mq
        c2.01.mae" -r "--dir data -jq c2.01.mae" --init
  
  Then start your python interpreter...

  >>> from parameters import *
  >>> import pickle
  >>> with open('gradient.pickle', 'rb') as f:
  >>>     dic = pickle.load(f)
  >>> print dic
  >>> print dic['Params.']
  >>> print len(dic['Params.'])
  >>> print dic['Ref. Data']
  >>> print dic['FF']
  >>> print dic['FF'].x2
  >>> print dic['FF'].data
  >>> print dic['FF'].params
  >>> print dic['FF'].params[0]
  >>> print dic['FF'].params[0].step_size
  >>> print dic['FF'].params[0].weight
  >>> print dic['FF'].params[0].value
  >>> print dic['Trial FFs']
  >>> print len(dic['Trial FFs'])

  As you can see, there are currently no trial FFs (length is 0). Exit
  the interpreter and try...

  $ python gradient.py --ffpath data/mm3.fld -c "--dir data -mq 
        c2.01.mae" -r "--dir data -jq c2.01.mae" --load --save
        --newton --basic

  Notice that you no longer had to do any calculations for central
  differentiation? The results from those time consuming calculations
  are stored in the pickle. Reload everything in Python and...

  >>> print len(dic['Trial FFs'])
  >>> print dic['Trial FFs'][0]
  >>> print [x.x2 for x in dic['Trial FFs']]
  >>> print sorted([x.x2 for x in dic['Trial FFs']], key=lambda x: x.x2)
  >>> print dic['Trial FFs'][0].gen_method
  >>> print dic['Trial FFs'][0].params

  $ python gradient.py --ffpath data/mm3.fld -c "--dir data -mq 
        c2.01.mae" -r "--dir data -jq c2.01.mae" --load --save 
        --lagrange --levenberg --svd

  >>> print len(dic['Trial FFs'])

  As you can see, almost all of this Python code can be run from the
  command line or natively in Python with a high degree of
  customization.

- loop.py functions quite similarly to gradient.py or simplex.py. Here
  is an example to almost exactly mimic the optimization method in
  Elaine's older versions of the Python code. Again, I am just using
  charges from a single Jaguar output file (c2.01.mae).

  $ python parameters.py data/mm3.fld -q --output
        options/parameters.yaml --substr "Diaminosulfone OPT"
  $ python loop.py --ffpath data/mm3.fld -r "--dir data -jq c2.01.mae"
        -c "--dir data -mq c2.01.mae" --default

Code to be reintegrated (soon):
===============================
This code has already been written by me, but is not implemented into
the current version yet. My old versions had lots of useful bells and
whistles, but the overall design of the program was sloppy. For this
release, I left out some of those features while I tried to figure out
the optimum organization of the code.

- Selection of charges to be used as reference or calculated data
  based upon atom number.
- Parameter tethering.
- Hessian weights that vary depending on atom linkage.
- Bond lengths.
- Angles.
- Inverse distances.
- Option to automatically scale weights based on Boltzmann contribution
  of the reference structure.
- gradient.py. I have a tournament style algorithm working with my last
  version of the code.

Some points I tried to emphasize in this version:
=================================================
- Ease in changing parts around. Very little should need changing to
  adopt the code for Amber, for example. You would have to create a new
  class in filetypes.py, add a new method similar to
  calculate.make_macromodel_coms, and add a few more options to
  calculate.py. Now, this isn't 100% true right now for a few reasons,
  namely not having the exporting and importing of FFs being a class
  method. That's a simple fix that I will get to soon.
- Intermediate results are backed up when something goes wrong. You can
  load them from the pickle like in the example above. At Nore Dame, we
  only have so many Schrodinger licenses. This software, by nature,
  frequently grabs and releases licenses, which led to problems when
  many licenses were already in use. Now, rather than having no idea
  where the optimizer left off, it is simple to restart right where it
  left off simply by reloading the pickle.
- Each piece of the optimization (simplex, gradient/NR, genetic) can be
  used as a standalone or in the greater loop optimization.
- Much less file reading and writing, which has allowed significant
  speed gains.
- Better documentation. Make good use of each script's help!
- Instead of doing three separate MacroModel calculations when the
  Hessian, charges, and optimized geometries are needed (just as one
  example), these are consolidated into a single .com file. This has
  led to the greatest speed increase by eliminating a significant
  amount of overhead wasted on starting and closing MacroModel
  repeatedly.
- There are no requirements on what directory data is stored in and 
  minimal requirements on the filenames. Corresponding reference and
  FF calculated data points are matched based on part of the filename
  (see calculate.sort_datum), the data type, and an index corresponding
  to the individual data point (often an atom label).
- Logging is used to make interpretation of what's going on significantly
  easier. I will be adding command line arguments to change the verbosity
  sometime soon, but for now just adjust it in options/logging.yaml.

Code planned for future releases:
=================================
- Everything that I have already written (vide supra) but not added to
  this version yet.
- New Hessian/eigenvalue method, recently described by Per-Ola.
- More fine tuned control of what optimization procedures will be 
  used during a single loop cycle through some sort of argument 
  or input file method. I had a neat idea with how it could be done
  using a simple YAML dictionary file.
- Argument based way to change verbosity and location of logging.yaml.
- Add support for Tinker.
- Add support for Amber.
- Create an additional penalty function score, which measures not the
  quality of fit to reference data, but instead the uniqueness of the
  given parameter set from all other trial parameter sets. This will
  be quite useful in the initial stages of parameter refinement.

Brief description of the directory:
===================================

calculate.py
------------
Helps select, extract, and organize data from various files before
insertion into the penalty function.

evaluate.py
-----------
Currently doesn't do much. I will probably just move its functions
into calculate.py.

filetypes.py
------------
For every filetype that calculate.py extracts data from, a class
for that filetype is defined in this script. That class is largely
responsible for extracting and processing the raw data in the file.
calculate.py sort of just organizes that data.

- This is where many of the changes will occur in the next couple days.
  I already wrote these data extraction methods for older version, but I
  want to encapsulate them better as I add them here.
- I need to test how Hessian extraction works using my method when there
  are dummy atoms.

gradient.py
-----------
An optimization method where many gradient based methods are located,
such as Newton-Raphson, SVD, Lagrange dampening factors, etc.

- Might move gradient.calc_derivatives elsewhere.

loop.py
-------
Continues looping through the desired optimization methods until the 
penalty function stops changing significantly or the maximum number of
cycles has been reached. Right now, there isn't a lot of room for
modifying the optimization cycle here, but that's coming.

Wouldn't it beat neat to 1st use the genetic algorithm initially to find
many diverse, yet well-performing parameter sets. Then each of these
sets could be fed into the simplex or gradient based optimizations for a
single cycle. The results could then be compared, and if the parameters
were fairly similar, the best force field would continue on and be
refined further. If the parameters sets were still fairly diverse, they
could continue to be optimized separately until some criterion of
similarity or performance is reached.

- Design more robust/complicated/better way to define what to do during
  a single optimization loop.
- Did I actually get around to adding the conditional for a maximum
  number of loops?

parameters.py
-------------
Defines force fields and parameters. Provides methods to extract them,
work with them, manipulate them, etc. What filetypes.py is to data
files, is what parameters.py is to force field files.

- The import and export FF methods should be built into their respective
  classes. Then simply self.export can be called regardless of what type
  of FF is actually in use.

setup_logging.py
----------------
Contains only a few basics functions to setup logging. Can be used to
create the default logging.yaml file.

simplex.py
----------
Optimizes parameters using the simplex method.

loop.py gradient.py simplex.py ...
----------------------------------
- Add argument based way to change location of logging.yaml.
- Add argument based way to change verbosity.

