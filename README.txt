Version 1.0
Eric Hansen, Per-Ola Norrby, Elaine Lime, Olaf Wiest
Contact: ericchansen@gmail.com

Python code that optimizes molecular mechanics parameters. Currently
only supports modifying MM3* force fields. Uses reference data from
Jaguar and/or Gaussian calculations.

Much more documentation is needed, but I wanted to get this online 
quickly.

Currently the only data types this code works with is MacroModel and
Jaguar charges. The rest will be added shortly. Nonetheless, this
version lays the foundations for how everything will function, so you
can learn to use it.

Getting your hands dirty:
=========================
- No installation is necessary. calculate.py, evaluate.py, filetypes.py,
  gradient.py, loop.py, parameters.py, setup_logging.py, and simplex.py
  should all be in your Python path. logging.yaml, parameters.yaml (you
  generate this), steps.yaml, and weights.yaml should all be in the
  subdirectory options, but you can change the default locations of these
  files through command line arguments. I am still debating on whether
  its better to have them search in the directory options or the present
  working directory by default.

- Setup a substructure in an MM3* force field that has all of the
  necessary parameters defined. You will be fitting the parameter
  values, so those aren't so important right now. Just pick reasonable
  values.

- Read: python parameters.py -h
  Start by figuring out this script. Use this script to make a parameter
  file in YAML, which contains only the parameters you want to optimize.
  The values that matter (most) are: mm3_col, mm3_row, and substr_name.
  This defines where the parameter is located in the force field file
  (probably named mm3.fld).

- Read: python calculate.py -h
  Use this script to output data. Check if the data you are trying to
  output matches that in your output files from MacroModel, Jaguar,
  Gaussian, etc. It's important that the reference data points are
  matched to their correct corresponding FF calculated data point.
  This is determined by the sort method in calculate.sort_datum. Check
  to ensure everything is lining up as it should.

- If you did the last step, you'll notice that the 1st column of the
  output is a semi-arbitrary label. The 3rd column is the value of the
  data point. The 2nd column is the weight used in the penalty function.
  These weights are defined using a YAML file, which is in
  options/weights.yaml by default. Look at the various YAML files in
  options. They defined various important settings that you can change
  as you see fit.

- Read: python simplex.py -h
  Use this script to optimize a few force field parameters. Unlike 
  older versions of the code, any optimization method I add to this
  version will be useable as a standalone or in combination with other
  methods in loop.py.

- Read: python gradient.py -h
  Use this script to optimize a few force field parameters.

- Try saving data using gradient.py or simplex.py. Pay attention to the
  help options under "Load/save options." The data is saved as a pickle.
  You may want to Google what that means to save data in pickle format
  or YAML format.

- After you've Googled it, try doing something like this. 1st off, go to
  options/logging.yaml and change the level of the root logger from INFO
  to DEBUG. Then (obviously you'll have to adjust your calculate.py 
  arguments to match whatever data you have. Here I am just selecting
  charges from Jaguar and from MacroModel):

  $ python gradient.py --ffpath data/mm3.fld -c "--dir data -mq
        c2.01.mae" -r "--dir data -jq c2.01.mae" --init
  
  Try it again with it the logging level back at INFO.  Start your
  Python interpreter and then...

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
  the interpreter and try:

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

  As you can hopefully see, this is a very powerful tool. Indeed,
  basically all of this Python code can be run from the command line
  or natively in Python with an extraordinarily high degree of 
  customization... if you know your way around the code. :)

- Read: loop.py -h 
  loop.py functions quite similarly to gradient.py or simplex.py. Here
  is an example to almost exactly mimic the optimization method in
  Elaine's older versions of the Python code. Again, here I am just
  using charges from a single Jaguar output file (c2.01.mae), and
  using it to recalculate those charges in MacroModel, while adjusting
  the parameters until convergence is reached.

  $ python parameters.py data/mm3.fld -q --output
        options/parameters.yaml --substr "Diaminosulfone OPT"
  $ python loop.py --ffpath data/mm3.fld -r "--dir data -jq c2.01.mae"
        -c "--dir data -mq c2.01.mae" --default

Code to be reintegrated:
========================
This code has already been written by me, but is not implemented into
the current version yet. My old versions had lots of useful bells and
whistles, but the overall design of the program was sloppy. For this
release, I focused on the overall design/program heirarchy. All these
things should be out very shortly. I just wanted to get this version
online to facilitate discussion.

- Selection of charges to be used as reference or calculated data
  based upon atom number.
- Parameter tethering.
- Hessian (Gaussian and Jaguar).
- Bond lengths.
- Angles.
- Inverse distances.
- Option to automatically scale weights based on Boltzmann contribution
  of the reference structure.
- gradient.py. I have a tournament style algorithm working with my last
  version of the code.
- Hessian inversion.

Some points I tried to emphasize in this version:
=================================================
- Ease in changing parts around. For example, using MM3* vs. Amber 
  should create minimal changes in what the user is required to do.
  The user syntax should be nearly identical.
- Backing up intermediate results when something goes wrong. At Notre
  Dame, we only have so many Schrodinger licenses. When this program
  attempts to do a calculation, but can't get a license, or when this
  program is actively using a license but gets kicked off that license,
  the optimization should still error out, but the results obtained
  thus far are backed up into a pickle. That pickle can very easily be
  reloaded to start where the previous run left off.
- Each piece of the optimization loop (simplex, gradient/NR, genetic)
  can easily be used in the loop or standalone with ease.
- Much, much, much less file reading and writing, which has allowed
  for significant speed gains!
- Everything can be controlled from the command line.
- There is no requirements on what directory data is stored in and 
  minimal requirements on the filenames. Corresponding reference and
  FF calculated data points are matched based on part of the filename
  (see calculate.sort_datum), the data type, and an index corresponding
  to the individual data point (often an atom label).
- Logging is used to make interpretation significantly easier. I will be
  adding command line arguments to change the verbosity level, but for now,
  modify the logging dictionary configuration file in options/logging.yaml.

Code planned for future releases:
=================================
- Everything that I have already written (vide supra) but not added to
  this version yet.
- New Hessian/eigenvalue method that Per-Ola has recently described.
- More fine tuned control of what optimization procedures will be 
  used during a single loop cycle through some sort of argument 
  or input file method. I had a neat idea with how it could be done
  using a simple YAML dictionary file.
- Argument based way to change verbosity and location of logging.yaml.
- Simple way using evaluate.py or calculate.py or something to compare
  the labels produced for the reference and FF data. It would be
  recommended to do this before starting an optimization to check that
  the data is aligning properly if you're not sure what you're doing.
- genetic.py
- Tinker compatibility.
- Amber compatibility.

Design comments:
================

calculate.py
------------
Organizes the data extracted from various files to be used to
evaluate the penalty function.

- Retest all the arguments in various combinations with single and
  multiple structure .mae files.

evaluate.py
-----------
Currently doesn't do much. Right now, just automates some calculation
of data for many force fields. Probably going to merge into
calculate.py.

filetypes.py
------------
For every filetype that calculate.py extracts data from, a class
for that file is defined in this script. That class is largely
responsible for extracting and processing the raw data in the file.
calculate.py sort of just organizes that data.

- This is where many of the changes will occur in the next couple days.
  I already wrote these data extraction methods, but they aren't 
  gracefully encapsulated yet. They will be once they're in
  filetypes.py.
- I need to test how Hessian extraction works using my method when there
  are dummy atoms.

gradient.py
-----------
An optimization method where many gradient based methods are located,
such as Newton-Raphson, SVD, Lagrange dampening factors, etc.

- Might move gradient.calc_derivatives.

loop.py
-------
Continues looping through the desired optimization methods until the 
penalty function stops significantly changing or the maximum number of
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
- Needs conditional statement for whether or not max number of loops is
  defined.

parameters.py
-------------
Defines force fields and parameters. Provides methods to extract them,
work with them, manipulate them, etc. What filetypes.py is to data
files, is what parameters.py is to force field files.

- The import and export FF methods should probably be built into their
  respective classes. Then simply the self.export method can be called
  regardless of which type of FF is actively in use. Will be important
  once Amber and Tinker functionality is added back in.
- Might make sense to move gradient.calc_derivatives into this and have
  it be part of BaseFF.

setup_logging.py
----------------
Contains only a few basics functions to setup the logging system in use.
Can be used to create the default logging.yaml file.

simplex.py
----------
Optimizes parameters using the simplex method.

loop.py gradient.py simplex.py ...
----------------------------------
- Add argument based way to change location of logging.yaml.
- Add argument based way to change verbosity.

