# <center>Q2MM</center>

Q2MM stands for quantum (mechanics) to molecular mechanics or quantum guided
molecular mechanics, depending on what you prefer. Q2MM is open source software
for force field optimization.

---

### <center>Python Dependencies</center>

Q2MM uses Python 2.7.4. I'm reluctant to use more recent versions of Python
because as far as I know, Schrödinger still uses 2.7.4.

The following modules are required, but are included in the standard library for
Python 2.7.4.

* argparse
* collections
* copy
* glob
* itertools
* logging
* mmap
* os
* random
* re
* string
* sqlite3
* subprocess
* sys
* textwrap
* time

These are required, but aren't in the standard library.

* numpy

Required for particular features.

* schrodinger

If you'd like to use Schrödinger features, they recommend running external
Python scripts using

```
$SCHRODINGER/run somepythoncode.py
```

---

### <center>Usage</center>

You can get help for most python Q2MM scripts using the command line. Here's an
example.

```
python calculate.py -h
```

### Setting up the Schrödinger MM3* force field

These force field files are labeled `mm3.fld`. Our custom parameters are stored
in substructures towards the end of theses files. For information on these
substructures, see the MacroModel reference manual.

Substructures are marked for optimization by adding the word "OPT" to their
title. For example, you could name your substructure "New Metal Parameters OPT".
Running

```
python parameters.py -f pathtomm3.fld -a -pp
```

will print a list of the parameters that Q2MM identified. You can redirect the
output parameter list to a file using standard Unix redirection.

```
python parameters.py -f pathtomm3fld -a -pp > params.txt
```

Here's an example of what `params.txt` might look like.

```
1854 1 0.0 inf
1854 2 0.0 inf
1854 3 -inf inf
1855 1 0.0 inf
1855 2 0.0 inf
1856 1 -inf inf
1856 2 -inf inf
1856 3 -inf inf
```

The first column refers to the line of the force field file where the parameter
is located. The second column is an index refering to the location of the
parameter in that line. For Schrödinger and `mm3.fld`, equilibrium bond lengths
are found in column 1, force constants in column 2, dipoles in column 3, etc.
See Schrödinger's documentation and the documentation inside
[`parameters`](q2mm/parameters.py).

The 3rd and 4th column in `params.txt` are optional and specify the allowed
parameter range. These values can be any floating points. Also, "inf" is used to
signify the parameter can go to infinity. If the 3rd and 4th column aren't
included, Q2MM will attempt to identify suitable parameter ranges based upon the
parameter type.

Currently, these parameter ranges are hard walls. In other words, if a step
is made to move outside that wall, the step is simply scaled down to not go too
far. Ideally, we should implement soft walls in the future.

To select only certain types of parameters, use

```
python parameters.py -f mm3.fld -pt bf af
```

This command would print the bond and angle force constants in the format
described above. See the help dialogue for
[`parameters`](q2mm/parameters.py) for more information.

### <center>Running an optimization loop</center>

I would always recommend making a backup of your force field before beginning an
optimization.

The loop module uses customized input files to manage the optimization of
parameters. You can supply the input file using the simple command shown below.

```
python loop.py someinputfile
```

Here's an example of what the input file could look like.

```
DIR somedir
FFLD read mm3.fld # This is a comment.
PARM params.txt
RDAT -d somedir -je str_a.mae str_b.mae str_c.mae -je str_d.mae str_e.mae -jb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
CDAT -d somedir -me str_a.mae str_b.mae str_c.mae -me str_d.mae str_e.mae -mb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
COMP -o opt_start.txt
# Here's another comment.
LOOP 0.15
GRAD
SIMP
END
LOOP 0.05
GRAD
END
FFLD write smm3.fld
CDAT
COMP -o opt_end.txt
```

Let's breakdown each line.

```
DIR somedir
```

This sets the directory where all the data files, `atom.typ`, and `mm3.fld`
files are located. Also, the MacroModel calculations will be run from this
directory, and Q2MM intermediate or temporary files will be written here.

```
FFLD read mm3.fld
```

Read the initial force field.

```
PARM params.txt
```

Select certain parameters from the force field you just read. Without this, all
parameters in the substructre are selected.

```
RDAT -d somedir -je str_a.mae str_b.mae str_c.mae -je str_d.mae str_e.mae -jb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

This gathers the reference data used throughout the optimization. All of the
arguments following `RDAT` are the same as the arguments used for the calculate
module. See
[`calculate`](q2mm/calculate.py) help for more information. Note that the
directory is still included in the command shown above.

```
CDAT -d somedir -me str_a.mae str_b.mae str_c.mae -me str_d.mae str_e.mae -mb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

Same as above, except this is for the force field data.

```
COMP -o opt_start.txt
```

Compare the reference data to the force field data to determine the initial
objective function score. Write the data and scores out to
somedir/opt_start.txt.

```
LOOP 0.15
```

This marks the beginning of an optimization loop. All commands located between
this line and the line containg `END` will be repeated until convergence is
reached. In this case, it loops back and forth between
[`gradient`](q2mm/gradient.py) and
[`simplex`](q2mm/simplex.py)
until the objective function changes by less than 15%.

```
GRAD
```

Use the gradient methods to optimize parameters. See the
[`gradient`](q2mm/gradient.py) module for
more information.

```
SIMP
```

Use the simplex method to optimize parameters. See the
[`simplex`](q2mm/simplex.py) module for more information.

```
END
```

Marks the end of the commands being looped.

```
LOOP 0.05
```

Start another loop, but this time with a stricter convergence of 5% change.
Really, you will hardly ever need to include two separate loops. I'm just
showing you that you can if you want to for whatever reason.

```
GRAD
```

In this loop, we're only using the
[`gradient`](q2mm/gradient.py) method.

```
END
```

Marks the end of the second loop.

```
FFLD write mm3.fld
```

Write the optimized force field parammeters to `somedir/mm3.fld`.

```
CDAT
```

Calculate the force field data again. If `CDAT` is used without additional
arguments, as shown here, then it remembers the previously entered arguments and
repeats them. Calculating the force field data again here may seem excessive,
but I wanted to make sure that the FF data stored in memory was calculated using
the most recent optimized parameters. Better safe than sorry!

```
COMP -o opt_end.txt
```

Compare the reference and force field data for the optimized force field. Note
that the reference data remains the same through this entire command file, so
there's no need for me to use another `RDAT` command. Write the data and scores
out to `somedir/opt_end.txt`. At this point, it would be wise to examine
`mm3.fld`. Check that the parameters seem reasonable, and cross-check them with
the backed up original parameters.


### Changes to default settings

#### Maximum parameters for simplex optimization:
The maximum parameters that are used for simplex optimizations is defaulted to 
10, but can be changed with the max_params command.
```
SIMP max_params=6
```

#### Control of gradient methods:
Five gradient methods are available: least-squared, lagrange, levenberg, 
newton-raphson, and SVD. Changing default settings for these require the use of 
the shortened name (lstsq, legrange, levenberg, newton, and svd) followed by "="
and the settings the user wants to use. True and False commands will turn on and
off the optimizers, respectively. Users can also change the factors, cutoffs, and
radii, that are used by including the setting they want to change followed by the
values seperated by "/" nested in brackets.
```
GRAD lstsq=False newton=True,cutoffs[None],radii[0.01/0.1/2.0] svd=True,factor[0.01/0.1]
```

#### Changing default weights and step sizes:
A user may want to change the weights of certain data, or step sizes during 
differentiation of parameters. This can be accomplished with the keywords WGHT and
STEP followed by the data/parameter type and value.
```
WGHT b 10.0
STEP be 1.0
```


