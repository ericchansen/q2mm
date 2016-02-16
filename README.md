# Q2MM

Q2MM stands for quantum (mechanics) to molecular mechanics, quantum guided molecular mechanics, etc. Q2MM is free software for force field optimization.


## Python Dependencies

Q2MM uses Python 2.7.4. I'm reluctant to use more recent versions of Python because as far as I know, Schrodinger still uses 2.7.4.

The following modules are required, but are included in the standard library for Python 2.7.4

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

Required but not in the standard library

* numpy

For particular features

* schrodinger

## Usage

If you'd like to use Schrodinger features, they recommend running external Python scripts using

```
$SCHRODINGER/run somepythoncode.py
```

I will update this when I have more information on how to configure your environment variables such that you can use

```
python somepythoncode.py
```

and still use the Schrodinger modules.

### Running most Q2MM code

You can get help for most python Q2MM scripts using the command line. Here's an example.

```
python calculate.py -h
```

### Using Schrodinger mm3.fld

Towards the end Schrodinger mm3.fld files, there is a section for specific substructure parameters. For information on how to setup a MM3* substructure, see the MacroModel reference manual.

Substructures are marked for optimization by adding the word OPT to their name. For example, you could name your substructure "New Metal Parameters OPT". Running

```
python parameters.py -f pathtomm3fld -a -pp
```

will print a list of the parameters that Q2MM identified. You can redirect the output parameter list to a file using standard Unix redirection,

```
python parameters.py -f pathtomm3fld -a -pp > params.txt
```

The format of params.txt looks like

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

The first column refers to the line of the force field file where the parameter is located. The second column is an index refering to the location of the parameter in that line. For Schrodinger and mm3.fld, equilibrium bond lengths are found in column 1, force constants in column 2, dipoles in column 3, etc. See the documentation inside the parameters module for more details.

The 3rd and 4th column are optional and specify the allowed parameter range. These values can be any floating points. Also, inf is used to signify the parameter can go to infinity. If the 3rd and 4th column aren't included, Q2MM will attempt to identify suitable parameter ranges based upon the parameter type.

To select only certain types of parameters, use

```
python parameters.py -f mm3.fld -pt bf af
```

This command would print the bond and angle force constants in the format described above. See the help dialogue for parameters.py for more information on the available parameter types and commands.

### Running an optimization loop

I would always recommend making a backup of your force field before beginning an optimization.

The loop module uses customized input files to manage the optimization of parameters. You can supply the input file using the simple command

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

This sets the directory where all the data files, atom.typ, and mm3.fld files are located. Also, the MacroModel calculations will be run from this directory, and Q2MM intermediate or temporary files will be written here.

```
FFLD read mm3.fld
```

Read the initial force field.

```
PARM params.txt
```

Select certain parameters in the force field you just read. Without this, all parameters in the substructre are selected.

```
RDAT -d somedir -je str_a.mae str_b.mae str_c.mae -je str_d.mae str_e.mae -jb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

This gathers the reference data used throughout the optimization. All of the arguments following RDAT are the same as the arguments used for the calculate module. See the calculate module's help for more information. Note that the directory is still included.

```
CDAT -d somedir -me str_a.mae str_b.mae str_c.mae -me str_d.mae str_e.mae -mb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

Same as above, except this is for the force field data.

```
COMP -o opt_start.txt
```

Compare the reference data to the force field data and determine the initial objective function score. Write the data and scores out to somedir/opt_start.txt.

```
LOOP 0.15
```

This marks the beginning of an optimization loop. All commands located between this line and the line containg END will be repeated until convergence is reached. In this case, it loops back and forth between gradient and simplex methods until the objective function changes by less than 15%.

```
GRAD
```
Use the gradient methods to optimize parameters. See the gradient module for more information.

```
SIMP
```

Use the simplex method to optimize parameters. See the simplex module for more information.

```
END
```

Marks the end of the commands being looped.

```
LOOP 0.05
```

Start another loop, but this time with a stricter convergence of 5% change.

```
GRAD
```

Just use the gradient optimization methods in this loop.

```
END
```

Marks the end of the second loop.

```
FFLD write mm3.fld
```

Write the optimized force field parammeters to somedir/mm3.fld.

```
CDAT
```

Calculate the force field data again. If CDAT is used without additional arguments as shown here, then it remembers the previously entered arguments and repeats them. Calculating the force field data again here may seem excessive, but I wanted to make sure that the FF data stored in memory was calculated using the most recent optimized parameters.

```
COMP -o opt_end.txt
```

Compare the reference and force field data for the optimized force field. Write the data and scores out to somedir/opt_end.txt. At this point, it would be wise to examine mm3.fld. Check that the parameters seem reasonable, and cross-check them with the backed up original parameters.
