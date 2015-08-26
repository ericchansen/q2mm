# Q2MM

Q2MM stands for quantum (mechanics) to molecular mechanics, quantum guided molecular mechanics, etc. Q2MM is free software for force field optimization.


## Python Dependencies

Q2MM uses Python 2.7.4. I'm reluctant to use more recent versions of Python because as far as I know, Schrodinger still uses 2.7.4.

The following modules are required, but are included in the standard library for Python 2.7.4

* argparse
* collections
* copy
* itertools
* logging
* mmap
* os
* random
* re
* sqlite3
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

Most Q2MM code can be executed from the command line to get help. Here's an example.

```
python calculate.py -h
```

### Using Schrodinger mm3.fld

Towards the end Schrodinger mm3.fld files, there is a section for specific substructure parameters. Substructures can marked for optimization by adding the word OPT to their name. For example, you could name your substructure New Metal Parameters OPT. Try running

```
python parameters.py -f pathtomm3fld -a -pp
```

to print a list of the parameters that Q2MM identified.

### Running an optimization loop

The loop module uses customized input files to manage the optimization of parameters. You can supply it the input file by simply typing

```
python loop.py someinputfile
```

Here's an example of what the input file could look like.

```
FFLD read somedir/mm3.fld # This is a comment.
PARM somedir/params.txt
RDAT -d somedir -je str_a.mae str_b.mae str_c.mae -je str_d.mae str_e.mae -jb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
CDAT -d somedir -me str_a.mae str_b.mae str_c.mae -me str_d.mae str_e.mae -mb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
COMP -o somedir/opt_start.txt
# Here's another comment.
LOOP 0.15
GRAD
SIMP
END
LOOP 0.05
GRAD
END
FFLD write somedir/mm3.fld
CDAT
COMP -o somedir/opt_end.txt
```

Let's breakdown each line.

```
FFLD read somedir/mm3.fld
```

Read the initial force field.

```
PARM somedir/params.txt
```

Select certain parameters in the force field you just read. Without this, all parameters in the substructre are selected. The format of params.txt looks like

```
1854 1
1854 3
1855 2
1856 1
1856 3
1854 3
1855 2
1857 1

```

The first column refers to the line of the force field file where the parameter is located. The second column is an index refering to the location of the parameter in that line. For Schrodinger and mm3.fld, equilibrium bond lengths are found in column 1, force constants in column 2, dipoles in column 3, etc. See the documentation inside the parameters module for more details. Output like this can also be generated using

```
python parameters.py -f mm3.fld -pt bf af -o params.txt
```

This command would print out the bond and angle force constants in the format described above.

```
RDAT -d somedir -je str_a.mae str_b.mae str_c.mae -je str_d.mae str_e.mae -jb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

This gathers the reference data used throughout the optimization. All of the arguments following RDAT are the same as the arguments used for the calculate module. See that module's help for more information.

```
CDAT -d somedir -me str_a.mae str_b.mae str_c.mae -me str_d.mae str_e.mae -mb str_a.mae str_b.mae str_c.mae str_d.mae str_e.mae
```

Same as above, except this is for the force field data.

```
COMP -o somedir/opt_start.txt
```

Compare the reference data to the force field data and determine the initial objective function score. Write the data and scores out to somedir/opt_start.txt.

```
LOOP 0.15
```

This marks the beginning of an optimization loop. All commands located between this line and the line containg END will be repeated until convergence is reached. The loop stops if the change in the objective function is less than 15% after a complete cycle. In this case, that includes using the gradient and simplex methods.

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
FFLD write somedir/mm3.fld
```

Write the optimized force field parammeters to somedir/mm3.fld.

```
CDAT
```

Calculate the force field data again. If CDAT is used without additional arguments as shown here, then it remembers the previously entered arguments and repeats them. Calculating the force field data again here may seem excessive, but Q2MM often doesn't store all of that data for the sake of saving memory.

```
COMP -o somedir/opt_end.txt
```

Compare the reference and force field data for the optimized force field. Write the data and scores out to somedir/opt_end.txt. At this point, it would be wise to look at mm3.fld and compare it a backup you hopefully made of the initial force field.
