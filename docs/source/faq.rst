FAQ
===

- I have a *very* large simulation trajectory, and :py:mod:`kinisi` is taking too long. How can I speed things up?

    There are currently two easy ways to make your analysis run faster. The first is to pass the :py:attr:`parser_params` with a key :py:attr:`'sub_sample_traj'` which is an integer. By default this is :py:attr:`1` and every simulation step is used. However, if :py:attr:`2` is given, every second simulation step will be used, etc. A similar integer can be passed with the :py:attr:`bootstrap_params` which will perform the same style of subsampling on the :py:attr:`dt` values, this has the key :py:attr:`'sub_sample_dt'`.
