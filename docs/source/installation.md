# Installation

## Release

The [latest release](https://github.com/bjmorgan/kinisi/releases/latest) of `kinisi` can be installed from [PyPI](https://pypi.org/project/kinisi/) with `pip`:

```console
$ pip install kinisi
```

Or using `conda`/`mamba`: 

```console
$ conda install -c conda-forge kinisi
```

## Bleeding-edge

Alternatively, the latest development build can be installed  [GitHub](https://github.com/bjmorgan/kinisi), which can be installed directly from GitHub with `pip`:

```console
$ pip install git+https://github.com/bjmorgan/kinisi.git
```

Note, that if you already have `kinisi` on your system, you may need to run `pip uninstall kinisi` first to ensure you get the latest version.

## Development 

If you are interesting in modifying the `kinisi` code, you should clone the git repository and install `kinisi` with the `dev` option in editable mode. 

```console
$ git clone https://github.com/bjmorgan/kinisi.git
$ cd kinisi
$ pip install -e '.[dev]'
```

To run the notebooks included in the `docs` directory of the GitHub repository, it is necessary that the `[docs]` installation is performed. 

```console
$ git clone https://github.com/bjmorgan/kinisi.git
$ cd kinisi
$ pip install -e '.[docs]'
```

The documentation can then be built from the `Makefile` as follows (note that `pandoc` needs to be installed, either on `conda`/`mamba` or by following the [online instructions](https://pandoc.org/installing.html)). 

```console
$ cd docs
$ make html 
```
