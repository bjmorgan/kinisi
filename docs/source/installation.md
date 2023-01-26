# Installation

## Release

The [latest release](https://github.com/bjmorgan/kinisi/releases/latest) of `kinisi` can be installed from [PyPI](https://pypi.org/project/kinisi/) with `pip`:

```console
$ pip install kinisi
```

## Bleeding-edge

Alternatively, the latest development build can be installed  [Github](https://github.com/bjmorgan/kinisi), which can be installed directly from Github with `pip`:

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