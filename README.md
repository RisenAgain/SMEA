SOMPY
-----
A Python Library for Self Organizing Map (SOM)

As much as possible, the structure of SOM is similar to `somtoolbox` in Matlab. It has the following functionalities:

1. Only Batch training, which is faster than online training. It has parallel processing option similar to `sklearn` format and it speeds up the training procedure, but it depends on the data size and mainly the size of the SOM grid.I couldn't manage the memory problem and therefore, I recommend single core processing at the moment. But nevertheless, the implementation of the algorithm is carefully done for all those important matrix calculations, such as `scipy` sparse matrix and `numexpr` for calculation of Euclidean distance.
2. PCA (or RandomPCA (default)) initialization, using `sklearn` or random initialization.
3. component plane visualization (different modes).
4. Hitmap.
5. U-Matrix visualization.
6. 1-d or 2-d SOM with only rectangular, planar grid. (works well in comparison with hexagonal shape, when I was checking in Matlab with somtoolbox).
7. Different methods for function approximation and predictions (mostly using Sklearn).


### Dependencies:
SOMPY has the following dependencies:
- numpy
- scipy
- scikit-learn
- numexpr
- matplotlib
- pandas

### Installation:
```Python
python setup.py install
```


Many thanks to @sebastiandev, the library is now standardized in a pythonic tradition. Here ([notebook](http://nbviewer.jupyter.org/gist/sevamoo/ec0eb28229304f4575085397138ba5b1)) you can see some basic examples, showing how to use the library.
But I recommend you to go through the codes. There are several functionalities already implemented, but not documented.

The following examples, need some modifications, since the library has been modified. However, it should be easy to figure out how to use them.

- [Notebook 1](http://bit.ly/1eZvaCM)
- [Notebook 2](http://bit.ly/1DHdLpn)
- [Notebook 3](http://bit.ly/1zfn77s)
- [Notebook 4]( http://vahidmoosavi.com/2014/02/18/a-self-organizing-map-som-package-in-python-sompy/)
- [Notebook 5](http://bit.ly/1ujaD36)

### Citation

There is no published paper about this library. However if possible, please cite the library as follows:

```
Vahid Moosavi and Sebastian Packmann, SOMPY: A Self Organizing Map Library in Python, 2015, available online:https://github.com/sevamoo/SOMPY.
```


For more information, you can contact me via sevamoo@gmail.com or svm@arch.ethz.ch




Thanks a lot. Best Vahid
