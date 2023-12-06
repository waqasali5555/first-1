{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "163825c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 5, 5])"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([5,5,5])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "de9d4fa8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b2feafd2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d2bbca02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1995c44c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 5, 5])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[0:]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b48f78ff",
   "metadata": {},
   "source": [
    "## two dimension array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8a751d2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# list of lists "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f5f73f98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5, 5, 5],\n",
       "       [5, 5, 5],\n",
       "       [5, 5, 5]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b=np.array([[5,5,5],[5,5,5],[5,5,5]])\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2ac0125c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "73f94ede",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8cdedf3d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5f004a0d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 5, 5])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5cd88e41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5, 5, 5],\n",
       "       [5, 5, 5],\n",
       "       [5, 5, 5]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " b[0:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3721e5b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np \n",
    "a=np.array([1,2,3,4,5])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3d07c05d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0.])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b=np.zeros(2)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3c78b363",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 1., 1.])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c=np.ones(3)\n",
    "c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c7282cb2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 1., 1.])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#  create an empty array with 2 element\n",
    "\n",
    "d=np.empty(3)\n",
    "d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d4cb9982",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# with rage of element \n",
    "e=np.arange(6)\n",
    "e"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "732f4075",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\n",
       "       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#specific range of element \n",
    "f=np.arange (1,30)\n",
    "f"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "75d2082d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#continue\n",
    "g=np.arange(2,20,2)\n",
    "g"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "48e1f772",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0. ,  2.5,  5. ,  7.5, 10. ])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#linernly  spaced array\n",
    "h=np.linspace(0,10,num=5)\n",
    "h"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "7e26118a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1], dtype=int8)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#specific data type  array \n",
    "i=np.ones(5, dtype=np.int8)\n",
    "i"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ef83414c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 1., 1.])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "j=np.ones(3, dtype=np.float64)\n",
    "j\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38d17a91",
   "metadata": {},
   "source": [
    "# 2d Array "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3c7b55a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0.]])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros((3,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "55335941",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.ones((5,6))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "eee8ffc7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0.]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.empty((3,4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33e870fa",
   "metadata": {},
   "source": [
    "# 3D Array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e718c91e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  1,  2,  3],\n",
       "        [ 4,  5,  6,  7],\n",
       "        [ 8,  9, 10, 11]],\n",
       "\n",
       "       [[12, 13, 14, 15],\n",
       "        [16, 17, 18, 19],\n",
       "        [20, 21, 22, 23]]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c=np.arange(24).reshape(2,3,4)\n",
    "c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b6cca5d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
