### Requirements
Install python via anaconda 
https://docs.anaconda.com/free/anaconda/install/index.html

### Installation
Then clone this repo (or download the zip file). Unzip the `data.zip` (it was too large for GitHub).

Once inside the project folder, create a new conda eviroment 

```
conda env create -f knotsBM.yml
```

and activate it

```
conda activate knotsBM
```

now you're ready to use the scripts.

### Usage

run the GUI with

```
bokeh serve --show graph_viewer.py --session-token-expiration 9999999  --args pkl/mobm_NA_50_40.pkl
```


The path after `—args` refers to the file containing the MoBM. There are several in the `pkl` folder, with different values of the BM radius and the DBscan epsilon parameter (first and second number, respectively). The `NA` refers to the MoBM on non-alternating knots only.

Depending on the size of the graph, it could take several minutes for the GUI to load. 


In the GUI you can choose the colouring function, selecting the vertices you are interest in either by clicking on them or by using the rectangular selection tool on the right. Once selected, you can save your selection by clicking the green button.

Once saved, you can inspect the original data tables with the `after_selection.ipynb` notebook.
